"""End-to-end entry point.

    python src/run_pipeline.py --data-dir <dataset dir with train/ and test/> [--stage all]

Stages (each caches its result under work/ and is skipped when present):
  prep      normalise records of both splits
  block     candidate generation for both splits
  train     2-fold cross-fitted stage-1 LightGBM pair models (A, B) -> out-of-fold scores for
            every training pair -> stage-2 group-consensus model -> validation report
  predict   score blocking candidates with mean(A, B) (learned pruning), keep each record's best
            S1 (p1 >= 0.02) as the final candidate set, stage-2 veto, per-entity expected-F0.5
            selection, write output/matching_results.tsv and output/candidate_pairs.tsv
"""
import argparse
import gc
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import config


def qfold(col="q_rid"):
    """Cross-fitting fold of an S2/S3 query record (0 or 1)."""
    return (pl.col(col).hash(seed=5) % 2).cast(pl.Int8)


def stream(split, handle, q_keep: pl.DataFrame = None, step=400_000):
    """Stream candidate pairs of `split` by q_rid range (complete candidate groups per
    query), build features and pass each feature frame to `handle`. Returns the list of
    non-None handle results."""
    from features import RecordStore, pair_features, FEAT_REC_COLS
    from prep import load_records
    from blocking import candidates_path

    t0 = time.time()
    cpath = candidates_path(split)
    top1 = (pl.scan_parquet(cpath).filter(pl.col("brank") == 1)
              .group_by("s1_rid").len("s1_top1_cnt").collect())
    lo, hi = (pl.scan_parquet(cpath)
                .select(pl.col("q_rid").min().alias("a"), pl.col("q_rid").max().alias("b")).collect().row(0))
    store = RecordStore(load_records(split, FEAT_REC_COLS))
    out = []
    for a in range(lo, hi + 1, step):
        part = pl.scan_parquet(cpath).filter(pl.col("q_rid").is_between(a, a + step - 1)).collect()
        if q_keep is not None:
            part = part.join(q_keep, on="q_rid", how="semi")
        if part.height:
            part = part.join(top1, on="s1_rid", how="left").with_columns(pl.col("s1_top1_cnt").fill_null(0))
            r = handle(pair_features(store, part))
            if r is not None:
                out.append(r)
        print(f"[stream:{split}] q_rid up to {min(a + step - 1, hi):,}/{hi:,} ({time.time() - t0:.0f}s)", flush=True)
    del store
    gc.collect()
    return out


def _predict(booster, feats):
    from features import FEATURES
    return booster.predict(feats.select(FEATURES).to_numpy(), num_threads=0).astype(np.float32)


def stage_train():
    from features import FEATURES
    from model import train_model, assign, tune_threshold, macro_f05, select_sets
    from prep import load_records, load_truth
    from blocking import candidates_path
    from stage2 import group_features, train_stage2, apply_stage2

    t0 = time.time()
    W = config.WORK_DIR
    truth = load_truth().with_columns(pl.col("q_rid").cast(pl.UInt32), pl.col("s1_rid").cast(pl.UInt32))
    labels = truth.with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))

    # ---- 1. features for a sample of each query fold
    feat_dir = W / "feat_cf"
    if not feat_dir.exists():
        qs = (pl.scan_parquet(candidates_path("train")).select("q_rid").unique().collect()
                .with_columns(qfold().alias("fold")))
        sample = pl.concat([qs.filter(pl.col("fold") == f).sample(n=config.TRAIN_QUERIES_PER_FOLD, seed=config.SEED + f)
                            for f in (0, 1)]).select("q_rid")
        feat_dir.mkdir()
        counter = iter(range(10_000))
        stream("train", lambda f: f.write_parquet(feat_dir / f"part_{next(counter):04d}.parquet"), q_keep=sample)

    # ---- 2. one stage-1 model per fold (trained on that fold's queries only)
    boosters = {}
    for f in (0, 1):
        path = W / f"model_fold{f}.txt"
        if path.exists():
            boosters[f] = lgb.Booster(model_file=str(path))
            continue
        tr = (pl.scan_parquet(feat_dir / "*.parquet").filter(qfold() == f).collect()
                .join(labels, on=["q_rid", "s1_rid"], how="left").with_columns(pl.col("y").fill_null(0)))
        print(f"[train] fold {f}: {tr.height:,} pairs, {tr['y'].sum():,} positives")
        es = (tr["q_rid"].hash(seed=3) % 10 == 0).to_numpy()
        x, y = tr.select(FEATURES).to_numpy(), tr["y"].to_numpy()
        del tr
        boosters[f] = train_model(x[~es], y[~es], x[es], y[es])
        boosters[f].save_model(str(path))
        del x, y
        gc.collect()

    # ---- 3. out-of-fold scores for every training candidate pair
    oof_path = W / "train_oof_scored.parquet"
    if not oof_path.exists():
        def handle(feats):
            fold = feats.select(qfold()).to_series().to_numpy()
            p = np.empty(feats.height, dtype=np.float32)
            for f in (0, 1):   # score fold-f queries with the model trained on the other fold
                m = fold == f
                if m.any():
                    p[m] = _predict(boosters[1 - f], feats.filter(pl.Series(m)))
            return feats.select("q_rid", "s1_rid").with_columns(pl.Series("p", p))
        pl.concat(stream("train", handle)).write_parquet(oof_path)
    scored = pl.read_parquet(oof_path)

    # ---- 4. stage-1 evaluation on all training S1 entities
    best = assign(scored)
    del scored
    gc.collect()
    s1_all = load_records("train", ["rid", "src"]).filter(pl.col("src") == 1)["rid"].cast(pl.UInt32)
    thr, f_thr, curve = tune_threshold(best, truth, s1_all)
    f_ef = macro_f05(select_sets(best).select("q_rid", "s1_rid"), truth, s1_all)
    print(f"[train] stage 1 OOF over {len(s1_all):,} S1: flat thr {thr:.2f} -> {f_thr:.4f}; expected-F -> {f_ef:.4f}")

    # training positive rate among best attachments, per country (for label-shift correction)
    save_train_priors(best, truth)

    # ---- 5. stage 2 (group consensus) on OOF groups; honest eval on held-out S1 fold
    gf = group_features(best, load_records("train", ["nums", "name_cat", "addr"]))
    gf = gf.join(labels, on=["q_rid", "s1_rid"], how="left").with_columns(pl.col("y").fill_null(0))
    ev_mask = (gf["s1_rid"].hash(seed=13) % 5 == 0)
    g_tr, g_ev = gf.filter(~ev_mask), gf.filter(ev_mask)
    del gf
    s1_ev = s1_all.filter(s1_all.hash(seed=13) % 5 == 0)
    tr_es = (g_tr["s1_rid"].hash(seed=17) % 10 == 0)
    b2 = train_stage2(g_tr.filter(~tr_es), g_tr.filter(~tr_es)["y"].to_numpy(),
                      g_tr.filter(tr_es), g_tr.filter(tr_es)["y"].to_numpy())
    truth_ev = truth.filter(pl.col("s1_rid").is_in(s1_ev.implode()))
    f1 = macro_f05(select_sets(best.filter(pl.col("s1_rid").is_in(s1_ev.implode()))).select("q_rid", "s1_rid"), truth_ev, s1_ev)
    best_ev = best.filter(pl.col("s1_rid").is_in(s1_ev.implode()))
    f2 = macro_f05(select_sets(demote(best_ev, apply_stage2(b2, g_ev))).select("q_rid", "s1_rid"), truth_ev, s1_ev)
    print(f"[train] held-out 20% S1 ({len(s1_ev):,}): stage 1 {f1:.4f} -> stage 1 + stage-2 veto {f2:.4f}")
    b2 = train_stage2(pl.concat([g_tr, g_ev]), pl.concat([g_tr, g_ev])["y"].to_numpy(),
                      rounds=int(b2.best_iteration * 1.1) + 1)
    b2.save_model(str(W / "model_stage2.txt"))

    imp = sorted(zip(FEATURES, boosters[0].feature_importance("gain")), key=lambda r: -r[1])
    report = {"stage1_oof_expected_f": f_ef, "stage1_oof_flat_threshold": f_thr, "threshold": thr,
              "heldout_stage1": f1, "heldout_stage1_2": f2, "n_s1": len(s1_all), "n_s1_heldout": len(s1_ev),
              "curve": curve, "feature_gain": [(k, float(v)) for k, v in imp]}
    (W / "val_report.json").write_text(json.dumps(report, indent=1))
    print(f"[train] done in {time.time() - t0:.0f}s")


def save_train_priors(best, truth):
    from prep import load_records
    from stage2 import FLOOR
    ctry = load_records("train", ["rid", "country"]).select(pl.col("rid").cast(pl.UInt32).alias("s1_rid"), "country")
    b = (best.filter(pl.col("p") >= FLOOR).join(ctry, on="s1_rid")
             .join(truth.with_columns(pl.lit(1).alias("y")), on=["q_rid", "s1_rid"], how="left")
             .with_columns(pl.col("y").fill_null(0)))
    pri = {c: float(v) for c, v in b.group_by("country").agg(pl.col("y").mean()).iter_rows()}
    pri["*"] = float(b["y"].mean())
    (config.WORK_DIR / "train_priors.json").write_text(json.dumps(pri, indent=1))
    print(f"[train] positive rate of best attachments per country: {pri}")
    return pri


def stage_predict():
    from model import assign, select_sets, prior_adjust
    from prep import load_records
    from blocking import candidates_path
    from stage2 import group_features, apply_stage2

    t0 = time.time()
    W = config.WORK_DIR
    scored_path = W / "test_scored.parquet"
    if not scored_path.exists():
        boosters = [lgb.Booster(model_file=str(W / f"model_fold{f}.txt")) for f in (0, 1)]
        handle = lambda feats: feats.select("q_rid", "s1_rid").with_columns(
            pl.Series("p", (_predict(boosters[0], feats) + _predict(boosters[1], feats)) / 2))
        pl.concat(stream("test", handle)).write_parquet(scored_path)
    scored = pl.read_parquet(scored_path)
    b2 = lgb.Booster(model_file=str(W / "model_stage2.txt"))
    best = assign(scored)
    del scored
    gf = group_features(best, load_records("test", ["nums", "name_cat", "addr"]))
    # candidate_pairs.tsv = exactly the pairs the final (stage-2) matcher runs inference on:
    # each S2/S3 record's best S1 candidate after the learned stage-1 pruning (p1 >= FLOOR)
    final_cand = gf.select("q_rid", "s1_rid")
    final = demote(best, apply_stage2(b2, gf))
    # label-shift correction: test has a higher share of look-alike records than train
    pri_path = W / "train_priors.json"
    if not pri_path.exists():
        from prep import load_truth
        save_train_priors(assign(pl.read_parquet(W / "train_oof_scored.parquet")),
                          load_truth().with_columns(pl.col("q_rid").cast(pl.UInt32), pl.col("s1_rid").cast(pl.UInt32)))
    ctry = load_records("test", ["rid", "country"]).select(pl.col("rid").cast(pl.UInt32).alias("s1_rid"), "country")
    final = prior_adjust(final, ctry, json.loads(pri_path.read_text()))
    sel = select_sets(final)
    del gf
    gc.collect()
    write_outputs(final_cand, sel)
    print(f"[predict] {final_cand.height:,} candidate pairs, {sel.height:,} matches written in {time.time() - t0:.0f}s")


def demote(best: pl.DataFrame, p2: pl.DataFrame) -> pl.DataFrame:
    """Final score = min(stage-1 p, stage-2 p): the group model may veto (twins, look-alike
    clusters) but never promotes a record the pair model finds unlikely. Promotion by group
    features did not transfer to the test distribution (more look-alike clusters)."""
    return (best.join(p2.rename({"p": "p2"}), on=["q_rid", "s1_rid"])
                .select("q_rid", "s1_rid", pl.min_horizontal("p", "p2").alias("p")))


def write_outputs(cand: pl.DataFrame, best: pl.DataFrame):
    from prep import load_records
    ids = load_records("test", ["rid", "entity_id", "src"]).with_columns(pl.col("rid").cast(pl.UInt32))
    s1 = ids.filter(pl.col("src") == 1).select(pl.col("rid").alias("s1_rid"), pl.col("entity_id").alias("source1_entity_id"))
    qid = ids.filter(pl.col("src") != 1).select(pl.col("rid").alias("q_rid"), pl.col("entity_id").alias("eid"))
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def dump(pairs, col, name):
        lists = (pairs.join(qid, on="q_rid").sort("eid")
                      .group_by("s1_rid").agg(pl.col("eid").unique(maintain_order=True).str.join(",").alias(col)))
        out = s1.join(lists, on="s1_rid", how="left").with_columns(pl.col(col).fill_null(""))
        out = out.select("source1_entity_id", col)
        with open(config.OUTPUT_DIR / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"source1_entity_id\t{col}\n")
            for a, b in out.iter_rows():
                fh.write(f"{a}\t{b}\n")
        print(f"[output] {name}: {out.height:,} rows")

    dump(cand.select("q_rid", "s1_rid"), "candidate_entity_ids", "candidate_pairs.tsv")
    dump(best.select("q_rid", "s1_rid"), "matched_entity_ids", "matching_results.tsv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, help="dataset directory containing train/ and test/")
    ap.add_argument("--stage", default="all", choices=["all", "prep", "block", "train", "predict"])
    args = ap.parse_args()
    if args.data_dir:
        config.set_data_dir(args.data_dir)
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[config] data={config.DATA_DIR} work={config.WORK_DIR} output={config.OUTPUT_DIR}")

    if args.stage in ("all", "prep"):
        from prep import prepare, load_truth
        prepare("train"); prepare("test"); load_truth()
    if args.stage in ("all", "block"):
        from blocking import build_candidates
        build_candidates("train"); build_candidates("test")
    if args.stage in ("all", "train"):
        stage_train()
    if args.stage in ("all", "predict"):
        stage_predict()


if __name__ == "__main__":
    main()
