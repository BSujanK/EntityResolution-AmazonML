"""Phase 3: evaluate (train, held-out S1) and build (test) the candidate-expansion variant of v9.
Only records whose best S1 CHANGED after expansion get new inputs (new p1, new features, new CE scores);
every other record keeps its exact v9 inputs, so differences come from the expansion alone.
    python src/ext_phase3.py eval
    python src/ext_phase3.py build <out_dir> [france_extra_factor]"""
import json, os, sys
import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "src")
import config, prep
from model import assign, em_prior, macro_f05, select_sets
from run_pipeline import write_outputs
from stage2 import apply_stage2, group_features, train_stage2

W = config.WORK_DIR
COLS = json.loads((W / "ce/stacker3_cols.json").read_text())
STACK = lgb.Booster(model_file=str(W / "ce/stacker3.txt"))
logit = lambda c: (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()


def changed_stacked(split):
    """ps for changed & uncertain new-best pairs: stacker3(lp1, margin, lce(v8), lce7(v7), 72 features)."""
    new = pl.read_parquet(W / f"ext/{split}_best_new.parquet")
    feats = pl.read_parquet(W / f"ext/{split}_ext_feats.parquet")
    ce = {m: pl.read_parquet(W / f"ext/{split}_ce_{m}.parquet") for m in ("v8", "v7")}
    d = (new.join(feats, on=["q_rid", "s1_rid"]).join(ce["v8"].rename({"ce": "ce8"}), on=["q_rid", "s1_rid"])
            .join(ce["v7"].rename({"ce": "ce7"}), on=["q_rid", "s1_rid"])
            .with_columns(logit("p").alias("lp1"), logit("ce8").alias("lce"), logit("ce7").alias("lce7")))
    d = d.with_columns(pl.Series("ps", STACK.predict(d.select(COLS).to_numpy()).astype(np.float32)))
    return new, d.select("q_rid", "s1_rid", "ps")


def merged_best(old_best, split):
    """old v9 best table (q_rid, s1_rid, p, margin[, ps]) with changed records replaced."""
    new, ps_changed = changed_stacked(split)
    ch = new.join(old_best.select("q_rid", pl.col("s1_rid").alias("o")), on="q_rid").filter(pl.col("s1_rid") != pl.col("o")).drop("o")
    ch = ch.join(ps_changed, on=["q_rid", "s1_rid"], how="left").with_columns(pl.col("ps").fill_null(pl.col("p")))
    keep = old_best.join(ch.select("q_rid"), on="q_rid", how="anti")
    print(f"[phase3] {split}: records whose best S1 changed: {ch.height:,}")
    return pl.concat([keep, ch.select(keep.columns)], how="vertical_relaxed")


def decide(best, p2, ctry, seen, extra=None, em=True, pri=None, stage1_best=None):
    d = (best.join(p2.rename({"p": "p2"}), on=["q_rid", "s1_rid"], how="left").with_columns(pl.col("p2").fill_null(pl.col("ps")))
             .join(ctry, on="s1_rid", how="left"))
    d = d.with_columns(pl.when(pl.col("country").is_in(seen) & (pl.col("p2") >= 0.1)).then(pl.col("ps"))
                       .otherwise(pl.min_horizontal("ps", "p2")).alias("p"))
    if em:
        b1 = stage1_best.join(ctry, on="s1_rid").filter(pl.col("p") >= 0.02)
        fac = {}
        for c in ctry["country"].unique().to_list():
            pt = pri.get(c, pri["*"]); pi = em_prior(b1.filter(pl.col("country") == c)["p"].to_numpy(), pt)
            fac[c] = (pi / (1 - pi)) / (pt / (1 - pt)) * (extra or {}).get(c, 1.0)
        print("[phase3] EM factors:", {k: round(v, 3) for k, v in fac.items()})
        f = pl.col("country").replace_strict(fac, default=1.0, return_dtype=pl.Float64)
        odds = pl.col("p") / (1 - pl.col("p")).clip(1e-12)
        d = d.with_columns((odds * f / (1 + odds * f)).cast(pl.Float32).alias("p"))
    return select_sets(d.select("q_rid", "s1_rid", "p")).select("q_rid", "s1_rid")


if sys.argv[1] == "eval":
    truth = pl.read_parquet(W / "train_truth.parquet").with_columns(pl.col("q_rid").cast(pl.UInt32), pl.col("s1_rid").cast(pl.UInt32))
    lab = truth.with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
    s1 = prep.load_records("train", ["rid", "src", "country"]).filter(pl.col("src") == 1).with_columns(pl.col("rid").cast(pl.UInt32))
    ctry = s1.select(pl.col("rid").alias("s1_rid"), "country")
    ev_ids = s1.filter(pl.col("rid").hash(seed=13) % 5 == 0)["rid"]
    tv = truth.filter(pl.col("s1_rid").is_in(ev_ids.implode()))
    old = (pl.read_parquet(W / "ce/train_best_v6.parquet").join(pl.read_parquet(W / "ce/ev_ps_stack3.parquet"), on=["q_rid", "s1_rid"], how="left")
             .with_columns(pl.col("ps").fill_null(pl.col("p"))))
    recs = prep.load_records("train", ["nums", "name_cat", "addr"])
    # held-out stage-2 model exactly as in the v9 evaluation (trained on the other 80% of S1 groups)
    gf0 = group_features(old.drop("ps"), recs).join(lab, on=["q_rid", "s1_rid"], how="left").with_columns(pl.col("y").fill_null(0))
    evm = gf0["s1_rid"].hash(seed=13) % 5 == 0
    g_tr = gf0.filter(~evm); es = g_tr["s1_rid"].hash(seed=17) % 10 == 0
    b2 = train_stage2(g_tr.filter(~es), g_tr.filter(~es)["y"].to_numpy(), g_tr.filter(es), g_tr.filter(es)["y"].to_numpy())
    for name, best in [("v9 (no expansion)", old), ("v9 + candidate expansion", merged_best(old, "train"))]:
        be = best.filter(pl.col("s1_rid").is_in(ev_ids.implode()))
        p2 = apply_stage2(b2, group_features(best.drop("ps"), recs).filter(pl.col("s1_rid").is_in(ev_ids.implode())))
        sel = decide(be, p2, ctry, ["US", "India"], em=False)
        line = f"{name}: all {macro_f05(sel, tv, ev_ids):.5f}"
        for c in ["US", "India"]:
            cid = s1.filter((pl.col("country") == c) & pl.col("rid").is_in(ev_ids.implode()))["rid"]
            line += f"  {c} {macro_f05(sel, tv.filter(pl.col('s1_rid').is_in(cid.implode())), cid):.5f}"
        print(line, flush=True)
else:
    out_dir = sys.argv[2]
    extra = None
    if len(sys.argv) > 3:
        extra = json.loads(sys.argv[3]) if sys.argv[3].strip().startswith("{") else {"France": float(sys.argv[3])}
    fp = pl.read_parquet(W / "ce/test_final_p.parquet")  # not used for decisions; kept for reference
    stage1 = assign(pl.read_parquet(W / "test_scored.parquet"))
    # v9 per-record stacked scores: rebuild exactly as ce_predict does for unchanged records
    import importlib.util
    spec = importlib.util.spec_from_file_location("cp", "src/ce_predict.py"); cp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cp)
    ce = cp_ce = None
    def ce_mean(paths, name):
        a, b = (pl.read_parquet(W / x) for x in paths)
        return a.rename({"ce": "a"}).join(b.rename({"ce": "b"}), on=["q_rid", "s1_rid"]).select("q_rid", "s1_rid", ((pl.col("a") + pl.col("b")) / 2).alias(name))
    ce = ce_mean(cp.CE_V8, "ce").join(ce_mean(cp.CE_V7, "ce7"), on=["q_rid", "s1_rid"])
    unc = (stage1.join(ce, on=["q_rid", "s1_rid"]).join(pl.read_parquet(W / "ce/test_uncertain_feats.parquet"), on=["q_rid", "s1_rid"])
                 .with_columns(logit("p").alias("lp1"), logit("ce").alias("lce"), logit("ce7").alias("lce7")))
    unc = unc.with_columns(pl.Series("ps", STACK.predict(unc.select(COLS).to_numpy()).astype(np.float32)))
    old = stage1.join(unc.select("q_rid", "s1_rid", "ps"), on=["q_rid", "s1_rid"], how="left").with_columns(pl.col("ps").fill_null(pl.col("p")))
    best = merged_best(old, "test")
    recs = prep.load_records("test", ["nums", "name_cat", "addr"])
    gf = group_features(best.drop("ps"), recs)
    p2 = apply_stage2(lgb.Booster(model_file=str(W / "model_stage2.txt")), gf)
    ctry = prep.load_records("test", ["rid", "country"]).select(pl.col("rid").cast(pl.UInt32).alias("s1_rid"), "country")
    pri = json.loads((W / "train_priors.json").read_text())
    sel = decide(best, p2, ctry, ["US", "India"], extra=extra, em=True, pri=pri, stage1_best=stage1)
    config.OUTPUT_DIR = __import__("pathlib").Path(out_dir)
    import run_pipeline; run_pipeline.config.OUTPUT_DIR = config.OUTPUT_DIR
    write_outputs(gf.select("q_rid", "s1_rid"), sel)
    print(f"[phase3] built {out_dir}: {sel.height:,} matches", flush=True)
