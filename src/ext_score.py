"""Re-score weak records over their EXPANDED candidate groups (old top-10 + new rank 11-30) with stage 1.
Memory-safe + resumable: groups are written to disk once, then streamed in q_rid-range chunks; every chunk's
results are written as a part file immediately (existing parts are skipped on restart).
train: out-of-fold scores (fold models), restricted to records that can touch held-out S1 entities.
test : mean of both fold models.
Outputs {split}_ext_scored.parquet (q_rid, s1_rid, p) and {split}_ext_feats.parquet (72 features of each
record's new best pair when uncertain)."""
import gc, sys, time
import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "src")
import config
from blocking import candidates_path
from features import FEAT_REC_COLS, FEATURES, RecordStore, pair_features
from prep import load_records
from run_pipeline import qfold

W = config.WORK_DIR
boosters = [lgb.Booster(model_file=str(W / f"model_fold{f}.txt")) for f in (0, 1)]
pred = lambda b, f: b.predict(f.select(FEATURES).to_numpy(), num_threads=0).astype(np.float32)
STEP = 150_000
import os
TAG = os.environ.get("EXT_TAG", "")

for split in ["test", "train"]:
    out = W / f"ext/{split}_ext{TAG}_scored.parquet"
    if out.exists():
        print(split, "exists"); continue
    t0 = time.time()
    parts = W / f"ext/{split}_parts{TAG}"; parts.mkdir(exist_ok=True)
    gpath = W / f"ext/{split}_groups{TAG}.parquet"
    if not gpath.exists():
        ext = pl.scan_parquet(W / f"ext/{split}_cand_ext{TAG}.parquet")
        old_all = pl.scan_parquet(candidates_path(split))
        qs = ext.select("q_rid").unique()
        if split == "train":   # only records that can attach to a held-out S1 (hash%5==0)
            is_ev = lambda c: (pl.col(c).hash(seed=13) % 5) == 0
            touch = pl.concat([ext.filter(is_ev("s1_rid")).select("q_rid"), old_all.filter(is_ev("s1_rid")).select("q_rid")]).unique()
            qs = qs.join(touch, on="q_rid", how="semi")
        top1 = old_all.filter(pl.col("brank") == 1).group_by("s1_rid").len("s1_top1_cnt")
        cols = old_all.collect_schema().names()
        groups = (pl.concat([old_all.join(qs, on="q_rid", how="semi"), ext.join(qs, on="q_rid", how="semi").select(cols)])
                    .unique(["q_rid", "s1_rid"]).join(top1, on="s1_rid", how="left")
                    .with_columns(pl.col("s1_top1_cnt").fill_null(0).cast(pl.UInt32)).sort("q_rid"))
        groups.sink_parquet(gpath)
        print(f"[ext-score] {split}: groups written ({time.time() - t0:.0f}s)", flush=True)
    uq = pl.scan_parquet(gpath).select("q_rid").unique().sort("q_rid").collect()["q_rid"]
    print(f"[ext-score] {split}: {len(uq):,} records", flush=True)
    store = RecordStore(load_records(split, FEAT_REC_COLS))
    for k, i in enumerate(range(0, len(uq), STEP)):
        ps, pf = parts / f"scored_{k:04d}.parquet", parts / f"feats_{k:04d}.parquet"
        if ps.exists() and pf.exists():
            continue
        lo, hi = uq[i], uq[min(i + STEP, len(uq)) - 1]
        part = pl.scan_parquet(gpath).filter(pl.col("q_rid").is_between(lo, hi)).collect()
        f = pair_features(store, part)
        del part
        if split == "train":
            fold = f.select(qfold()).to_series().to_numpy()
            p = np.empty(f.height, dtype=np.float32)
            for kk in (0, 1):
                m = fold == kk
                if m.any():
                    p[m] = pred(boosters[1 - kk], f.filter(pl.Series(m)))
        else:
            p = (pred(boosters[0], f) + pred(boosters[1], f)) / 2
        f = f.with_columns(pl.Series("p", p))
        best = f.sort("p", descending=True).group_by("q_rid", maintain_order=True).head(1)
        best.filter(pl.col("p").is_between(0.02, 0.98)).drop("p").write_parquet(pf)
        f.select("q_rid", "s1_rid", "p").write_parquet(ps)
        del f, best; gc.collect()
        print(f"[ext-score] {split}: {min(i + STEP, len(uq)):,}/{len(uq):,} records ({time.time() - t0:.0f}s)", flush=True)
    del store; gc.collect()
    pl.scan_parquet(parts / "scored_*.parquet").sink_parquet(out)
    pl.scan_parquet(parts / "feats_*.parquet").sink_parquet(W / f"ext/{split}_ext{TAG}_feats.parquet")
    print(f"[ext-score] {split} done in {time.time() - t0:.0f}s", flush=True)
