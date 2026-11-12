"""Candidate expansion: re-run key blocking with top-30 and keep only the new candidates (rank 11-30)
for 'weak' records, i.e. records whose current best stage-1 probability is below 0.5. These are where the
true S1 was most often ranked out of the top 10 (blocking misses). Output: {split}_cand_ext.parquet"""
import sys, time
import polars as pl

sys.path.insert(0, "src")
import config
from blocking import REC_COLS, run_blocking
from model import assign
from prep import load_records

W = config.WORK_DIR
for split, scored in [("test", "test_scored.parquet"), ("train", "train_oof_scored.parquet")]:
    out = W / f"ext/{split}_cand_ext.parquet"
    if out.exists():
        print(split, "exists"); continue
    t = time.time()
    weak = assign(pl.read_parquet(W / scored)).filter(pl.col("p") < 0.5).select("q_rid")
    print(f"[ext] {split}: weak records {weak.height:,}", flush=True)
    recs = load_records(split, REC_COLS)
    cand = run_blocking(split, recs, top_k=30, query_rids=weak["q_rid"])
    ext = cand.filter(pl.col("brank") > 10)
    ext.write_parquet(out)
    print(f"[ext] {split}: {ext.height:,} new candidate pairs (rank 11-30) in {time.time() - t:.0f}s", flush=True)
    del recs, cand, ext
