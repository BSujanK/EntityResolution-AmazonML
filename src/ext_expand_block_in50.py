"""Second expansion: India only, key blocking top-50 for weak records (best p1 < 0.5) -> ranks 11-50."""
import sys, time
import polars as pl
sys.path.insert(0, "src")
import config
from blocking import REC_COLS, run_blocking
from model import assign
from prep import load_records
W = config.WORK_DIR
for split, scored in [("test", "test_scored.parquet"), ("train", "train_oof_scored.parquet")]:
    out = W / f"ext/{split}_cand_ext50.parquet"
    if out.exists():
        print(split, "exists"); continue
    t = time.time()
    recs = load_records(split, REC_COLS)
    india = recs.filter(pl.col("country") == "India").select(pl.col("rid").cast(pl.UInt32).alias("q_rid"))
    weak = assign(pl.read_parquet(W / scored)).filter(pl.col("p") < 0.5).select("q_rid").join(india, on="q_rid", how="semi")
    print(f"[ext50] {split}: weak India records {weak.height:,}", flush=True)
    cand = run_blocking(split, recs, top_k=50, query_rids=weak["q_rid"])
    cand.filter(pl.col("brank") > 10).write_parquet(out)
    print(f"[ext50] {split}: done in {time.time() - t:.0f}s", flush=True)
    del recs, cand
