"""Compute the 72 stage-1 pair features for the uncertain (record, best S1) pairs only.
Whole candidate groups are streamed so query-context features (gaps to rivals) stay correct."""
import sys
import polars as pl

sys.path.insert(0, "src")
import config
from run_pipeline import stream

W = config.WORK_DIR
jobs = {"train": W / "ce/rows_oof.parquet", "test": W / "ce/test_uncertain.parquet"}
for split, rows_path in jobs.items():
    out = W / f"ce/{split}_uncertain_feats.parquet"
    if out.exists():
        print(split, "exists"); continue
    rows = pl.read_parquet(rows_path, columns=["q_rid", "s1_rid"])
    parts = stream(split, lambda f: f.join(rows, on=["q_rid", "s1_rid"], how="semi"), q_keep=rows.select("q_rid").unique())
    df = pl.concat(parts)
    df.write_parquet(out)
    print(split, "features for", df.height, "uncertain pairs", flush=True)
