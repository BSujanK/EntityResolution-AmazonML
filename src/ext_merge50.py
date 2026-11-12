"""Merge: India weak records take the ranks 11-50 expansion; everyone else keeps the ranks 11-30 expansion."""
import sys, shutil
import polars as pl
sys.path.insert(0, "src")
import config
W = config.WORK_DIR
for split in ["test", "train"]:
    for kind in ["scored", "feats"]:
        base, new = W / f"ext/{split}_ext_{kind}.parquet", W / f"ext/{split}_ext50_{kind}.parquet"
        bak = W / f"ext/{split}_ext_{kind}_v10b.parquet"
        if not bak.exists():
            shutil.copy(base, bak)
        b, n = pl.read_parquet(bak), pl.read_parquet(new)
        merged = pl.concat([b.join(n.select("q_rid").unique(), on="q_rid", how="anti"), n], how="vertical_relaxed")
        merged.write_parquet(base)
        print(split, kind, "v10b rows", b.height, "-> merged", merged.height, flush=True)
