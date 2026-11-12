"""Phase 1 of finishing the candidate expansion.
New best S1 per expanded (weak) record; for records whose best S1 CHANGED and whose new pair is uncertain,
export raw text so the cross-encoders can score them (unchanged pairs reuse existing scores).
Writes ext/{split}_best_new.parquet (q_rid, s1_rid, p, margin) and ext/{split}_ce_rows.parquet."""
import sys
import polars as pl
from anyascii import anyascii

sys.path.insert(0, "src")
import config
from model import assign
from prep import load_records
from run_pipeline import qfold

W = config.WORK_DIR
base = {"train": W / "ce/train_best_v6.parquet", "test": None}
for split in ["test", "train"]:
    new = assign(pl.read_parquet(W / f"ext/{split}_ext_scored.parquet"))
    old = pl.read_parquet(base[split]) if base[split] else assign(pl.read_parquet(W / "test_scored.parquet"))
    cmp_ = new.join(old.select("q_rid", pl.col("s1_rid").alias("old_s1"), pl.col("p").alias("old_p")), on="q_rid", how="left")
    changed = cmp_.filter(pl.col("s1_rid") != pl.col("old_s1"))
    need = changed.filter(pl.col("p").is_between(0.02, 0.98)).with_columns(qfold().alias("fold"))
    print(f"[phase1] {split}: expanded records {new.height:,}; best S1 changed {changed.height:,} "
          f"(new rank>10: see brank); changed & uncertain -> CE: {need.height:,}", flush=True)
    new.write_parquet(W / f"ext/{split}_best_new.parquet")
    ids = load_records(split, ["rid", "entity_id"]).with_columns(pl.col("rid").cast(pl.UInt32))
    rids = pl.concat([need["q_rid"], need["s1_rid"]]).unique()
    ids = ids.filter(pl.col("rid").is_in(rids.implode()))
    raw = pl.concat([pl.read_csv(config.DATA_DIR / f"{split}/{split}_source{s}.tsv", separator="\t", quote_char=None,
                                 infer_schema=False).join(ids, on="entity_id") for s in (1, 2, 3)])
    txt = raw.select("rid", pl.concat_str([pl.col("business_name").fill_null(""), pl.lit(" | "), pl.col("business_address").fill_null("")])
                     .map_elements(anyascii, return_dtype=pl.String).alias("txt"))
    rows = (need.select("q_rid", "s1_rid", "fold").join(txt.rename({"rid": "s1_rid", "txt": "t1"}), on="s1_rid")
                .join(txt.rename({"rid": "q_rid", "txt": "tq"}), on="q_rid"))
    rows.write_parquet(W / f"ext/{split}_ce_rows.parquet")
    print(f"[phase1] {split}: wrote {rows.height:,} CE rows", flush=True)
