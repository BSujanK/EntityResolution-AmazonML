"""Export uncertain (0.02 <= p1 <= 0.98) best-attachment pairs with raw text, for the cross-encoders."""
import polars as pl, config, prep
from anyascii import anyascii
from model import assign
from run_pipeline import qfold
W = config.WORK_DIR; OUT = W/"ce"
LO, HI = 0.02, 0.98
def texts(split, rids):
    ids = prep.load_records(split, ["rid","entity_id"]).with_columns(pl.col("rid").cast(pl.UInt32)).filter(pl.col("rid").is_in(rids.implode()))
    raw = pl.concat([pl.read_csv(config.DATA_DIR/f"{split}/{split}_source{s}.tsv", separator="\t", quote_char=None, infer_schema=False).join(ids, on="entity_id") for s in (1,2,3)])
    return raw.select("rid", pl.concat_str([pl.col("business_name").fill_null(""), pl.lit(" | "), pl.col("business_address").fill_null("")])
                      .map_elements(anyascii, return_dtype=pl.String).alias("txt"))
def export(split, scored_path, labels=None):
    b = assign(pl.read_parquet(scored_path)).filter(pl.col("p").is_between(LO, HI)).with_columns(qfold().alias("fold"))
    if labels is not None:
        b = b.join(labels, on=["q_rid","s1_rid"], how="left").with_columns(pl.col("y").fill_null(0))
    t = texts(split, pl.concat([b["q_rid"], b["s1_rid"]]).unique())
    b = b.join(t.rename({"rid":"s1_rid","txt":"t1"}), on="s1_rid").join(t.rename({"rid":"q_rid","txt":"tq"}), on="q_rid")
    b.write_parquet(OUT/f"{split}_uncertain.parquet", compression="zstd")
    print(split, b.height, "pairs", "" if labels is None else f"pos rate {b['y'].mean():.3f}")
truth = pl.read_parquet(W/"train_truth.parquet").with_columns(pl.col("q_rid").cast(pl.UInt32), pl.col("s1_rid").cast(pl.UInt32), pl.lit(1, dtype=pl.Int8).alias("y"))
export("train", W/"train_oof_scored.parquet", truth)
export("test", W/"test_scored.parquet")
