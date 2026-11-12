"""Stage 0: normalise all source records of a split and cache them as parquet."""
import time

import polars as pl

import config
from normalize import load_split


def records_path(split):
    return config.WORK_DIR / f"{split}_records.parquet"


def prepare(split):
    out = records_path(split)
    if out.exists():
        print(f"[prep] {out.name} exists, skipping")
        return
    t = time.time()
    df = load_split(config.DATA_DIR, split)
    df = df.drop("business_name", "business_address", "name_toks")
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    print(f"[prep] {split}: {df.height:,} records normalised in {time.time() - t:.0f}s")


def load_records(split, columns=None) -> pl.DataFrame:
    return pl.read_parquet(records_path(split), columns=columns)


def load_truth() -> pl.DataFrame:
    """Ground-truth links as (q_rid, s1_rid) row-id pairs of the train split."""
    out = config.WORK_DIR / "train_truth.parquet"
    if out.exists():
        return pl.read_parquet(out)
    gt = pl.read_csv(config.DATA_DIR / "train" / "train_ground_truth.tsv", separator="\t",
                     quote_char=None, infer_schema=False)
    links = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
               .explode("matched_entity_ids")
               .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != "")))
    ids = load_records("train", ["rid", "entity_id"])
    links = (links.join(ids.rename({"rid": "s1_rid", "entity_id": "source1_entity_id"}), on="source1_entity_id")
                  .join(ids.rename({"rid": "q_rid", "entity_id": "matched_entity_ids"}), on="matched_entity_ids")
                  .select("q_rid", "s1_rid"))
    links.write_parquet(out)
    return links
