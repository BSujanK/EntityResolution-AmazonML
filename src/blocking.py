"""Stage 1: candidate generation (blocking).

Each record emits several hashed blocking keys, always scoped by country:
  A  pair of name tokens               (robust to word re-ordering / extra words)
  B  name token x address number       (robust to name noise when house no. agrees)
  C  address number x address token    (address-only: transliterated / DBA names)
  D  prefix of concatenated core name  (domains, glued names: "eelegal.com")
  E  prefix of phonetic name skeleton  (typos, transliteration)
  F  name token x address token        (no house number, common name words)
  G  pair of address tokens            (address-only, no house number)
Name tokens are phonetic skeletons (raw token when the skeleton is too short).
Per record only the rarest tokens (document frequency among Source-1 records of the
same country) are used, so key choice is stable under component re-ordering.

Keys shared by more than MAX_S1_PER_KEY Source-1 records are dropped (too generic).
Every (S2/S3 record, S1 record) pair sharing a key is scored by
sum(1/sqrt(block size)); each S2/S3 record keeps its TOP_K best S1 records.
Per-key-type hit counts are kept as model features.
"""
import time

import polars as pl

import config

KEY_TYPES = ["A", "B", "C", "D", "E", "F", "G"]
MAX_NAME_TOKS = 4
MAX_NUMS = 3
MAX_ADDR_TOKS = 5
REC_COLS = ["rid", "src", "country", "core_toks", "skel_toks", "nums", "addr_toks", "name_cat", "name_skel"]


def _hash(*parts):
    return pl.concat_str(list(parts), separator="\x1f").hash(seed=7)


def token_df(s1: pl.DataFrame):
    """Document frequency of blocking tokens among Source-1 records, per country."""
    nt = (_name_tokens(s1).group_by("country", "t").len("df"))
    at = (s1.select("country", pl.col("addr_toks").alias("a")).explode("a").drop_nulls("a")
            .group_by("country", "a").len("df"))
    return nt, at


def _name_tokens(rec):
    return (rec.select("rid", "country", "core_toks", "skel_toks")
               .explode("core_toks", "skel_toks")
               .drop_nulls("core_toks")
               .select("rid", "country",
                       pl.when(pl.col("skel_toks").str.len_chars() >= 3).then(pl.col("skel_toks"))
                         .otherwise(pl.lit("~") + pl.col("core_toks")).alias("t"))
               .unique(["rid", "t"]))


def _rarest(df, dft, col, n):
    return (df.join(dft, on=["country", col], how="left")
              .with_columns(pl.col("df").fill_null(0))
              .sort("rid", "df", col)
              .with_columns(pl.int_range(pl.len()).over("rid").alias("pos"))
              .filter(pl.col("pos") < n)
              .select("rid", "country", col))


def make_keys(rec: pl.DataFrame, ndf, adf) -> pl.DataFrame:
    """-> (rid, key:u64, kt:u8)"""
    toks = _rarest(_name_tokens(rec), ndf, "t", MAX_NAME_TOKS)
    atoks = _rarest(rec.select("rid", "country", pl.col("addr_toks").alias("a")).explode("a").drop_nulls("a")
                       .unique(["rid", "a"]), adf, "a", MAX_ADDR_TOKS)
    nums = (rec.select("rid", "country", "nums").explode("nums").drop_nulls("nums")
               .with_columns(pl.int_range(pl.len()).over("rid").alias("pos"))
               .filter(pl.col("pos") < MAX_NUMS).select("rid", "country", pl.col("nums").alias("n")))

    def emit(df, kt, *cols):
        return df.select("rid", _hash("country", pl.lit(kt), *cols).alias("key"),
                         pl.lit(KEY_TYPES.index(kt), dtype=pl.UInt8).alias("kt"))

    ntok = toks.group_by("rid").len("ntok")
    parts = [
        emit(toks.join(toks.select("rid", pl.col("t").alias("t2")), on="rid")
                 .filter(pl.col("t") < pl.col("t2")), "A", "t", "t2"),
        emit(toks.join(ntok.filter(pl.col("ntok") == 1), on="rid"), "A", "t"),
        emit(toks.join(nums.drop("country"), on="rid"), "B", "t", "n"),
        emit(nums.join(atoks.drop("country"), on="rid"), "C", "n", "a"),
        emit(rec.filter(pl.col("name_cat").str.len_chars() >= 6)
                .with_columns(pl.col("name_cat").str.slice(0, 8).alias("p")), "D", "p"),
        emit(rec.filter(pl.col("name_skel").str.len_chars() >= 5)
                .with_columns(pl.col("name_skel").str.slice(0, 7).alias("p")), "E", "p"),
        emit(toks.join(atoks.drop("country"), on="rid"), "F", "t", "a"),
        emit(atoks.join(atoks.select("rid", pl.col("a").alias("a2")), on="rid")
                  .filter(pl.col("a") < pl.col("a2")), "G", "a", "a2"),
    ]
    return pl.concat(parts).unique(["rid", "key"])


def candidates_path(split):
    return config.WORK_DIR / f"{split}_candidates.parquet"


def run_blocking(split, recs: pl.DataFrame, top_k=config.TOP_K, max_s1=config.MAX_S1_PER_KEY,
                 query_rids=None) -> pl.DataFrame:
    """Return candidate pairs (q_rid, s1_rid, bscore, nA..nG, brank) for all S2/S3 records
    (or only `query_rids` if given)."""
    t0 = time.time()
    s1 = recs.filter(pl.col("src") == 1)
    ndf, adf = token_df(s1)
    s1_keys = pl.concat([make_keys(s1.slice(i, 500_000), ndf, adf).select("key", "rid")
                         for i in range(0, s1.height, 500_000)])
    freq = s1_keys.group_by("key").len("f").filter(pl.col("f") <= max_s1)
    s1_keys = (s1_keys.join(freq, on="key")
                      .select("key", pl.col("rid").cast(pl.UInt32).alias("s1_rid"),
                              (1.0 / pl.col("f").cast(pl.Float32).sqrt()).alias("w")))
    del freq
    print(f"[block] {split}: S1 key index {s1_keys.height:,} rows ({time.time() - t0:.0f}s)")

    q = recs.filter(pl.col("src") != 1)
    if query_rids is not None:
        q = q.filter(pl.col("rid").is_in(query_rids.implode()))
    out = []
    step = config.QUERY_CHUNK // 2
    for start in range(0, q.height, step):
        chunk = q.slice(start, step)
        hits = make_keys(chunk, ndf, adf).join(s1_keys, on="key")
        agg = (hits.group_by("rid", "s1_rid")
                   .agg(pl.col("w").sum().alias("bscore"),
                        *[(pl.col("kt") == i).sum().cast(pl.UInt8).alias(f"n{k}")
                          for i, k in enumerate(KEY_TYPES)]))
        agg = (agg.with_columns(pl.col("bscore").rank("ordinal", descending=True).over("rid")
                                  .cast(pl.UInt16).alias("brank"))
                  .filter(pl.col("brank") <= top_k)
                  .rename({"rid": "q_rid"})
                  .with_columns(pl.col("q_rid").cast(pl.UInt32), pl.col("brank").cast(pl.UInt8)))
        out.append(agg)
        print(f"[block] {split}: queries {start + chunk.height:,}/{q.height:,} "
              f"-> {sum(o.height for o in out):,} pairs ({time.time() - t0:.0f}s)", flush=True)
    return pl.concat(out)


def build_candidates(split):
    out = candidates_path(split)
    if out.exists():
        print(f"[block] {out.name} exists, skipping")
        return
    from prep import load_records
    recs = load_records(split, REC_COLS)
    cand = run_blocking(split, recs)
    cand.write_parquet(out)
    print(f"[block] {split}: {cand.height:,} candidate pairs written")
