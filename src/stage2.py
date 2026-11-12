"""Stage 4: entity-group consensus re-scoring.

After stage 1 each S2/S3 record points at its best S1 entity with probability p.
An S1 entity therefore collects a *group* of records. True matches of one business
tend to agree with each other (same house number, same name), while near-copy
distractors (same name, perturbed house number) disagree with the group. This stage
adds group-consensus features and re-scores each (record, S1) assignment with a
small LightGBM model trained on out-of-sample stage-1 scores (the validation fold).
"""
import lightgbm as lgb
import numpy as np
import polars as pl

import config

S2_FEATURES = ["p", "margin", "n_grp", "p_sum_oth", "p_max_oth", "n_conf_oth", "num_same_conf",
               "num_same_frac", "name_same_conf", "name_same_frac", "addr_same_conf", "s1_num_support",
               "rank_in_s1", "p_rel", "num1_eq", "q_has_num", "s1_has_num", "name_eq_s1",
               # twin-entity evidence over the whole argmax group (any probability)
               "g_n_all", "g_num_same", "g_twin_num", "g_s1num_cnt", "g_name_same", "g_addr_same",
               "g_pmax_same_num", "g_psum_s1num", "q_addr_empty"]
S2_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                 feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                 num_threads=0, verbose=-1, seed=config.SEED)
CONF = 0.5
FLOOR = 0.02


def group_features(best: pl.DataFrame, recs: pl.DataFrame) -> pl.DataFrame:
    """best: q_rid, s1_rid, p, margin (one row per query, its argmax S1).
    recs: rid, nums, name_cat, addr (rid dense == row index)."""
    b = best
    qi, si = b["q_rid"].to_numpy().astype(np.int64), b["s1_rid"].to_numpy().astype(np.int64)
    q = recs.select(pl.col("nums").list.first().alias("qn"), pl.col("name_cat").alias("qc"),
                    pl.col("addr").alias("qa"))[qi]
    s = recs.select(pl.col("nums").list.first().alias("sn"), pl.col("name_cat").alias("sc"))[si]
    df = pl.concat([b, q, s], how="horizontal").with_columns((pl.col("p") >= CONF).cast(pl.Int32).alias("conf"))
    g = "s1_rid"
    second = pl.when(pl.len().over(g) > 1).then(pl.col("p").top_k(2).min().over(g)).otherwise(0.0)
    df = df.with_columns(
        pl.len().over(g).alias("n_grp"),
        (pl.col("p").sum().over(g) - pl.col("p")).alias("p_sum_oth"),
        pl.when(pl.col("p") == pl.col("p").max().over(g)).then(second)
          .otherwise(pl.col("p").max().over(g)).alias("p_max_oth"),
        (pl.col("conf").sum().over(g) - pl.col("conf")).alias("n_conf_oth"),
        (pl.col("conf").sum().over(g, "qn") - pl.col("conf")).alias("num_same_conf"),
        (pl.col("conf").sum().over(g, "qc") - pl.col("conf")).alias("name_same_conf"),
        (pl.col("conf").sum().over(g, "qa") - pl.col("conf")).alias("addr_same_conf"),
        ((pl.col("qn") == pl.col("sn")).cast(pl.Int32) * pl.col("conf")).sum().over(g).alias("_sup"),
        pl.col("p").rank("ordinal", descending=True).over(g).alias("rank_in_s1"),
        (pl.col("p") / pl.col("p").max().over(g)).alias("p_rel"),
        (pl.col("qn") == pl.col("sn")).fill_null(False).alias("num1_eq"),
        pl.col("qn").is_not_null().alias("q_has_num"),
        pl.col("sn").is_not_null().alias("s1_has_num"),
        (pl.col("qc") == pl.col("sc")).alias("name_eq_s1"),
    )
    qn_ok = pl.col("qn").is_not_null()
    df = df.with_columns(
        pl.len().over(g).alias("g_n_all"),
        pl.when(qn_ok).then(pl.len().over(g, "qn") - 1).otherwise(0).alias("g_num_same"),
        pl.when(qn_ok & ~pl.col("num1_eq")).then(pl.len().over(g, "qn") - 1).otherwise(0).alias("g_twin_num"),
        ((pl.col("qn") == pl.col("sn")).fill_null(False).cast(pl.Int32).sum().over(g)
         - pl.col("num1_eq").cast(pl.Int32)).alias("g_s1num_cnt"),
        (pl.len().over(g, "qc") - 1).alias("g_name_same"),
        pl.when(pl.col("qa") != "").then(pl.len().over(g, "qa") - 1).otherwise(0).alias("g_addr_same"),
        pl.when(qn_ok & (pl.len().over(g, "qn") > 1))
          .then(pl.when(pl.col("p") == pl.col("p").max().over(g, "qn"))
                  .then(pl.col("p").top_k(2).min().over(g, "qn")).otherwise(pl.col("p").max().over(g, "qn")))
          .otherwise(0.0).alias("g_pmax_same_num"),
        ((pl.col("p") * (pl.col("qn") == pl.col("sn")).fill_null(False).cast(pl.Float32)).sum().over(g)
         - pl.col("p") * pl.col("num1_eq").cast(pl.Float32)).alias("g_psum_s1num"),
        (pl.col("qa") == "").alias("q_addr_empty"),
    )
    df = df.filter(pl.col("p") >= FLOOR)
    df = df.with_columns(
        (pl.col("num_same_conf") / pl.col("n_conf_oth").clip(1)).alias("num_same_frac"),
        (pl.col("name_same_conf") / pl.col("n_conf_oth").clip(1)).alias("name_same_frac"),
        ((pl.col("_sup") - (pl.col("num1_eq").cast(pl.Int32) * pl.col("conf"))) / pl.col("n_conf_oth").clip(1)).alias("s1_num_support"),
    )
    return df.select("q_rid", "s1_rid", *[pl.col(c).cast(pl.Float32) for c in S2_FEATURES])


def train_stage2(gf: pl.DataFrame, y: np.ndarray, gf_val=None, y_val=None, rounds=3000):
    dtr = lgb.Dataset(gf.select(S2_FEATURES).to_numpy(), y, feature_name=S2_FEATURES)
    if gf_val is None:
        return lgb.train(S2_PARAMS, dtr, num_boost_round=rounds)
    dva = lgb.Dataset(gf_val.select(S2_FEATURES).to_numpy(), y_val, reference=dtr)
    return lgb.train(S2_PARAMS, dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])


def apply_stage2(booster, gf: pl.DataFrame) -> pl.DataFrame:
    p2 = booster.predict(gf.select(S2_FEATURES).to_numpy(), num_threads=0).astype(np.float32)
    return gf.select("q_rid", "s1_rid").with_columns(pl.Series("p", p2))
