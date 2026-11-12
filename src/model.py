"""Stage 3: LightGBM pair classifier, one-to-one assignment and macro F0.5 evaluation.

Assignment rule: every S2/S3 record belongs to at most one S1 entity (Source 1 is
deduplicated and the training truth confirms each S2/S3 id appears in at most one
group), so each query record is attached to its highest-probability S1 candidate
if that probability clears the threshold tuned for macro F0.5 on validation.
"""
import lightgbm as lgb
import numpy as np
import polars as pl

import config
from features import FEATURES

BETA2 = 0.25
LGB_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=255, num_threads=0, verbose=-1, seed=config.SEED,
)


def train_model(x, y, x_val=None, y_val=None, rounds=4000):
    dtrain = lgb.Dataset(x, y, feature_name=FEATURES, free_raw_data=True)
    valid = [lgb.Dataset(x_val, y_val, reference=dtrain)] if x_val is not None else []
    cbs = [lgb.log_evaluation(100)]
    if valid:
        cbs.append(lgb.early_stopping(100))
    return lgb.train(LGB_PARAMS, dtrain, num_boost_round=rounds, valid_sets=valid, callbacks=cbs)


def assign(scored: pl.DataFrame) -> pl.DataFrame:
    """scored: q_rid, s1_rid, p -> best S1 per query (q_rid, s1_rid, p, margin)."""
    return (scored.sort("p", descending=True)
                  .group_by("q_rid", maintain_order=True)
                  .agg(pl.col("s1_rid").first(), pl.col("p").first(),
                       (pl.col("p").first() - pl.col("p").get(1, null_on_oob=True).fill_null(0)).alias("margin")))


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1_eval: pl.Series) -> float:
    """pred/truth: (q_rid, s1_rid) pairs. Macro F0.5 over the S1 ids in `s1_eval`
    (singletons included: empty prediction on an empty truth scores 1)."""
    ev = pl.DataFrame({"s1_rid": s1_eval})
    tp = pred.join(truth, on=["q_rid", "s1_rid"]).group_by("s1_rid").len("tp")
    npred = pred.group_by("s1_rid").len("np")
    ntrue = truth.group_by("s1_rid").len("nt")
    df = (ev.join(tp, on="s1_rid", how="left").join(npred, on="s1_rid", how="left")
            .join(ntrue, on="s1_rid", how="left").fill_null(0))
    tp_, np_, nt_ = (df[c].to_numpy().astype(np.float64) for c in ("tp", "np", "nt"))
    fp, fn = np_ - tp_, nt_ - tp_
    denom = (1 + BETA2) * tp_ + BETA2 * fn + fp
    f = np.where(denom > 0, (1 + BETA2) * tp_ / np.maximum(denom, 1e-12), 1.0)
    return float(f.mean())


def tune_threshold(best: pl.DataFrame, truth: pl.DataFrame, s1_eval: pl.Series):
    res = []
    for thr in np.arange(0.20, 0.92, 0.02):
        pred = best.filter(pl.col("p") >= thr).select("q_rid", "s1_rid")
        res.append((float(thr), macro_f05(pred, truth, s1_eval)))
    thr, f = max(res, key=lambda r: r[1])
    return thr, f, res


def select_sets(best: pl.DataFrame, floor=0.05, beta2=BETA2) -> pl.DataFrame:
    """Per-S1 expected-F0.5 decision. For each S1, sort its assigned queries by p and
    keep the prefix of size m maximising a plug-in estimate of expected F-beta:
        E[F](m) ~ (1+b2) S_m / ((1+b2) S_m + b2 (T - S_m) + (m - S_m)),  m >= 1
        E[F](0) = prod(1 - p_i)          (probability the entity is a singleton)
    with S_m = sum of the top-m probabilities and T = sum of all probabilities."""
    df = (best.filter(pl.col("p") >= floor).sort("s1_rid", pl.col("p"), descending=[False, True])
              .with_columns(pl.int_range(1, pl.len() + 1).over("s1_rid").alias("m"),
                            pl.col("p").cum_sum().over("s1_rid").alias("S"),
                            pl.col("p").sum().over("s1_rid").alias("T"),
                            (1 - pl.col("p")).log().sum().over("s1_rid").exp().alias("f0")))
    df = df.with_columns(((1 + beta2) * pl.col("S")
                          / ((1 + beta2) * pl.col("S") + beta2 * (pl.col("T") - pl.col("S"))
                             + (pl.col("m") - pl.col("S")))).alias("ef"))
    keep = (df.with_columns(pl.col("ef").max().over("s1_rid").alias("efmax"))
              .filter((pl.col("efmax") > pl.col("f0")))
              .with_columns(pl.col("ef").arg_max().over("s1_rid").alias("_am"),
                            (pl.int_range(pl.len()).over("s1_rid")).alias("_i"))
              .filter(pl.col("_i") <= pl.col("_am")))
    return keep.select("q_rid", "s1_rid", "p")


def em_prior(p: np.ndarray, pi_train: float, iters=200) -> float:
    """Saerens-Latinne-Decaestecker EM: estimate the positive rate of a new population
    from classifier posteriors calibrated on a population with positive rate pi_train."""
    pi = pi_train
    for _ in range(iters):
        a = (pi / pi_train) * p
        b = ((1 - pi) / (1 - pi_train)) * (1 - p)
        new = float(np.mean(a / (a + b)))
        if abs(new - pi) < 1e-8:
            break
        pi = new
    return pi


def prior_adjust(scored: pl.DataFrame, country: pl.DataFrame, pi_train: dict, floor=0.02):
    """Label-shift correction per country: re-estimate the positive rate among each record's
    best attachment (p >= floor) with EM and rescale the odds of every score accordingly.
    scored: q_rid, s1_rid, p (one row per query); country: s1_rid, country."""
    df = scored.join(country, on="s1_rid", how="left")
    factors = {}
    for c in df["country"].unique().to_list():
        pt = pi_train.get(c, pi_train["*"])
        pc = df.filter((pl.col("country") == c) & (pl.col("p") >= floor))["p"].to_numpy()
        pi = em_prior(pc, pt) if len(pc) else pt
        factors[c] = (pi / (1 - pi)) / (pt / (1 - pt))
    print("[prior] EM odds factors per country:", {k: round(v, 3) for k, v in factors.items()})
    f = pl.col("country").replace_strict(factors, default=1.0, return_dtype=pl.Float64)
    odds = pl.col("p") / (1 - pl.col("p")).clip(1e-12)
    return df.with_columns((odds * f / (1 + odds * f)).cast(pl.Float32).alias("p")).select(scored.columns)
