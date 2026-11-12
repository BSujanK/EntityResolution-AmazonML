"""Final test prediction with the cross-encoder stack.

For each S2/S3 test record's best S1 candidate:
  * uncertain pairs (0.02 <= p1 <= 0.98): p = stacker(logit p1, margin, logit mean(CE_A, CE_B))
  * other pairs: p = p1
  * final = min(p, p_stage2)   (group model can only veto)
then the per-country EM label-shift correction and per-S1 expected-F0.5 selection.
Writes output/matching_results.tsv and output/candidate_pairs.tsv (same candidate set as before).

    python src/ce_predict.py
"""
import json

import lightgbm as lgb
import numpy as np
import polars as pl

import config
from blocking import candidates_path
from model import assign, em_prior, select_sets
from prep import load_records
from run_pipeline import write_outputs
from stage2 import apply_stage2, group_features

W = config.WORK_DIR
STACKER = "ce/stacker3.txt"            # 76 inputs: logit p1, margin, logit CE(v8), logit CE(v7) + 72 stage-1 features
COLS_FILE = "ce/stacker3_cols.json"
CE_V8 = ("ce/v8_test_ce_A.parquet", "ce/v8_test_ce_B.parquet")      # 220k-pair cross-encoders
CE_V7 = ("ce/v7_artifacts/test_ce_A.parquet", "ce/v7_artifacts/test_ce_B.parquet")  # 90k-pair cross-encoders
SOFT_VETO = 0.1                         # stage-2 veto applies only when it is confident: p2 < 0.1 ...
# ... but only in countries with training data (where the soft veto was validated). Unseen countries
# (France) keep the full veto min(p, p2): the conservative rule is the only one validated anywhere.


def logit(c):
    x = pl.col(c).clip(1e-6, 1 - 1e-6)
    return (x / (1 - x)).log()


def main():
    best = assign(pl.read_parquet(W / "test_scored.parquet"))
    def ce_mean(paths, name):
        a, b = (pl.read_parquet(W / x) for x in paths)
        return (a.rename({"ce": "a"}).join(b.rename({"ce": "b"}), on=["q_rid", "s1_rid"])
                 .select("q_rid", "s1_rid", ((pl.col("a") + pl.col("b")) / 2).alias(name)))
    ce = ce_mean(CE_V8, "ce").join(ce_mean(CE_V7, "ce7"), on=["q_rid", "s1_rid"])
    cols = json.loads((W / COLS_FILE).read_text())
    feats = pl.read_parquet(W / "ce/test_uncertain_feats.parquet")
    unc = (best.join(ce, on=["q_rid", "s1_rid"]).join(feats, on=["q_rid", "s1_rid"])
               .with_columns(logit("p").alias("lp1"), logit("ce").alias("lce"), logit("ce7").alias("lce7")))
    stacker = lgb.Booster(model_file=str(W / STACKER))
    unc = unc.with_columns(pl.Series("ps", stacker.predict(unc.select(cols).to_numpy()).astype(np.float32)))
    print(f"[ce-predict] uncertain test pairs re-scored: {unc.height:,}")
    stacked = (best.join(unc.select("q_rid", "s1_rid", "ps"), on=["q_rid", "s1_rid"], how="left")
                   .with_columns(pl.col("ps").fill_null(pl.col("p")).alias("p")).drop("ps"))

    gf = group_features(best, load_records("test", ["nums", "name_cat", "addr"]))
    final_cand = gf.select("q_rid", "s1_rid")
    b2 = lgb.Booster(model_file=str(W / "model_stage2.txt"))
    ctry = load_records("test", ["rid", "country"]).select(pl.col("rid").cast(pl.UInt32).alias("s1_rid"), "country")
    seen = [c for c in json.loads((W / "train_priors.json").read_text()) if c != "*"]
    final = (stacked.join(apply_stage2(b2, gf).rename({"p": "p2"}), on=["q_rid", "s1_rid"]).join(ctry, on="s1_rid")
                    .select("q_rid", "s1_rid",
                            pl.when(pl.col("country").is_in(seen) & (pl.col("p2") >= SOFT_VETO)).then(pl.col("p"))
                              .otherwise(pl.min_horizontal("p", "p2")).alias("p")))
    print(f"[ce-predict] soft veto in {seen}; full veto elsewhere")
    # Label-shift factors must be estimated on the STAGE-1 scores (the scores train_priors.json was
    # measured on), then applied to the final stacked scores. Estimating them on stacked scores mixes
    # two calibrations and inflated India's factor to 2.24 (v7b, never submitted).
    pri = json.loads((W / "train_priors.json").read_text())
    b1 = best.join(ctry, on="s1_rid").filter(pl.col("p") >= 0.02)
    fac = {}
    for c in ctry["country"].unique().to_list():
        pt = pri.get(c, pri["*"])
        pi = em_prior(b1.filter(pl.col("country") == c)["p"].to_numpy(), pt)
        fac[c] = (pi / (1 - pi)) / (pt / (1 - pt))
    # optional extra per-country odds factors, e.g. ER_EXTRA_FACTORS='{"France": 0.5}': the France probe
    # (0.978 < 0.979) showed French borderline pairs are worse than the US/India-based EM estimate assumes
    import os
    for c, x in json.loads(os.environ.get("ER_EXTRA_FACTORS", "{}")).items():
        fac[c] = fac.get(c, 1.0) * x
    print("[ce-predict] EM odds factors (from stage-1 scores):", {k: round(v, 3) for k, v in fac.items()})
    f = pl.col("country").replace_strict(fac, default=1.0, return_dtype=pl.Float64)
    odds = pl.col("p") / (1 - pl.col("p")).clip(1e-12)
    final = (final.join(ctry, on="s1_rid").with_columns((odds * f / (1 + odds * f)).cast(pl.Float32).alias("p"))
                  .select("q_rid", "s1_rid", "p"))
    final.write_parquet(W / "ce/test_final_p.parquet")   # final per-link probabilities (for variants)
    sel = select_sets(final)
    write_outputs(final_cand, sel)
    print(f"[ce-predict] {final_cand.height:,} candidate pairs, {sel.height:,} matches written")


if __name__ == "__main__":
    main()
