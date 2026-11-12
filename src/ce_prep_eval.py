"""Held-out evaluation setup: best attachments, out-of-fold stage-2 veto scores for held-out S1, CE scoring rows."""
import polars as pl, config, prep
from model import assign
from stage2 import group_features, train_stage2, apply_stage2
W = config.WORK_DIR
truth = pl.read_parquet(W/"train_truth.parquet").with_columns(pl.col("q_rid").cast(pl.UInt32), pl.col("s1_rid").cast(pl.UInt32))
lab = truth.with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
best = assign(pl.read_parquet(W/"train_oof_scored.parquet"))
best.write_parquet(W/"ce/train_best_v6.parquet")
gf = group_features(best, prep.load_records("train", ["nums","name_cat","addr"])).join(lab, on=["q_rid","s1_rid"], how="left").with_columns(pl.col("y").fill_null(0))
ev = gf["s1_rid"].hash(seed=13) % 5 == 0
g_tr, g_ev = gf.filter(~ev), gf.filter(ev)
es = g_tr["s1_rid"].hash(seed=17) % 10 == 0
b2 = train_stage2(g_tr.filter(~es), g_tr.filter(~es)["y"].to_numpy(), g_tr.filter(es), g_tr.filter(es)["y"].to_numpy())
apply_stage2(b2, g_ev).write_parquet(W/"ce/heldout_p2_v6.parquet")
unc = pl.read_parquet(W/"ce/train_uncertain.parquet", columns=["q_rid","s1_rid","fold"])
unc = unc.with_columns((pl.col("s1_rid").hash(seed=13) % 5 == 0).alias("is_ev"))
rows_ev = unc.filter(pl.col("is_ev")); rows_st = unc.filter(~pl.col("is_ev")).sample(120_000, seed=5)
pl.concat([rows_ev, rows_st]).write_parquet(W/"ce/rows_oof.parquet")
print("heldout p2 rows", g_ev.height, "| uncertain ev rows", rows_ev.height, "| stacker rows", rows_st.height)
