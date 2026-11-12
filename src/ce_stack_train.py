"""Richer stacker: cross-encoder score + the 72 stage-1 features on uncertain pairs. Held-out macro F0.5."""
import json, sys
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
import config, prep
from features import FEATURES
from model import select_sets, macro_f05

W = config.WORK_DIR
logit = lambda c: (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()
truth = pl.read_parquet(W / "train_truth.parquet").with_columns(pl.col("q_rid").cast(pl.UInt32), pl.col("s1_rid").cast(pl.UInt32))
unc = pl.read_parquet(W / "ce/train_uncertain.parquet", columns=["q_rid", "s1_rid", "y", "p"])
oof = pl.concat([pl.read_parquet(W / f"ce/oof_ce_{m}.parquet") for m in ("A", "B")]).join(pl.concat([pl.read_parquet(W / f"ce/v7_artifacts/oof_ce_{m}.parquet") for m in ("A", "B")]).rename({"ce": "ce7"}), on=["q_rid", "s1_rid"])
feats = pl.read_parquet(W / "ce/train_uncertain_feats.parquet")
d = (unc.join(oof, on=["q_rid", "s1_rid"]).join(feats, on=["q_rid", "s1_rid"])
        .join(pl.read_parquet(W / "ce/train_best_v6.parquet").select("q_rid", "margin"), on="q_rid")
        .with_columns(logit("p").alias("lp1"), logit("ce").alias("lce"), logit("ce7").alias("lce7"), (pl.col("s1_rid").hash(seed=13) % 5 == 0).alias("is_ev")))
COLS = ["lp1", "margin", "lce", "lce7"] + FEATURES
tr, ev = d.filter(~pl.col("is_ev")), d.filter(pl.col("is_ev"))
es = (tr["q_rid"].hash(seed=3) % 10 == 0).to_numpy()
X, y = tr.select(COLS).to_numpy(), tr["y"].to_numpy()
params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)
m = lgb.train(params, lgb.Dataset(X[~es], y[~es]), 3000, valid_sets=[lgb.Dataset(X[es], y[es])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
ev = ev.with_columns(pl.Series("ps", m.predict(ev.select(COLS).to_numpy()).astype(np.float32)))
yy = ev["y"].to_numpy()
print(f"stacker3(v7+v8 CE) rounds {m.best_iteration}; uncertain eval AUC: stage1 {roc_auc_score(yy, ev['p']):.4f}  CE {roc_auc_score(yy, ev['ce']):.4f}  stacked2 {roc_auc_score(yy, ev['ps']):.4f}")
m.save_model(str(W / "ce/stacker3.txt"))
(W / "ce/stacker3_cols.json").write_text(json.dumps(COLS))

s1 = prep.load_records("train", ["rid", "src", "country"]).filter(pl.col("src") == 1).with_columns(pl.col("rid").cast(pl.UInt32))
s1_ev = s1.filter(pl.col("rid").hash(seed=13) % 5 == 0); ids = s1_ev["rid"]
tv = truth.filter(pl.col("s1_rid").is_in(ids.implode()))
best = pl.read_parquet(W / "ce/train_best_v6.parquet").filter(pl.col("s1_rid").is_in(ids.implode()))
p2 = pl.read_parquet(W / "ce/heldout_p2_v6.parquet").rename({"p": "p2"})
j = (best.join(p2, on=["q_rid", "s1_rid"], how="left").with_columns(pl.col("p2").fill_null(pl.col("p")))
         .join(ev.select("q_rid", "s1_rid", "ps"), on=["q_rid", "s1_rid"], how="left").with_columns(pl.col("ps").fill_null(pl.col("p"))))
for name, expr in {"v6 baseline  min(p1,p2)": pl.min_horizontal("p", "p2"), "stack3      min(ps,p2)": pl.min_horizontal("ps", "p2"),
                   "stack3 only  ps       ": pl.col("ps")}.items():
    sel = select_sets(j.select("q_rid", "s1_rid", expr.alias("p"))).select("q_rid", "s1_rid")
    line = f"{name}: all {macro_f05(sel, tv, ids):.5f}"
    for c in ["US", "India"]:
        cid = s1_ev.filter(pl.col("country") == c)["rid"]
        line += f"  {c} {macro_f05(sel, tv.filter(pl.col('s1_rid').is_in(cid.implode())), cid):.5f}"
    print(line)
