#!/bin/sh
# Cross-encoder scores for changed & uncertain pairs. train: out-of-fold; test: mean of both fold models.
cd "$(dirname "$0")/.."   # work from the repository root
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error PYTHONPATH=src
E=work/ext; S="python src/ce_score.py"
for split in train test; do
  if [ $split = train ]; then F0="--fold 1"; F1="--fold 0"; else F0=""; F1=""; fi
  $S --model work/ce/model_v2_f0 --data $E/${split}_ce_rows.parquet $F0 --out $E/${split}_v8a.parquet >> $E/phase2.log 2>&1
  $S --model work/ce/model_v2_f1 --data $E/${split}_ce_rows.parquet $F1 --out $E/${split}_v8b.parquet >> $E/phase2.log 2>&1
  $S --model work/ce/model_f0    --data $E/${split}_ce_rows.parquet $F0 --out $E/${split}_v7a.parquet >> $E/phase2.log 2>&1
  $S --model work/ce/model_f1    --data $E/${split}_ce_rows.parquet $F1 --out $E/${split}_v7b.parquet >> $E/phase2.log 2>&1
  python - <<PY >> $E/phase2.log 2>&1
import polars as pl
E = "$E"; split = "$split"
for v in ("v8", "v7"):
    a, b = pl.read_parquet(f"{E}/{split}_{v}a.parquet"), pl.read_parquet(f"{E}/{split}_{v}b.parquet")
    if split == "train":   # out-of-fold: each row was scored by exactly one model
        d = pl.concat([a, b])
    else:
        d = a.rename({"ce": "x"}).join(b.rename({"ce": "y"}), on=["q_rid", "s1_rid"]).select("q_rid", "s1_rid", ((pl.col("x") + pl.col("y")) / 2).alias("ce"))
    d.write_parquet(f"{E}/{split}_ce_{v}.parquet"); print(split, v, d.height)
PY
done
echo PHASE2_DONE >> $E/phase2.log
