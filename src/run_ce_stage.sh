#!/bin/sh
# Cross-encoder stage (v7-v10a). Run from the repository root AFTER `run_pipeline.py --stage train`
# and `--stage predict` (which produce the stage-1/stage-2 models, out-of-fold scores and test_scored.parquet).
# CPU: ~10-12 h in total (a GPU makes it ~20x faster; ce_train/ce_score use CUDA automatically when present).
set -e
export PYTHONPATH=src PYTHONIOENCODING=utf-8 TRANSFORMERS_VERBOSITY=error
W=work/ce; mkdir -p $W
python src/ce_export.py                      # -> work/ce/{train,test}_uncertain.parquet (887k / 1.11M pairs)
python src/ce_prep_eval.py                   # -> held-out stage-2 scores, work/ce/rows_oof.parquet
python src/ce_uncertain_features.py          # -> 72 stage-1 features for uncertain pairs
# v7 cross-encoders (90k pairs/fold) and v8 cross-encoders (220k pairs/fold), 2-fold cross-fitted
for f in 0 1; do
  python src/ce_train.py --data $W/train_uncertain.parquet --fold $f --n 90000  --out $W/model_f$f
  python src/ce_train.py --data $W/train_uncertain.parquet --fold $f --n 220000 --out $W/model_v2_f$f
done
mkdir -p $W/v7_artifacts
# out-of-fold scores (model trained on fold k scores the other fold) for stacker training
python src/ce_score.py --model $W/model_f0    --data $W/train_uncertain.parquet --rows $W/rows_oof.parquet --fold 1 --out $W/v7_artifacts/oof_ce_A.parquet
python src/ce_score.py --model $W/model_f1    --data $W/train_uncertain.parquet --rows $W/rows_oof.parquet --fold 0 --out $W/v7_artifacts/oof_ce_B.parquet
python src/ce_score.py --model $W/model_v2_f0 --data $W/train_uncertain.parquet --rows $W/rows_oof.parquet --fold 1 --out $W/oof_ce_A.parquet
python src/ce_score.py --model $W/model_v2_f1 --data $W/train_uncertain.parquet --rows $W/rows_oof.parquet --fold 0 --out $W/oof_ce_B.parquet
python src/ce_stack_train.py                 # trains work/ce/stacker3.txt, prints held-out macro F0.5
# test scores from all four cross-encoders
python src/ce_score.py --model $W/model_f0    --data $W/test_uncertain.parquet --out $W/v7_artifacts/test_ce_A.parquet
python src/ce_score.py --model $W/model_f1    --data $W/test_uncertain.parquet --out $W/v7_artifacts/test_ce_B.parquet
python src/ce_score.py --model $W/model_v2_f0 --data $W/test_uncertain.parquet --out $W/v8_test_ce_A.parquet
python src/ce_score.py --model $W/model_v2_f1 --data $W/test_uncertain.parquet --out $W/v8_test_ce_B.parquet
# final prediction (v10a): soft veto in countries with training data, full veto elsewhere, France odds x0.5
ER_EXTRA_FACTORS='{"France": 0.5}' python src/ce_predict.py      # -> output/*.tsv

# ---- v10b: candidate expansion (top-30 key blocking for weak records, best p1 < 0.5) ----
mkdir -p work/ext
python src/ext_expand_block.py              # rank 11-30 candidates for weak records
python src/ext_score.py                     # re-score expanded groups with stage 1 (streamed, resumable)
python src/ext_phase1.py                    # new best S1 per record; export changed & uncertain pairs
sh src/ext_phase2.sh                        # cross-encoders (v7 + v8) on the changed pairs
python src/ext_phase3.py eval               # held-out macro F0.5: v9 vs v9 + expansion
python src/ext_phase3.py build work/v10b 0.5        # v10b outputs (France odds x0.5)

# ---- v10c (FINAL): second expansion for India records, ranks 31-50 (top-50 key blocking) ----
python src/ext_expand_block_in50.py                 # weak India records -> ranks 11-50
EXT_TAG=50 python src/ext_score.py                  # re-score their expanded groups
python src/ext_merge50.py                           # India records take the rank 11-50 expansion
python src/ext_phase1.py && sh src/ext_phase2.sh && python src/ext_phase3.py eval
python src/ext_phase3.py build output '{"France": 0.5}'   # FINAL v10c outputs
