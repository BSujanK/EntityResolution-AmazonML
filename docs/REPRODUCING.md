# Reproducing the results

## 1. Requirements

* Python ≥ 3.11 (developed on CPython 3.13, Windows 11; the shell scripts need `sh`, e.g. Git Bash or WSL on Windows)
* ≈ 16 GB RAM (peak ≈ 9 GB), ≈ 30 GB free disk for `work/`
* CPU is enough. A CUDA GPU is picked up automatically by the cross-encoder scripts and makes part 2 about 20× faster.

```bash
python -m pip install -r requirements.txt
```

The cross-encoder base model `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache-2.0) is downloaded from Hugging Face on first use.

## 2. Data

The challenge dataset is **not** included in this repository. Place it like this, or point `ER_DATA_DIR` / `--data-dir` elsewhere:

```
data/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

All files are tab-separated, with columns `entity_id, business_name, business_address, country`.
The ground-truth file has columns `source1_entity_id, matched_entity_ids`.

## 3. Part 1: classical cascade (~5 h on CPU)

```bash
python src/run_pipeline.py --data-dir data            # = --stage all
```

or one stage at a time:

| Command | Produces (in `work/`) | Time |
|---|---|---|
| `python src/run_pipeline.py --data-dir data --stage prep` | `{train,test}_records.parquet`, `train_truth.parquet` | ≈ 3.5 min |
| `… --stage block` | `{train,test}_candidates.parquet` (top-10 S1 per record) | ≈ 38 min per split |
| `… --stage train` | `model_fold{0,1}.txt`, `train_oof_scored.parquet`, `model_stage2.txt`, `train_priors.json`, `val_report.json` | ≈ 97 min |
| `… --stage predict` | `test_scored.parquet` and a stage-1/stage-2 submission in `output/` | ≈ 107 min |

Each stage is skipped when its artefact already exists. Delete the artefact to force a re-run.

## 4. Part 2: cross-encoders and expansions (~10–12 h on CPU)

```bash
sh src/run_ce_stage.sh
```

This runs, in order (see the script for the exact commands):

1. `ce_export.py`, `ce_prep_eval.py`, `ce_uncertain_features.py`: the uncertain pairs (887 k train / 1.11 M test) with raw text and features
2. `ce_train.py` × 4: two folds × {90 k, 220 k} training pairs
3. `ce_score.py`: out-of-fold scores on train, fold-averaged scores on test
4. `ce_stack_train.py`: the stacker (prints held-out macro F0.5 and the uncertain-band AUC)
5. `ce_predict.py` with `ER_EXTRA_FACTORS='{"France": 0.5}'`: the v10a outputs
6. `ext_expand_block.py` → `ext_score.py` → `ext_phase1.py` → `ext_phase2.sh` → `ext_phase3.py eval|build`: v10b
7. the India top-50 pass (`ext_expand_block_in50.py`, `EXT_TAG=50 ext_score.py`, `ext_merge50.py`, phases 1–3): **v10c, written to `output/`**

## 5. Validate and package

With the organisers' stdlib-only checker (shipped with the challenge resources, not in this repo):

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir data/test --check-ids
```

Expected: `PASS`, 1,732,544 rows (one per test S1 entity), 5,772,673 matched links, 6,595,983 candidate pairs.

Build the challenge submission zip:

```bash
python tools/make_zip.py <team_name>
```

## 6. Regenerating the figures and the report

```bash
python tools/make_figures.py          # docs/assets/*.png from the published numbers
python report/cpu_projection.py       # docs/assets/cpu_projection.png
python report/build_report.py         # docs/Entity_Resolution_Project_Report.pdf (needs reportlab; uses Windows system fonts)
```

## 7. Practical notes

* **Never run two memory-heavy stages at the same time** on a 16 GB machine. Each one is designed to peak at about 9 GB.
* Long stages are best started detached (`nohup … &` or a terminal multiplexer), because they run for hours.
* Set `PYTHONIOENCODING=utf-8` on Windows consoles, because records contain Indic scripts.
* Stage-1 label-shift factors must be estimated on **stage-1** scores (what `ce_predict.py` does). Estimating them on
  stacked scores inflates them.
* Results are deterministic up to LightGBM/PyTorch thread scheduling. Expect the 4th–5th decimal of F0.5 to vary.
