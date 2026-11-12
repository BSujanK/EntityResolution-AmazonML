"""Paths and global settings for the entity-resolution pipeline."""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

# Dataset root containing train/ and test/ (the challenge files are not redistributed in
# this repository). Override with the ER_DATA_DIR env var or the --data-dir flag of run_pipeline.py.
DATA_DIR = Path(os.environ.get("ER_DATA_DIR", PROJECT_DIR / "data"))
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", PROJECT_DIR / "work"))
OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT_DIR", PROJECT_DIR / "output"))

SEED = 42

# Blocking
MAX_S1_PER_KEY = 200     # blocking keys shared by more S1 records than this are dropped
TOP_K = 10               # candidate S1 entities kept per S2/S3 record after cheap scoring
QUERY_CHUNK = 1_000_000  # S2/S3 records processed per blocking/feature chunk

# Stage-1 training: S2/S3 query records are split into 2 cross-fitting folds by hash;
# each fold's model is fit on this many sampled queries (all their candidate pairs).
TRAIN_QUERIES_PER_FOLD = 1_200_000


def set_data_dir(path):
    global DATA_DIR
    DATA_DIR = Path(path)
