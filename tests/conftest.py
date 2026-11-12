"""Shared test setup: the pipeline modules import each other as top-level modules
(``import config``), so ``src/`` has to be on sys.path."""
import sys
from pathlib import Path

import polars as pl
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from normalize import normalize


def make_records(rows):
    """Normalised record frame from (name, address, src[, country]) tuples; rid is dense,
    country defaults to 'US'."""
    raw = pl.DataFrame({
        "business_name": [r[0] for r in rows],
        "business_address": [r[1] for r in rows],
        "country": [r[3] if len(r) > 3 else "US" for r in rows],
    })
    src = pl.Series("src", [r[2] for r in rows], dtype=pl.Int8)
    return normalize(raw).with_columns(src).with_row_index("rid")


def ids(*values):
    """UInt32 series of row ids (the dtype the pipeline uses for q_rid / s1_rid)."""
    return pl.Series("s1_rid", values, dtype=pl.UInt32)


def pairs(q_rid, s1_rid, **extra):
    """(q_rid, s1_rid[, extra columns]) frame with UInt32 ids."""
    return pl.DataFrame({"q_rid": q_rid, "s1_rid": s1_rid, **extra}).cast(
        {"q_rid": pl.UInt32, "s1_rid": pl.UInt32})


@pytest.fixture
def norm():
    """norm(name, address='') -> dict of the normalised columns of a single record."""
    def _norm(name, address=""):
        df = pl.DataFrame({"business_name": [name], "business_address": [address], "country": ["US"]})
        return normalize(df).row(0, named=True)
    return _norm
