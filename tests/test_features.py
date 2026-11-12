"""Pairwise features and OCR repair (src/features.py)."""
import numpy as np
import polars as pl
import pytest

import features as F
from conftest import make_records, pairs

ROWS = [
    ("Oak Hardware", "100 Oak St", 1),             # 0: the S1 entity
    ("Oak Hardware", "101 Oak St", 2),             # 1: last digit off by one
    ("Oak Hardware", "900 Oak St", 2),             # 2: leading digit differs
    ("Oak Hardware", "10 Oak St", 2),              # 3: house number is a prefix
    ("Oak Hardware", "5100 Oak St", 2),            # 4: house number is a suffix
    ("Oak Hardware Ltd", "100 Oak Street", 3),     # 5: same record from source 3, extra legal form
    ("Oak Hardware Pvt Ltd", "", 2),               # 6: no address at all
    ("Maple Bakery", "100 Oak St", 2),             # 7: other business at the same address
]


def candidate_pairs(q_rid, s1_rid, bscore=None):
    n = len(q_rid)
    return pairs(q_rid, s1_rid, bscore=bscore or [1.0] * n, brank=list(range(1, n + 1)),
                 nA=[1] * n, nB=[1] * n, nC=[1] * n, nD=[0] * n, nE=[0] * n, nF=[0] * n, nG=[0] * n,
                 s1_top1_cnt=[1] * n).cast({"bscore": pl.Float32, "brank": pl.UInt8})


@pytest.fixture(scope="module")
def feats():
    store = F.RecordStore(make_records(ROWS).select(F.FEAT_REC_COLS))
    q = list(range(1, len(ROWS)))
    return F.pair_features(store, candidate_pairs(q, [0] * len(q), bscore=[1.0] * len(q))).sort("q_rid")


def row(feats, q_rid):
    return feats.filter(pl.col("q_rid") == q_rid).row(0, named=True)


class TestFeatureFrame:
    def test_feature_names_are_unique(self):
        assert len(F.FEATURES) == len(set(F.FEATURES))

    def test_output_schema_is_ids_plus_float32_features(self, feats):
        assert feats.columns == ["q_rid", "s1_rid", *F.FEATURES]
        assert all(feats.schema[f] == pl.Float32 for f in F.FEATURES)
        assert feats.height == len(ROWS) - 1

    def test_no_nulls_or_nans(self, feats):
        assert np.isfinite(feats.select(F.FEATURES).to_numpy()).all()


class TestNameFeatures:
    def test_identical_names_are_exact_matches(self, feats):
        r = row(feats, 1)
        assert r["n_exact"] == 1 and r["n_ratio"] == 1 and r["n_jac"] == 1
        assert r["n_cos"] == pytest.approx(1)

    def test_different_names_share_nothing(self, feats):
        r = row(feats, 7)
        assert r["n_exact"] == 0 and r["n_jac"] == 0 and r["n_cos"] == 0

    def test_legal_form_only_on_one_side_is_not_a_conflict(self, feats):
        r = row(feats, 6)
        assert r["legal_eq"] == 0 and r["legal_conflict"] == 0
        assert r["n_exact"] == 1                        # core names ignore legal forms

    def test_source3_flag(self, feats):
        assert row(feats, 5)["is_s3"] == 1
        assert row(feats, 1)["is_s3"] == 0


class TestAddressFeatures:
    def test_empty_query_address_is_flagged(self, feats):
        assert row(feats, 6)["a_empty_q"] == 1
        assert row(feats, 1)["a_empty_q"] == 0

    def test_abbreviation_difference_does_not_hurt_address_similarity(self, feats):
        r = row(feats, 5)                               # "street" vs "st"
        assert r["a_cos"] == pytest.approx(1)
        assert r["num1_eq"] == 1


class TestHouseNumberFeatures:
    def test_last_digit_edit(self, feats):
        r = row(feats, 1)                               # 101 vs 100
        assert r["num1_eq"] == 0
        assert r["num1_hamming"] == 1
        assert r["num1_diff_from_left"] == 2 and r["num1_diff_from_right"] == 0
        assert r["num1_last_digit_near"] == 1

    def test_leading_digit_edit(self, feats):
        r = row(feats, 2)                               # 900 vs 100
        assert r["num1_hamming"] == 1
        assert r["num1_diff_from_left"] == 0 and r["num1_diff_from_right"] == 2
        assert r["num1_last_digit_near"] == 0

    def test_prefix_and_suffix_relations(self, feats):
        prefix, suffix = row(feats, 3), row(feats, 4)   # 10 vs 100, 5100 vs 100
        assert prefix["num1_prefix"] == 1 and prefix["num1_suffix"] == 0
        assert suffix["num1_suffix"] == 1 and suffix["num1_prefix"] == 0
        assert prefix["num1_hamming"] == -1             # different lengths: no positional comparison

    def test_equal_numbers(self, feats):
        r = row(feats, 7)                               # 100 vs 100 at another business
        assert r["num1_eq"] == 1 and r["num1_hamming"] == 0 and r["num1_logdiff"] == 0

    def test_missing_number_uses_sentinels(self, feats):
        r = row(feats, 6)
        assert r["num1_eq"] == 0 and r["num1_logdiff"] == -1 and r["num_len_q"] == 0


class TestCompetitorContext:
    def test_single_candidate_query_has_no_gap(self, feats):
        r = row(feats, 1)
        assert r["n_cand"] == 1 and r["bscore_gap"] == 0

    def test_gap_features_compare_with_the_best_competitor(self):
        recs = make_records([("Oak Hardware", "100 Oak St", 1), ("Oak Hardwares", "100 Oak St", 1),
                             ("Oak Hardware", "100 Oak St", 2)])
        store = F.RecordStore(recs.select(F.FEAT_REC_COLS))
        out = F.pair_features(store, candidate_pairs([2, 2], [0, 1], bscore=[2.0, 1.0])).sort("s1_rid")
        assert out["n_cand"].to_list() == [2, 2]
        assert out["bscore_gap"].to_list() == pytest.approx([0.0, -1.0])
        assert out["n_cos_gap"][0] == 0 and out["n_cos_gap"][1] < 0


class TestRepairOcr:
    @staticmethod
    def repaired(names):
        return F.repair_ocr(make_records([(n, "", 1) for n in names]))

    def test_digits_inside_words_become_letters(self):
        out = self.repaired(["M0unt 8aptist Church"])
        assert out["core_toks"][0].to_list() == ["mount", "baptist", "church"]
        assert out["name_core"][0] == "mount baptist church"
        assert out["name_cat"][0] == "mountbaptistchurch"

    def test_glued_names_are_repaired(self):
        assert self.repaired(["Nunezr0yalsierra"])["name_cat"][0] == "nunezroyalsierra"

    def test_pure_numbers_are_left_alone(self):
        out = self.repaired(["Studio 54", "7 Eleven"])
        assert out["core_toks"].to_list() == [["studio", "54"], ["7", "eleven"]]

    def test_skeleton_is_recomputed_after_repair(self):
        noisy = self.repaired(["M0unt 8aptist Church"])["name_skel"][0]
        clean = make_records([("Mount Baptist Church", "", 1)])["name_skel"][0]
        assert noisy == clean

    def test_clean_names_are_unchanged(self):
        recs = make_records([("Sunrise Bakery", "", 1)])
        assert F.repair_ocr(recs)["name_core"].to_list() == recs["name_core"].to_list()


class TestRecordStore:
    def test_requires_dense_row_ids(self):
        recs = make_records([("A Co", "", 1), ("B Co", "", 2)])
        with pytest.raises(AssertionError):
            F.RecordStore(recs.with_columns((pl.col("rid") + 5).alias("rid")).select(F.FEAT_REC_COLS))

    def test_tfidf_rows_are_unit_norm_and_self_similar(self):
        recs = make_records([("Sunrise Bakery", "45 Elm Road", 1), ("Harbor Auto Repair", "", 1)])
        store = F.RecordStore(recs.select(F.FEAT_REC_COLS))
        idx = np.array([0, 1])
        assert F._rowdot(store.name_vec, idx, idx) == pytest.approx([1.0, 1.0])
        assert F._rowdot(store.name_vec, np.array([0]), np.array([1])) == pytest.approx([0.0])
        assert store.addr_info[1] == 0                  # empty address carries no IDF mass
