"""Blocking-key generation and candidate retrieval (src/blocking.py)."""
import polars as pl
import pytest

import blocking as B
from conftest import make_records


def _hash_value(*parts):
    return pl.select(B._hash(*[pl.lit(p) for p in parts])).item()


class TestHash:
    def test_is_deterministic(self):
        assert _hash_value("US", "A", "x") == _hash_value("US", "A", "x")

    def test_depends_on_every_part_and_their_order(self):
        base = _hash_value("US", "A", "x")
        assert _hash_value("FR", "A", "x") != base
        assert _hash_value("US", "B", "x") != base
        assert _hash_value("US", "A", "y") != base
        assert _hash_value("A", "US", "x") != base

    def test_separator_prevents_concatenation_collisions(self):
        assert _hash_value("ab", "c") != _hash_value("a", "bc")


class TestTokenDocumentFrequency:
    def test_counts_records_per_country_and_token(self):
        s1 = make_records([("Sunrise Bakery", "45 Elm Road", 1),
                           ("Sunrise Cafe", "45 Elm Road", 1),
                           ("Sunrise Bakery", "9 Oak Lane", 1, "FR")])
        ndf, adf = B.token_df(s1)
        names = {(c, t): n for c, t, n in ndf.iter_rows()}
        assert names[("US", "snrs")] == 2          # token = phonetic skeleton of "sunrise"
        assert names[("US", "bkr")] == 1
        assert names[("FR", "snrs")] == 1
        addrs = {(c, a): n for c, a, n in adf.iter_rows()}
        assert addrs[("US", "elm")] == 2
        assert addrs[("FR", "oak")] == 1

    def test_short_skeletons_fall_back_to_the_raw_token(self):
        rec = make_records([("Ab Corp", "", 1)])   # "corp" is a legal form; "ab" skeletonises to 2 letters
        assert B._name_tokens(rec)["t"].to_list() == ["~ab"]


class TestMakeKeys:
    ROWS = [("Sunrise Bakery", "45 Elm Road, Austin", 1),
            ("Sunrise Bakery", "45 Elm Road, Austin", 2),
            ("Harbor Auto Repair", "", 2)]

    def _keys(self, rows):
        recs = make_records(rows)
        ndf, adf = B.token_df(recs.filter(pl.col("src") == 1))
        return B.make_keys(recs, ndf, adf)

    def test_identical_records_produce_identical_key_sets(self):
        keys = self._keys(self.ROWS)
        per_rid = {rid: set(g["key"].to_list()) for (rid,), g in keys.group_by("rid")}
        assert per_rid[0] == per_rid[1]

    def test_all_key_types_fire_for_a_name_and_address_rich_record(self):
        keys = self._keys(self.ROWS)
        kts = set(keys.filter(pl.col("rid") == 0)["kt"].to_list())
        assert kts == {B.KEY_TYPES.index(k) for k in "ABCDEFG"}

    def test_record_without_address_only_gets_name_keys(self):
        keys = self._keys(self.ROWS)
        kts = set(keys.filter(pl.col("rid") == 2)["kt"].to_list())
        assert kts <= {B.KEY_TYPES.index(k) for k in "ADE"}
        assert B.KEY_TYPES.index("A") in kts

    def test_keys_are_unique_per_record(self):
        keys = self._keys(self.ROWS)
        assert keys.height == keys.unique(["rid", "key"]).height

    def test_keys_are_scoped_by_country(self):
        rows = [("Sunrise Bakery", "45 Elm Road", 1, "US"), ("Sunrise Bakery", "45 Elm Road", 2, "FR")]
        keys = self._keys(rows)
        assert set(keys.filter(pl.col("rid") == 0)["key"]).isdisjoint(keys.filter(pl.col("rid") == 1)["key"])

    def test_is_stable_across_calls(self):
        a, b = self._keys(self.ROWS), self._keys(self.ROWS)
        assert a.sort("rid", "key").equals(b.sort("rid", "key"))


class TestRunBlocking:
    ROWS = [
        ("Mount Baptist Church", "12 Main Street", 1),        # 0
        ("Sunrise Bakery LLC", "45 Elm Road, Austin, Texas", 1),   # 1
        ("Mount Baptist Church", "12 Main Street", 2),        # 2: exact copy of 0
        ("M0unt 8aptist Church", "12 Main Street", 2),        # 3: OCR noise in the name
        ("Sunrise Bakery", "45 Elm Rd Austin TX", 2),         # 4: legal form dropped, address abbreviated
        ("Harbor Auto Repair", "", 3),                        # 5: unrelated
    ]

    @pytest.fixture
    def cand(self):
        return B.run_blocking("t", make_records(self.ROWS), top_k=3, max_s1=50)

    def test_finds_the_true_s1_record_for_each_noisy_query(self, cand):
        found = dict(zip(cand["q_rid"].to_list(), cand["s1_rid"].to_list()))
        assert found == {2: 0, 3: 0, 4: 1}

    def test_only_s2_s3_records_are_queries(self, cand):
        assert set(cand["q_rid"].to_list()) <= {2, 3, 4, 5}

    def test_ocr_noisy_name_is_found_through_address_keys(self, cand):
        row = cand.filter(pl.col("q_rid") == 3).row(0, named=True)
        assert row["nD"] == 0 and row["nE"] == 0       # name-prefix keys cannot agree
        assert row["nC"] >= 1                           # number x address token does

    def test_exact_copy_scores_higher_than_noisy_copy(self, cand):
        score = {r["q_rid"]: r["bscore"] for r in cand.iter_rows(named=True)}
        assert score[2] > score[3]

    def test_ranks_start_at_one_and_respect_top_k(self):
        rows = [("Oak Hardware", "100 Oak St", 1), ("Oak Hardware", "101 Oak St", 1),
                ("Oak Hardware", "102 Oak St", 1), ("Oak Hardware", "100 Oak St", 2)]
        cand = B.run_blocking("t", make_records(rows), top_k=2, max_s1=50)
        assert sorted(cand["brank"].to_list()) == [1, 2]
        assert cand.filter(pl.col("brank") == 1)["s1_rid"].to_list() == [0]

    def test_keys_shared_by_too_many_s1_records_are_dropped(self):
        rows = [("Oak Hardware", "100 Oak St", 1), ("Oak Hardware", "100 Oak St", 1),
                ("Oak Hardware", "100 Oak St", 2)]
        assert B.run_blocking("t", make_records(rows), top_k=5, max_s1=1).height == 0
        assert B.run_blocking("t", make_records(rows), top_k=5, max_s1=2).height == 2

    def test_query_rids_restricts_the_queries(self):
        recs = make_records(self.ROWS)
        cand = B.run_blocking("t", recs, top_k=3, max_s1=50, query_rids=pl.Series([4], dtype=pl.UInt32))
        assert cand["q_rid"].unique().to_list() == [4]

    def test_records_in_other_countries_are_never_candidates(self):
        rows = [("Sunrise Bakery", "45 Elm Road", 1, "US"), ("Sunrise Bakery", "45 Elm Road", 2, "FR")]
        assert B.run_blocking("t", make_records(rows), top_k=3, max_s1=50).height == 0
