"""Assignment, macro F0.5, expected-F0.5 set selection and label-shift prior (src/model.py)."""
import numpy as np
import polars as pl
import pytest
from scipy.stats import beta

import model as M
from conftest import ids, pairs

EMPTY = pairs([], [])


class TestAssign:
    def test_keeps_best_s1_per_query_and_reports_margin(self):
        scored = pairs([1, 1, 1, 2], [7, 8, 9, 7], p=[0.6, 0.9, 0.2, 0.4])
        out = M.assign(scored).sort("q_rid")
        assert out["q_rid"].to_list() == [1, 2]
        assert out["s1_rid"].to_list() == [8, 7]
        assert out["p"].to_list() == [0.9, 0.4]
        assert out["margin"].to_list() == pytest.approx([0.3, 0.4])   # single candidate: margin = p

    def test_output_has_one_row_per_query(self):
        scored = pairs([1, 1, 2, 2, 2], [5, 6, 5, 6, 7], p=[0.5, 0.4, 0.3, 0.2, 0.1])
        assert M.assign(scored)["q_rid"].n_unique() == 2
        assert M.assign(scored).height == 2


class TestMacroF05:
    EVAL = ids(1, 2, 3)

    def test_perfect_prediction_scores_one(self):
        truth = pairs([10, 11, 20], [1, 1, 2])
        assert M.macro_f05(truth, truth, self.EVAL) == pytest.approx(1.0)

    def test_empty_prediction_on_empty_truth_scores_one(self):
        assert M.macro_f05(EMPTY, EMPTY, self.EVAL) == pytest.approx(1.0)

    def test_empty_prediction_loses_only_the_entities_with_matches(self):
        truth = pairs([10], [1])
        assert M.macro_f05(EMPTY, truth, self.EVAL) == pytest.approx(2 / 3)

    def test_false_positive_on_a_singleton_scores_zero_for_that_entity(self):
        assert M.macro_f05(pairs([10], [3]), EMPTY, self.EVAL) == pytest.approx(2 / 3)

    def test_hand_computed_mixed_case(self):
        truth = pairs([10, 11, 20], [1, 1, 2])
        pred = pairs([10, 99], [1, 1])
        # entity 1: tp=1 fp=1 fn=1 -> 1.25 / (1.25 + 0.25 + 1) = 0.5
        # entity 2: tp=0 fn=1      -> 0
        # entity 3: no truth, no prediction -> 1
        assert M.macro_f05(pred, truth, self.EVAL) == pytest.approx((0.5 + 0.0 + 1.0) / 3)

    def test_precision_is_weighted_more_than_recall(self):
        truth = pairs([10, 11], [1, 1])
        only_one = pairs([10], [1])                 # tp=1 fp=0 fn=1
        one_extra = pairs([10, 11, 12], [1, 1, 1])  # tp=2 fp=1 fn=0
        f_miss = M.macro_f05(only_one, truth, ids(1))
        f_extra = M.macro_f05(one_extra, truth, ids(1))
        assert f_miss == pytest.approx(1.25 / 1.5)
        assert f_extra == pytest.approx(2.5 / 3.5)
        assert f_miss > f_extra

    def test_only_listed_entities_are_scored(self):
        truth = pairs([10, 20], [1, 2])
        assert M.macro_f05(pairs([10], [1]), truth, ids(1)) == pytest.approx(1.0)


class TestTuneThreshold:
    def test_picks_the_threshold_with_best_score(self):
        best = pairs([1, 2, 3], [1, 1, 2], p=[0.95, 0.5, 0.3])
        truth = pairs([1], [1])
        thr, f, grid = M.tune_threshold(best, truth, ids(1, 2))
        # accepting only q1 is optimal; any threshold in (0.5, 0.95] does that
        assert f == pytest.approx(1.0)
        assert 0.5 < thr <= 0.95
        assert len(grid) == 36
        assert max(score for _, score in grid) == pytest.approx(f)


class TestSelectSets:
    @staticmethod
    def select(ps, s1=1, **kw):
        df = pl.DataFrame({"q_rid": list(range(len(ps))), "s1_rid": [s1] * len(ps), "p": ps})
        return M.select_sets(df, **kw)

    def test_keeps_confident_prefix_and_drops_the_tail(self):
        out = self.select([0.95, 0.9, 0.1])
        assert out["p"].to_list() == [0.95, 0.9]

    def test_single_confident_match_is_kept(self):
        assert self.select([0.9])["p"].to_list() == [0.9]

    def test_unlikely_single_match_becomes_a_singleton(self):
        # P(entity is a singleton) = 0.7 beats the expected F0.5 of predicting the match
        assert self.select([0.3]).height == 0

    def test_probabilities_below_floor_are_ignored(self):
        assert self.select([0.04, 0.03]).height == 0
        assert self.select([0.9, 0.04])["p"].to_list() == [0.9]

    def test_many_confident_records_are_all_kept(self):
        out = self.select([0.99, 0.99, 0.99, 0.98, 0.2])
        assert out["p"].to_list() == [0.99, 0.99, 0.99, 0.98]

    def test_entities_are_decided_independently(self):
        df = pl.DataFrame({"q_rid": [0, 1, 2], "s1_rid": [1, 1, 2], "p": [0.9, 0.8, 0.3]})
        out = M.select_sets(df)
        assert out["s1_rid"].to_list() == [1, 1]
        assert out.columns == ["q_rid", "s1_rid", "p"]

    def test_output_is_sorted_by_probability_within_entity(self):
        df = pl.DataFrame({"q_rid": [0, 1, 2], "s1_rid": [1, 1, 1], "p": [0.92, 0.97, 0.95]})
        assert M.select_sets(df)["q_rid"].to_list() == [1, 2, 0]


def _posteriors(pi_new, pi_train, n=20_000, seed=0):
    """Calibrated-on-pi_train posteriors for a population whose true positive rate is pi_new."""
    rng = np.random.default_rng(seed)
    y = rng.random(n) < pi_new
    s = np.where(y, rng.beta(5, 1.5, n), rng.beta(1.5, 5, n))
    f1, f0 = beta.pdf(s, 5, 1.5), beta.pdf(s, 1.5, 5)
    return pi_train * f1 / (pi_train * f1 + (1 - pi_train) * f0)


class TestEmPrior:
    @pytest.mark.parametrize("pi_new", [0.1, 0.2, 0.5, 0.8])
    def test_recovers_the_prior_of_a_shifted_population(self, pi_new):
        est = M.em_prior(_posteriors(pi_new, 0.5), 0.5)
        assert est == pytest.approx(pi_new, abs=0.02)

    def test_recovers_the_prior_with_an_imbalanced_training_rate(self):
        est = M.em_prior(_posteriors(0.6, 0.3), 0.3)
        assert est == pytest.approx(0.6, abs=0.02)

    def test_unchanged_population_is_a_fixed_point(self):
        assert M.em_prior(np.full(100, 0.3), 0.3) == pytest.approx(0.3)


class TestPriorAdjust:
    def test_rescales_scores_in_the_direction_of_the_shift(self):
        hi, lo = _posteriors(0.8, 0.5, n=2000, seed=1), _posteriors(0.2, 0.5, n=2000, seed=2)
        scored = pairs(range(4000), [0] * 2000 + [1] * 2000, p=np.concatenate([hi, lo]))
        scored = scored.with_columns(pl.col("p").cast(pl.Float32))
        country = pairs([0, 0], [0, 1]).select("s1_rid").with_columns(pl.Series("country", ["HI", "LO"]))
        out = M.prior_adjust(scored, country, {"*": 0.5})
        assert out.columns == scored.columns
        assert out["p"].dtype == pl.Float32
        assert out["p"][:2000].mean() > hi.mean()      # higher positive rate than training: scores go up
        assert out["p"][2000:].mean() < lo.mean()      # lower positive rate: scores go down
        assert out["p"].min() >= 0 and out["p"].max() <= 1
