"""Stage-1 model training smoke test and stage-2 group-consensus features
(src/model.py::train_model, src/stage2.py, src/config.py)."""
import numpy as np
import polars as pl
import pytest

import config
import model as M
import stage2 as S
from conftest import make_records, pairs

RECS = make_records([
    ("Oak Hardware", "100 Oak St", 1),     # 0: S1 entity
    ("Oak Hardware", "100 Oak St", 2),     # 1: confident match
    ("Oak Hardware", "100 Oak St", 2),     # 2: confident match
    ("Oak Hardware", "900 Oak St", 2),     # 3: same name, different house number (look-alike)
    ("Pine Cafe", "", 1),                  # 4: second S1 entity
    ("Pine Cafe", "", 2),                  # 5: below the probability floor
])


@pytest.fixture(scope="module")
def group():
    best = pairs([1, 2, 3, 5], [0, 0, 0, 4], p=[0.9, 0.8, 0.1, 0.01], margin=[0.8, 0.7, 0.1, 0.0])
    best = best.with_columns(pl.col("p", "margin").cast(pl.Float32))
    return S.group_features(best, RECS).sort("q_rid")


class TestGroupFeatures:
    def test_one_row_per_assignment_above_the_floor(self, group):
        assert group["q_rid"].to_list() == [1, 2, 3]            # q5 has p=0.01 < FLOOR
        assert group.columns == ["q_rid", "s1_rid", *S.S2_FEATURES]

    def test_group_size_and_rank(self, group):
        assert group["n_grp"].to_list() == [3, 3, 3]
        assert group["rank_in_s1"].to_list() == [1, 2, 3]
        assert group["p_rel"].to_list() == pytest.approx([1.0, 0.8 / 0.9, 0.1 / 0.9])

    def test_other_members_probabilities(self, group):
        assert group["p_sum_oth"].to_list() == pytest.approx([0.9, 1.0, 1.7])
        assert group["p_max_oth"].to_list() == pytest.approx([0.8, 0.9, 0.9])
        assert group["n_conf_oth"].to_list() == [1, 1, 2]       # others with p >= 0.5

    def test_house_number_agreement_with_the_entity(self, group):
        assert group["num1_eq"].to_list() == [1, 1, 0]
        assert group["q_has_num"].to_list() == [1, 1, 1]

    def test_look_alike_disagrees_with_the_confident_group(self, group):
        look_alike = group.filter(pl.col("q_rid") == 3).row(0, named=True)
        confident = group.filter(pl.col("q_rid") == 1).row(0, named=True)
        assert look_alike["num_same_frac"] == 0 < confident["num_same_frac"]
        assert look_alike["g_s1num_cnt"] == 2 and confident["g_s1num_cnt"] == 1

    def test_empty_address_flag(self):
        best = pairs([5], [4], p=[0.9], margin=[0.9]).with_columns(pl.col("p", "margin").cast(pl.Float32))
        assert S.group_features(best, RECS)["q_addr_empty"].to_list() == [1.0]

    def test_all_features_are_float32_and_finite(self, group):
        assert all(group.schema[f] == pl.Float32 for f in S.S2_FEATURES)
        assert np.isfinite(group.select(S.S2_FEATURES).to_numpy()).all()


class TestStage2Model:
    def test_trains_and_scores_a_separable_problem(self):
        rng = np.random.default_rng(0)
        n = 1500
        y = (rng.random(n) < 0.5).astype(np.float64)
        cols = {f: rng.random(n).astype(np.float32) for f in S.S2_FEATURES}
        cols["p"] = (y * 0.6 + rng.random(n) * 0.4).astype(np.float32)      # informative feature
        gf = pl.DataFrame({"q_rid": np.arange(n, dtype=np.uint32), "s1_rid": np.zeros(n, dtype=np.uint32), **cols})
        booster = S.train_stage2(gf, y, rounds=20)
        scored = S.apply_stage2(booster, gf)
        assert scored.columns == ["q_rid", "s1_rid", "p"]
        assert scored["p"].is_between(0, 1).all()
        p = scored["p"].to_numpy()
        assert p[y == 1].mean() > p[y == 0].mean() + 0.3


class TestStage1Model:
    def test_trains_on_the_declared_feature_set(self):
        from features import FEATURES
        rng = np.random.default_rng(0)
        x = rng.random((600, len(FEATURES))).astype(np.float32)
        y = (x[:, 0] > 0.5).astype(np.float64)
        booster = M.train_model(x, y, rounds=10)
        assert booster.feature_name() == FEATURES
        p = booster.predict(x)
        assert p[y == 1].mean() > p[y == 0].mean()


class TestConfig:
    def test_set_data_dir_overrides_the_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "DATA_DIR", config.DATA_DIR)
        config.set_data_dir(str(tmp_path))
        assert config.DATA_DIR == tmp_path

    def test_directories_live_under_the_project_by_default(self):
        assert config.PROJECT_DIR.name != "src"
        assert (config.PROJECT_DIR / "src" / "config.py").exists()
