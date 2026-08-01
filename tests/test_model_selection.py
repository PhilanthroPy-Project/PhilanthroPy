"""Tests for FiscalYearGroupedSplitter input-validation and integration paths."""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score

from philanthropy.model_selection import FiscalYearGroupedSplitter


def test_groups_none_raises():
    s = FiscalYearGroupedSplitter(n_splits=2)
    with pytest.raises(ValueError, match="groups"):
        list(s.split(np.zeros((10, 2))))


def test_length_mismatch_raises():
    s = FiscalYearGroupedSplitter(n_splits=2)
    with pytest.raises(ValueError, match="match"):
        list(s.split(np.zeros((10, 2)), groups=[2020, 2021, 2022]))


def test_single_fiscal_year_raises():
    s = FiscalYearGroupedSplitter(n_splits=2)
    with pytest.raises(ValueError, match="at least 2"):
        list(s.split(np.zeros((10, 2)), groups=[2020] * 10))


def test_get_n_splits_without_groups():
    assert FiscalYearGroupedSplitter(n_splits=4).get_n_splits() == 4


def test_cross_val_score_integration():
    fy = np.array([2018] * 40 + [2019] * 50 + [2020] * 55 + [2021] * 30 + [2022] * 25)
    X = np.zeros((len(fy), 3))
    y = np.random.default_rng(0).integers(0, 2, len(fy))
    s = FiscalYearGroupedSplitter(n_splits=3)
    scores = cross_val_score(
        DummyClassifier(strategy="most_frequent"), X, y, cv=s, groups=fy
    )
    assert len(scores) == 3


def test_n_samples_list_input():
    s = FiscalYearGroupedSplitter(n_splits=2)
    X = [[0, 0]] * 10  # plain list, no .shape attribute
    groups = [2019] * 3 + [2020] * 3 + [2021] * 4
    assert len(list(s.split(X, groups=groups))) == 2


def test_n_samples_none_raises():
    s = FiscalYearGroupedSplitter()
    with pytest.raises(ValueError):
        list(s.split(None, groups=[2019, 2020]))


# --------------------------------------------------------------------------- #
# Moved here from the deleted tests/test_coverage_boost.py.
# --------------------------------------------------------------------------- #
_FY_GROUPS = [2018, 2018, 2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022]


def test_gap_years_withholds_the_year_before_each_test_fold():
    X = np.zeros((10, 2))
    splitter = FiscalYearGroupedSplitter(n_splits=2, gap_years=1)
    splits = list(splitter.split(X, groups=_FY_GROUPS))
    assert len(splits) == 2

    fy = np.asarray(_FY_GROUPS)
    for train_idx, test_idx in splits:
        test_fy = fy[test_idx].min()
        # gap_years=1 means the fiscal year immediately before the test year is
        # excluded from training, not merely that train < test.
        assert fy[train_idx].max() < test_fy - 1


def test_not_enough_fiscal_years_names_the_shortfall():
    X = np.zeros((10, 2))
    with pytest.raises(
        ValueError,
        match=r"Not enough fiscal years \(5\) for n_splits=2 with gap_years=5\.\s+"
              r"Need at least 8 distinct fiscal years\.",
    ):
        list(FiscalYearGroupedSplitter(n_splits=2, gap_years=5).split(
            X, groups=_FY_GROUPS
        ))


def test_repr_and_get_n_splits_reflect_the_groups():
    splitter = FiscalYearGroupedSplitter(n_splits=2, gap_years=1)
    assert repr(splitter) == (
        "FiscalYearGroupedSplitter(n_splits=2, gap_years=1)"
    )
    assert splitter.get_n_splits(groups=_FY_GROUPS) == 2
