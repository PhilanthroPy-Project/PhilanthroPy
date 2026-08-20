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


def test_default_splitter_no_leakage_gap_years_zero():
    """Default gap_years=0: training fold never contains a FY at or after the test FY,
    and each test fold is exactly one fiscal year."""
    X = np.zeros((10, 2))
    splitter = FiscalYearGroupedSplitter(n_splits=3)
    splits = list(splitter.split(X, groups=_FY_GROUPS))
    assert len(splits) == 3

    fy = np.asarray(_FY_GROUPS)
    train_sizes = []
    for train_idx, test_idx in splits:
        # No leakage: max training FY is strictly less than min test FY.
        assert fy[train_idx].max() < fy[test_idx].min()
        # Each test fold is exactly one fiscal year.
        assert len(set(fy[test_idx])) == 1
        train_sizes.append(len(train_idx))

    # Expanding window: training set grows with each split.
    assert train_sizes == sorted(train_sizes)
    assert len(set(train_sizes)) == len(train_sizes)  # strictly increasing


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


# ---------------------------------------------------------------------------
# n_splits validation — split() and get_n_splits() must never disagree
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_n_splits", [0, -1, -3])
def test_non_positive_n_splits_raises_instead_of_slicing(bad_n_splits):
    # Pre-fix, a non-positive n_splits reached `unique_fy[-(n_splits):]`, where
    # it flips the slice open-ended: n_splits=0 yielded 3 folds while
    # get_n_splits() reported 0. cross_val_score sizes its output from
    # get_n_splits(), so the disagreement is a real failure.
    X = np.zeros((100, 3))
    fy = np.array([2019] * 25 + [2020] * 25 + [2021] * 25 + [2022] * 25)
    splitter = FiscalYearGroupedSplitter(n_splits=bad_n_splits)

    with pytest.raises(ValueError, match="n_splits must be >= 1"):
        list(splitter.split(X, groups=fy))
    with pytest.raises(ValueError, match="n_splits must be >= 1"):
        splitter.get_n_splits()


def test_negative_gap_years_raises():
    X = np.zeros((100, 3))
    fy = np.array([2019] * 25 + [2020] * 25 + [2021] * 25 + [2022] * 25)
    with pytest.raises(ValueError, match="gap_years must be >= 0"):
        list(FiscalYearGroupedSplitter(gap_years=-1).split(X, groups=fy))


def test_non_integer_params_raise():
    X = np.zeros((100, 3))
    fy = np.array([2019] * 50 + [2020] * 50)
    with pytest.raises(ValueError, match="must be integers"):
        list(FiscalYearGroupedSplitter(n_splits="three").split(X, groups=fy))


@pytest.mark.parametrize("n_splits", [1, 2, 3])
@pytest.mark.parametrize("gap_years", [0, 1])
def test_get_n_splits_matches_the_folds_actually_yielded(n_splits, gap_years):
    # The invariant sklearn relies on. Assert it directly rather than trusting
    # that the two independent code paths stay in step.
    X = np.zeros((150, 3))
    fy = np.array([2018] * 30 + [2019] * 30 + [2020] * 30 + [2021] * 30 + [2022] * 30)
    splitter = FiscalYearGroupedSplitter(n_splits=n_splits, gap_years=gap_years)
    assert splitter.get_n_splits(groups=fy) == len(list(splitter.split(X, groups=fy)))
