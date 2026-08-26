"""Tests for FiscalYearGroupedSplitter input-validation and integration paths."""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score

from philanthropy.model_selection import FiscalYearGroupedSplitter


def test_groups_none_raises():
    s = FiscalYearGroupedSplitter(drop_repeat_donors=False, n_splits=2)
    with pytest.raises(ValueError, match="groups"):
        list(s.split(np.zeros((10, 2))))


def test_length_mismatch_raises():
    s = FiscalYearGroupedSplitter(drop_repeat_donors=False, n_splits=2)
    with pytest.raises(ValueError, match="match"):
        list(s.split(np.zeros((10, 2)), groups=[2020, 2021, 2022]))


def test_single_fiscal_year_raises():
    s = FiscalYearGroupedSplitter(drop_repeat_donors=False, n_splits=2)
    with pytest.raises(ValueError, match="at least 2"):
        list(s.split(np.zeros((10, 2)), groups=[2020] * 10))


def test_get_n_splits_without_groups():
    assert FiscalYearGroupedSplitter(drop_repeat_donors=False, n_splits=4).get_n_splits() == 4


def test_cross_val_score_integration():
    fy = np.array([2018] * 40 + [2019] * 50 + [2020] * 55 + [2021] * 30 + [2022] * 25)
    X = np.zeros((len(fy), 3))
    y = np.random.default_rng(0).integers(0, 2, len(fy))
    s = FiscalYearGroupedSplitter(drop_repeat_donors=False, n_splits=3)
    scores = cross_val_score(
        DummyClassifier(strategy="most_frequent"), X, y, cv=s, groups=fy
    )
    assert len(scores) == 3


def test_n_samples_list_input():
    s = FiscalYearGroupedSplitter(drop_repeat_donors=False, n_splits=2)
    X = [[0, 0]] * 10  # plain list, no .shape attribute
    groups = [2019] * 3 + [2020] * 3 + [2021] * 4
    assert len(list(s.split(X, groups=groups))) == 2


def test_n_samples_none_raises():
    s = FiscalYearGroupedSplitter(drop_repeat_donors=False)
    with pytest.raises(ValueError):
        list(s.split(None, groups=[2019, 2020]))


# --------------------------------------------------------------------------- #
# Moved here from the deleted tests/test_coverage_boost.py.
# --------------------------------------------------------------------------- #
_FY_GROUPS = [2018, 2018, 2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022]


def test_gap_years_withholds_the_year_before_each_test_fold():
    X = np.zeros((10, 2))
    splitter = FiscalYearGroupedSplitter(drop_repeat_donors=False, n_splits=2, gap_years=1)
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
    splitter = FiscalYearGroupedSplitter(drop_repeat_donors=False, n_splits=3)
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
        list(FiscalYearGroupedSplitter(drop_repeat_donors=False, n_splits=2, gap_years=5).split(
            X, groups=_FY_GROUPS
        ))


def test_repr_and_get_n_splits_reflect_the_groups():
    with pytest.warns(DeprecationWarning):
        splitter = FiscalYearGroupedSplitter(n_splits=2, gap_years=1)
    assert repr(splitter) == (
        "FiscalYearGroupedSplitter(n_splits=2, gap_years=1, "
        "drop_repeat_donors='warn')"
    )
    assert splitter.get_n_splits(groups=_FY_GROUPS) == 2


# ---------------------------------------------------------------------------
# n_splits validation: split() and get_n_splits() must never disagree
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_n_splits", [0, -1, -3])
def test_non_positive_n_splits_raises_instead_of_slicing(bad_n_splits):
    # Pre-fix, a non-positive n_splits reached `unique_fy[-(n_splits):]`, where
    # it flips the slice open-ended: n_splits=0 yielded 3 folds while
    # get_n_splits() reported 0. cross_val_score sizes its output from
    # get_n_splits(), so the disagreement is a real failure.
    X = np.zeros((100, 3))
    fy = np.array([2019] * 25 + [2020] * 25 + [2021] * 25 + [2022] * 25)
    splitter = FiscalYearGroupedSplitter(drop_repeat_donors=False, n_splits=bad_n_splits)

    with pytest.raises(ValueError, match="n_splits must be >= 1"):
        list(splitter.split(X, groups=fy))
    with pytest.raises(ValueError, match="n_splits must be >= 1"):
        splitter.get_n_splits()


def test_negative_gap_years_raises():
    X = np.zeros((100, 3))
    fy = np.array([2019] * 25 + [2020] * 25 + [2021] * 25 + [2022] * 25)
    with pytest.raises(ValueError, match="gap_years must be >= 0"):
        list(FiscalYearGroupedSplitter(drop_repeat_donors=False, gap_years=-1).split(X, groups=fy))


def test_non_integer_params_raise():
    X = np.zeros((100, 3))
    fy = np.array([2019] * 50 + [2020] * 50)
    with pytest.raises(ValueError, match="must be integers"):
        list(FiscalYearGroupedSplitter(drop_repeat_donors=False, n_splits="three").split(X, groups=fy))


@pytest.mark.parametrize("n_splits", [1, 2, 3])
@pytest.mark.parametrize("gap_years", [0, 1])
def test_get_n_splits_matches_the_folds_actually_yielded(n_splits, gap_years):
    # The invariant sklearn relies on. Assert it directly rather than trusting
    # that the two independent code paths stay in step.
    X = np.zeros((150, 3))
    fy = np.array([2018] * 30 + [2019] * 30 + [2020] * 30 + [2021] * 30 + [2022] * 30)
    splitter = FiscalYearGroupedSplitter(drop_repeat_donors=False, n_splits=n_splits, gap_years=gap_years)
    assert splitter.get_n_splits(groups=fy) == len(list(splitter.split(X, groups=fy)))


# ---------------------------------------------------------------------------
# drop_repeat_donors: the static-per-donor-label case
# ---------------------------------------------------------------------------

def _repeat_donor_panel():
    """Donors 1-3 recur every year; the rest are new to the file each year."""
    fy, donor = [], []
    for year in (2019, 2020, 2021, 2022):
        for d in (1, 2, 3):
            fy.append(year)
            donor.append(d)
        for k in range(3):
            fy.append(year)
            donor.append(year * 10 + k)
    fy = np.array(fy)
    donor = np.array(donor)
    return np.zeros((len(fy), 2)), fy, donor


def test_default_leaves_repeat_donors_in_both_folds():
    # Documented and correct for a time-varying target; this pins the default so
    # the new flag cannot quietly become the default later.
    X, fy, donor = _repeat_donor_panel()
    with pytest.warns(DeprecationWarning):
        for train, test in FiscalYearGroupedSplitter(n_splits=2).split(X, groups=fy):
            assert set(donor[train]) & set(donor[test]) == {1, 2, 3}


def test_drop_repeat_donors_removes_the_overlap():
    X, fy, donor = _repeat_donor_panel()
    groups = np.column_stack([fy, donor])
    splitter = FiscalYearGroupedSplitter(n_splits=2, drop_repeat_donors=True)

    with pytest.warns(UserWarning, match="removed 3 test row"):
        folds = list(splitter.split(X, groups=groups))

    for train, test in folds:
        assert not set(donor[train]) & set(donor[test])
        assert len(test) > 0
        # Training rows are never dropped: history is kept in full.
        assert np.all(fy[train] < fy[test].min())


def test_drop_repeat_donors_keeps_get_n_splits_in_step():
    X, fy, donor = _repeat_donor_panel()
    groups = np.column_stack([fy, donor])
    splitter = FiscalYearGroupedSplitter(n_splits=2, drop_repeat_donors=True)
    with pytest.warns(UserWarning):
        n_folds = len(list(splitter.split(X, groups=groups)))
    assert splitter.get_n_splits(groups=groups) == n_folds


def test_drop_repeat_donors_requires_two_column_groups():
    X, fy, _ = _repeat_donor_panel()
    splitter = FiscalYearGroupedSplitter(n_splits=2, drop_repeat_donors=True)
    with pytest.raises(ValueError, match=r"shape \(n_samples, 2\)"):
        list(splitter.split(X, groups=fy))


def test_drop_repeat_donors_raises_rather_than_silently_dropping_a_fold():
    # Every donor recurs, so the test fold empties. Skipping it would put split
    # and get_n_splits back out of step, so it raises with an actionable message.
    groups = np.column_stack([[2019, 2019, 2020, 2020], [1, 2, 1, 2]])
    splitter = FiscalYearGroupedSplitter(n_splits=1, drop_repeat_donors=True)
    with pytest.raises(ValueError, match="emptied the test fold"):
        list(splitter.split(np.zeros((4, 2)), groups=groups))


def test_default_drop_repeat_donors_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="default of False"):
        FiscalYearGroupedSplitter()


def test_explicit_drop_repeat_donors_false_silences_warning():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        FiscalYearGroupedSplitter(drop_repeat_donors=False)
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)


def test_drop_repeat_donors_is_off_by_default():
    # BaseCrossValidator, not BaseEstimator, so there is no get_params here.
    with pytest.warns(DeprecationWarning):
        assert FiscalYearGroupedSplitter().drop_repeat_donors == "warn"


def test_repr_distinguishes_splitters_that_behave_differently():
    # This test used to assert the opposite, that drop_repeat_donors was absent
    # from __repr__. That pinned a defect: two splitters that split differently
    # printed identically, which is exactly what a repr exists to prevent.
    with pytest.warns(DeprecationWarning):
        assert "drop_repeat_donors='warn'" in repr(FiscalYearGroupedSplitter())
    assert "drop_repeat_donors=True" in repr(
        FiscalYearGroupedSplitter(drop_repeat_donors=True)
    )
    with pytest.warns(DeprecationWarning):
        assert repr(FiscalYearGroupedSplitter()) != repr(
            FiscalYearGroupedSplitter(drop_repeat_donors=True)
        )


def test_missing_donor_id_is_treated_as_already_seen():
    # np.isin never matches NaN to NaN, so a row with no donor id would have
    # been kept in the test fold. For a leakage guard that is the wrong default:
    # an unidentifiable donor cannot be shown to be absent from training.
    groups = np.column_stack([
        [2019.0, 2019.0, 2020.0, 2020.0, 2020.0],
        [1.0, 2.0, 3.0, np.nan, 4.0],
    ])
    X = np.zeros((5, 2))
    splitter = FiscalYearGroupedSplitter(n_splits=1, drop_repeat_donors=True)
    with pytest.warns(UserWarning, match="removed 1 test row"):
        folds = list(splitter.split(X, groups=groups))
    (train, test), = folds
    # donors 3 and 4 are new, the NaN row is dropped despite being "unseen".
    assert sorted(test) == [2, 4]


def test_string_donor_ids_still_split_correctly():
    # np.column_stack of int years and string ids upcasts everything to '<U21',
    # so the fiscal years arrive as strings. That used to fail later with a bare
    # numpy TypeError on `fiscal_years < cutoff`; the years are now coerced back
    # to float, so this is a working case rather than an error case.
    groups = np.column_stack([[2019, 2019, 2020], ["a", "b", "c"]])
    assert groups.dtype.kind == "U"
    splitter = FiscalYearGroupedSplitter(n_splits=1, drop_repeat_donors=True)
    (train, test), = list(splitter.split(np.zeros((3, 2)), groups=groups))
    assert sorted(train) == [0, 1]     # 2019
    assert sorted(test) == [2]         # 2020, donor "c" is new


def test_non_numeric_fiscal_years_raise_a_useful_error():
    groups = np.array([["FY19", "a"], ["FY19", "b"], ["FY20", "c"]])
    splitter = FiscalYearGroupedSplitter(n_splits=1, drop_repeat_donors=True)
    with pytest.raises(ValueError, match="must be numeric fiscal years"):
        list(splitter.split(np.zeros((3, 2)), groups=groups))


def test_row_loss_warning_fires_on_the_first_fold_not_at_exhaustion():
    # The warning used to sit after the loop in a generator, so a caller taking
    # only the first fold never saw it and lost rows silently.
    X, fy, donor = _repeat_donor_panel()
    groups = np.column_stack([fy, donor])
    gen = FiscalYearGroupedSplitter(n_splits=2, drop_repeat_donors=True).split(
        X, groups=groups
    )
    with pytest.warns(UserWarning, match="removed 3 test row"):
        next(gen)
