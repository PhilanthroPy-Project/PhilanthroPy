"""
tests/test_upgrade_snapshots.py
Tests for philanthropy.ingest.build_upgrade_snapshots.

The population for a snapshot year T is the "upgrade band": donors giving
between band[0] and band[1] in FY T, and strictly below threshold even if
band[1] reaches or exceeds it. Every feature is computed from data through
the end of FY T; target alone reads FY T+1. Most tests below are built
around proving that boundary never moves.
"""

import numpy as np
import pandas as pd
import pytest

from philanthropy.ingest import build_upgrade_snapshots
from philanthropy.model_selection import FiscalYearGroupedSplitter

# fiscal_year_start=7 (the default): FY N spans N-1's July 1 to N's June 30.
FY2018 = "2017-08-01"  # -> fiscal_year 2018
FY2019 = "2018-08-01"  # -> fiscal_year 2019
FY2020 = "2019-08-01"  # -> fiscal_year 2020
FY2021 = "2020-08-01"  # -> fiscal_year 2021
FY2022 = "2021-08-01"  # -> fiscal_year 2022


def _gifts(rows):
    """rows: iterable of (donor_id, date, amount)."""
    return pd.DataFrame(rows, columns=["donor_id", "gift_date", "gift_amount"])


# --------------------------------------------------------------------------- #
# Band population
# --------------------------------------------------------------------------- #
def test_donor_inside_band_qualifies():
    gifts = _gifts([("1", FY2020, 500)])
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020])
    assert list(snaps.index) == ["1"]


def test_donor_below_band_excluded():
    gifts = _gifts([("1", FY2020, 50)])
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020])
    assert snaps.empty


def test_band_bounds_are_inclusive():
    gifts = _gifts([("1", FY2020, 100), ("2", FY2020, 999)])
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020])
    assert set(snaps.index) == {"1", "2"}


def test_donor_at_threshold_excluded():
    gifts = _gifts([("1", FY2020, 1000)])
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020])
    assert snaps.empty


def test_donor_above_threshold_excluded():
    gifts = _gifts([("1", FY2020, 5000)])
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020])
    assert snaps.empty


def test_threshold_excludes_even_when_band_is_wider():
    # band[1] alone would admit 1200; threshold must still exclude it.
    gifts = _gifts([("1", FY2020, 999), ("2", FY2020, 1200), ("3", FY2020, 1500)])
    snaps = build_upgrade_snapshots(
        gifts, fiscal_years=[2020], threshold=1000, band=(100, 1500)
    )
    assert set(snaps.index) == {"1"}


def test_donor_with_no_gifts_at_all_never_appears():
    gifts = _gifts([("1", FY2020, 500)])
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020])
    assert "ghost" not in snaps.index


# --------------------------------------------------------------------------- #
# fiscal_year column and multi-year stacking
# --------------------------------------------------------------------------- #
def test_fiscal_year_column_is_the_snapshot_year_not_t_plus_1():
    gifts = _gifts([("1", FY2020, 500)])
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020])
    assert int(snaps.loc["1", "fiscal_year"]) == 2020


def test_multiple_fiscal_years_stack_rows_for_the_same_donor():
    gifts = _gifts([("1", FY2020, 500), ("1", FY2021, 300)])
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020, 2021])
    assert list(snaps.index) == ["1", "1"]
    assert sorted(snaps["fiscal_year"].tolist()) == [2020, 2021]


def test_year_with_no_candidates_contributes_no_rows():
    gifts = _gifts([("1", FY2020, 500)])
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2019, 2020])
    assert list(snaps["fiscal_year"]) == [2020]


def test_every_gift_row_unparseable_returns_empty_frame():
    gifts = _gifts([("1", "not-a-date", 500)])
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020])
    assert snaps.empty
    assert snaps.index.name == "donor_id"


def test_no_year_has_any_candidate_returns_empty_frame():
    gifts = _gifts([("1", FY2020, 5000)])
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2019, 2020])
    assert snaps.empty
    assert snaps.index.name == "donor_id"


def test_rows_sorted_by_fiscal_year_then_donor_id():
    gifts = _gifts([("9", FY2021, 500), ("1", FY2021, 500), ("1", FY2020, 500)])
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020, 2021])
    assert list(zip(snaps["fiscal_year"], snaps.index)) == [
        (2020, "1"),
        (2021, "1"),
        (2021, "9"),
    ]


# --------------------------------------------------------------------------- #
# Target: reads FY T+1 only
# --------------------------------------------------------------------------- #
def test_target_1_when_next_year_reaches_threshold():
    gifts = _gifts([("1", FY2020, 500), ("1", FY2021, 1500)])
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020])
    assert int(snaps.loc["1", "target"]) == 1


def test_target_0_when_next_year_stays_below_threshold():
    gifts = _gifts([("1", FY2020, 500), ("1", FY2021, 600)])
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020])
    assert int(snaps.loc["1", "target"]) == 0


def test_target_0_when_no_gift_at_all_in_next_year():
    gifts = _gifts([("1", FY2020, 500)])
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020])
    assert int(snaps.loc["1", "target"]) == 0
    assert not pd.isna(snaps.loc["1", "target"])


def test_target_reads_t_plus_1_and_nothing_further():
    # A huge gift two years out must not affect target for T.
    gifts = _gifts([("1", FY2020, 500), ("1", FY2022, 100_000)])
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020])
    assert int(snaps.loc["1", "target"]) == 0


def test_appending_a_t_plus_1_gift_can_flip_target():
    base = _gifts([("1", FY2020, 500)])
    before = build_upgrade_snapshots(base, fiscal_years=[2020])
    after_gifts = pd.concat([base, _gifts([("1", FY2021, 1500)])], ignore_index=True)
    after = build_upgrade_snapshots(after_gifts, fiscal_years=[2020])

    assert int(before.loc["1", "target"]) == 0
    assert int(after.loc["1", "target"]) == 1
    # Every feature column (everything but target) is untouched by the new
    # FY T+1 row: only the FY T+1-derived target may change.
    feature_cols = [c for c in before.columns if c != "target"]
    pd.testing.assert_frame_equal(before[feature_cols], after[feature_cols])


# --------------------------------------------------------------------------- #
# Leakage: features never see FY T+1 or later
# --------------------------------------------------------------------------- #
def test_features_unaffected_by_size_of_t_plus_1_gift():
    small = _gifts([("1", FY2020, 500), ("1", FY2021, 1)])
    large = _gifts([("1", FY2020, 500), ("1", FY2021, 999_999)])
    snaps_small = build_upgrade_snapshots(small, fiscal_years=[2020])
    snaps_large = build_upgrade_snapshots(large, fiscal_years=[2020])

    feature_cols = [c for c in snaps_small.columns if c != "target"]
    pd.testing.assert_frame_equal(
        snaps_small[feature_cols], snaps_large[feature_cols]
    )


def test_features_unaffected_by_rows_appended_at_t_plus_2_or_later():
    before = _gifts([("1", FY2020, 500)])
    after = pd.concat(
        [before, _gifts([("1", FY2022, 100_000)])], ignore_index=True
    )
    snaps_before = build_upgrade_snapshots(before, fiscal_years=[2020])
    snaps_after = build_upgrade_snapshots(after, fiscal_years=[2020])

    pd.testing.assert_frame_equal(snaps_before, snaps_after)


def test_idempotent_on_repeat():
    gifts = _gifts(
        [("1", FY2018, 200), ("1", FY2019, 300), ("1", FY2020, 500), ("1", FY2021, 1500)]
    )
    first = build_upgrade_snapshots(gifts, fiscal_years=[2020])
    second = build_upgrade_snapshots(gifts, fiscal_years=[2020])
    pd.testing.assert_frame_equal(first, second)


# --------------------------------------------------------------------------- #
# Gift-derived features
# --------------------------------------------------------------------------- #
def test_fy_totals_and_trend():
    gifts = _gifts(
        [("1", FY2018, 200), ("1", FY2019, 300), ("1", FY2020, 500)]
    )
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020])
    row = snaps.loc["1"]

    assert float(row["fy_total"]) == 500.0
    assert float(row["fy_total_prior1"]) == 300.0
    assert float(row["fy_total_prior2"]) == 200.0
    assert float(row["fy_trend"]) == 200.0


def test_prior_year_totals_default_to_zero_without_history():
    gifts = _gifts([("1", FY2020, 500)])
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020])
    row = snaps.loc["1"]

    assert float(row["fy_total_prior1"]) == 0.0
    assert float(row["fy_total_prior2"]) == 0.0


def test_largest_gift_and_gift_count():
    gifts = _gifts([("1", FY2020, 300), ("1", "2020-01-15", 200)])
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020])
    row = snaps.loc["1"]

    assert float(row["fy_total"]) == 500.0
    assert float(row["largest_gift"]) == 300.0
    assert int(row["gift_count"]) == 2


def test_consecutive_years_given():
    gifts = _gifts(
        [("1", FY2018, 200), ("1", FY2019, 300), ("1", FY2020, 500)]
    )
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020])
    assert int(snaps.loc["1", "consecutive_years_given"]) == 3


def test_consecutive_years_given_breaks_on_a_gap():
    # No gift at all in FY2019 breaks the streak before FY2018.
    gifts = _gifts([("1", FY2018, 200), ("1", FY2020, 500)])
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020])
    assert int(snaps.loc["1", "consecutive_years_given"]) == 1


def test_months_since_last_gift():
    gifts = _gifts([("1", FY2019, 300), ("1", FY2020, 500)])
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020])
    # Last gift on or before end of FY2020 (2020-06-30) is the FY2020 gift
    # itself, dated 2019-08-01.
    expected_days = (pd.Timestamp("2020-06-30") - pd.Timestamp(FY2020)).days
    assert float(snaps.loc["1", "months_since_last_gift"]) == round(
        expected_days / 30.0, 2
    )


# --------------------------------------------------------------------------- #
# activities integration
# --------------------------------------------------------------------------- #
def test_activities_columns_are_joined_in():
    gifts = _gifts([("1", FY2020, 500)])
    activities = [
        {"contact_id": "1", "activity_date": FY2020, "activity_type": "event"},
    ]
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020], activities=activities)
    assert "event_count_12m" in snaps.columns


def test_activities_cutoff_matches_end_of_fiscal_year_t():
    gifts = _gifts([("1", FY2020, 500)])
    # This event is dated in FY2021, after the end of FY2020: must not be
    # counted in the FY2020 snapshot's activity features.
    activities = [
        {"contact_id": "1", "activity_date": FY2021, "activity_type": "event"},
    ]
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020], activities=activities)
    assert "event_count_12m" not in snaps.columns


def test_donor_missing_from_activities_gets_nan_activity_columns():
    gifts = _gifts([("1", FY2020, 500), ("2", FY2020, 200)])
    activities = [
        {"contact_id": "1", "activity_date": FY2020, "activity_type": "event"},
    ]
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020], activities=activities)
    assert pd.isna(snaps.loc["2", "event_count_12m"])


# --------------------------------------------------------------------------- #
# donors integration
# --------------------------------------------------------------------------- #
def test_donor_attribute_columns_are_joined_in():
    gifts = _gifts([("1", FY2020, 500)])
    donors = pd.DataFrame({"wealth_rating": ["A"]}, index=pd.Index(["1"], name="donor_id"))
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020], donors=donors)
    assert snaps.loc["1", "wealth_rating"] == "A"


def test_donors_does_not_add_rows_for_non_candidates():
    gifts = _gifts([("1", FY2020, 500)])
    donors = pd.DataFrame(
        {"wealth_rating": ["A", "B"]}, index=pd.Index(["1", "2"], name="donor_id")
    )
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020], donors=donors)
    assert list(snaps.index) == ["1"]


# --------------------------------------------------------------------------- #
# FiscalYearGroupedSplitter integration
# --------------------------------------------------------------------------- #
def test_output_feeds_fiscal_year_grouped_splitter():
    # Three donors, each qualifying in exactly one distinct fiscal year, so
    # every test fold's donor is unseen in training and drop_repeat_donors's
    # default guard against an emptied fold never fires.
    gifts = _gifts(
        [("A", FY2020, 500), ("B", FY2021, 500), ("C", FY2022, 500)]
    )
    snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020, 2021, 2022])
    assert len(snaps) == 3

    groups = snaps.reset_index()[["fiscal_year", "donor_id"]].to_numpy()
    splitter = FiscalYearGroupedSplitter(n_splits=2)
    X = np.zeros((len(snaps), 1))

    splits = list(splitter.split(X, groups=groups))
    assert len(splits) == 2
    for train_idx, test_idx in splits:
        assert len(test_idx) > 0
        assert len(train_idx) > 0
        train_fy = snaps["fiscal_year"].to_numpy()[train_idx]
        test_fy = snaps["fiscal_year"].to_numpy()[test_idx]
        assert train_fy.max() < test_fy.min()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_missing_required_column_raises():
    gifts = pd.DataFrame({"donor_id": ["1"], "gift_date": [FY2020]})
    with pytest.raises(KeyError, match="gift_amount"):
        build_upgrade_snapshots(gifts, fiscal_years=[2020])


def test_invalid_fiscal_year_start_raises():
    gifts = _gifts([("1", FY2020, 500)])
    with pytest.raises(ValueError):
        build_upgrade_snapshots(gifts, fiscal_years=[2020], fiscal_year_start=13)


def test_band_low_above_high_raises():
    gifts = _gifts([("1", FY2020, 500)])
    with pytest.raises(ValueError):
        build_upgrade_snapshots(gifts, fiscal_years=[2020], band=(999, 100))


def test_dataframe_and_iterable_of_mappings_both_accepted():
    rows = [{"donor_id": "1", "gift_date": FY2020, "gift_amount": 500}]
    from_df = build_upgrade_snapshots(pd.DataFrame(rows), fiscal_years=[2020])
    from_iter = build_upgrade_snapshots(rows, fiscal_years=[2020])
    pd.testing.assert_frame_equal(from_df, from_iter)
