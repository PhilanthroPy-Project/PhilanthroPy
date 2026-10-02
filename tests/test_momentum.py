"""
tests/test_momentum.py
Tests for philanthropy.utils._momentum.trailing_slope_features, the shared
as-of slope/rel_slope helper behind RFMTransformer(include_momentum=True),
build_upgrade_snapshots, and activities_to_features(include_momentum=True).
"""

import numpy as np
import pandas as pd
import pytest

from philanthropy.utils._momentum import trailing_slope_features


def _gifts(rows):
    """rows: iterable of (donor_id, date, amount)."""
    return pd.DataFrame(rows, columns=["donor_id", "gift_date", "gift_amount"])


def test_five_year_toy_donor_hand_computed_slope():
    # Exactly one gift per fiscal year end, strictly linear: +100/year.
    gifts = _gifts([
        ("1", "2018-06-30", 100), ("1", "2019-06-30", 200), ("1", "2020-06-30", 300),
        ("1", "2021-06-30", 400), ("1", "2022-06-30", 500),
    ])
    gifts["gift_date"] = pd.to_datetime(gifts["gift_date"])
    out = trailing_slope_features(
        gifts, pd.Index(["1"]), pd.Timestamp("2022-06-30"),
        date_col="gift_date", value_col="gift_amount", agg="sum", ks=(3, 5), prefix="fy_total",
    )
    assert out.loc["1", "fy_total_slope_5y"] == pytest.approx(100.0)
    assert out.loc["1", "fy_total_rel_slope_5y"] == pytest.approx(100.0 / 301.0)
    assert out.loc["1", "fy_total_slope_3y"] == pytest.approx(100.0)
    assert out.loc["1", "fy_total_rel_slope_3y"] == pytest.approx(100.0 / 401.0)


def test_fewer_than_two_observed_years_is_nan():
    # Only one fiscal year of history exists at all: a 3-year window has at
    # most 1 observed year, so both slopes are NaN.
    gifts = _gifts([("1", "2022-01-15", 500)])
    gifts["gift_date"] = pd.to_datetime(gifts["gift_date"])
    out = trailing_slope_features(
        gifts, pd.Index(["1"]), pd.Timestamp("2022-06-30"),
        date_col="gift_date", value_col="gift_amount", agg="sum", ks=(3,), prefix="fy_total",
    )
    assert np.isnan(out.loc["1", "fy_total_slope_3y"])
    assert np.isnan(out.loc["1", "fy_total_rel_slope_3y"])


def test_data_after_cutoff_is_ignored():
    gifts = _gifts([
        ("1", "2018-06-30", 100), ("1", "2019-06-30", 200), ("1", "2020-06-30", 300),
    ])
    gifts["gift_date"] = pd.to_datetime(gifts["gift_date"])
    before = trailing_slope_features(
        gifts, pd.Index(["1"]), pd.Timestamp("2020-06-30"),
        date_col="gift_date", value_col="gift_amount", agg="sum", ks=(3,), prefix="fy_total",
    )
    gifts_with_future = pd.concat([gifts, _gifts([("1", "2025-01-01", 999999)])], ignore_index=True)
    gifts_with_future["gift_date"] = pd.to_datetime(gifts_with_future["gift_date"])
    after = trailing_slope_features(
        gifts_with_future, pd.Index(["1"]), pd.Timestamp("2020-06-30"),
        date_col="gift_date", value_col="gift_amount", agg="sum", ks=(3,), prefix="fy_total",
    )
    pd.testing.assert_frame_equal(before, after)


def test_bins_before_data_start_excluded_not_zero_filled():
    # A donor with only 2 years of flat giving, scored with a 3- and 5-year
    # window: the periods before data_start must be dropped from the
    # regression entirely, not zero-filled, or a flat history reads as a
    # fabricated upward trend (regression: previously gave +50/+30 here).
    gifts = _gifts([("1", "2021-06-30", 100), ("1", "2022-06-30", 100)])
    gifts["gift_date"] = pd.to_datetime(gifts["gift_date"])
    out = trailing_slope_features(
        gifts, pd.Index(["1"]), pd.Timestamp("2022-06-30"),
        date_col="gift_date", value_col="gift_amount", agg="sum", ks=(3, 5), prefix="fy_total",
        data_start=pd.Timestamp("2021-06-30"),
    )
    assert out.loc["1", "fy_total_slope_3y"] == pytest.approx(0.0)
    assert out.loc["1", "fy_total_slope_5y"] == pytest.approx(0.0)


def test_missing_years_treated_as_zero_not_dropped():
    # Donor gave in year 1 and year 5 only; the 3 years in between are real
    # zeros (the dataset covers them), not missing, and pull the slope down.
    gifts = _gifts([("1", "2018-06-30", 500), ("1", "2022-06-30", 500)])
    gifts["gift_date"] = pd.to_datetime(gifts["gift_date"])
    out = trailing_slope_features(
        gifts, pd.Index(["1"]), pd.Timestamp("2022-06-30"),
        date_col="gift_date", value_col="gift_amount", agg="sum", ks=(5,), prefix="fy_total",
    )
    # values oldest->newest: 500, 0, 0, 0, 500 -> flat trend, slope 0.
    assert out.loc["1", "fy_total_slope_5y"] == pytest.approx(0.0)
