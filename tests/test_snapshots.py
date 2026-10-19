"""Tests for philanthropy.ingest.build_snapshots / period_snapshots."""

import numpy as np
import pandas as pd
import pytest

from philanthropy.ingest import build_snapshots
from philanthropy.ingest._snapshots import CORE_COLUMNS, SCALE_FREE_COLUMNS, period_snapshots


def _gifts():
    # FY starts in July: 2018-08 is FY2019. Donor a gives FY19-21, b FY19-20
    # (lapses in FY21), c only FY20, d FY19 and FY21 (a gap year).
    rows = [
        ("a", "2018-08-01", 100), ("a", "2019-08-01", 150), ("a", "2020-08-01", 1200),
        ("b", "2018-08-01", 300), ("b", "2019-09-01", 200), ("b", "2019-10-01", 50),
        ("c", "2019-08-01", 40),
        ("d", "2018-08-01", 60), ("d", "2020-08-01", 70),
    ]
    return pd.DataFrame(rows, columns=["donor_id", "gift_date", "gift_amount"])


def test_lapse_population_and_label():
    snaps = build_snapshots(_gifts(), kind="lapse")
    fy20 = snaps[snaps["period"] == 2020]
    assert sorted(fy20.index) == ["a", "b", "c"]
    assert fy20["target"].to_dict() == {"a": 0, "b": 1, "c": 1}
    # FY2021 is the last year on file, so it has no label and no rows.
    assert snaps["period"].max() == 2020


def test_min_years_given_restricts_to_multi_year_donors():
    snaps = build_snapshots(_gifts(), kind="lapse", min_years_given=2)
    assert sorted(snaps.index) == ["a", "b"]
    assert (snaps["period"] == 2020).all()


def test_core_features_are_as_of_the_snapshot_period():
    snaps = build_snapshots(_gifts(), kind="lapse")
    b = snaps[snaps["period"] == 2020].loc["b"]
    assert b["period_total"] == 250 and b["period_total_prior1"] == 300 and b["period_trend"] == -50
    assert b["largest_gift"] == 200 and b["gift_count"] == 2
    assert b["consecutive_periods_given"] == 2 and b["gave_prior1"] == 1 and b["periods_since_first_gift"] == 1
    assert b["months_since_last_gift"] == pytest.approx(round((pd.Timestamp("2020-06-30") - pd.Timestamp("2019-10-01")).days / 30.0, 2))


def test_future_gifts_never_change_features():
    base = build_snapshots(_gifts(), kind="lapse")
    later = _gifts()
    later.loc[later["gift_date"] == "2020-08-01", "gift_amount"] = 99999
    changed = build_snapshots(later, kind="lapse")
    feats = [c for c in base.columns if c != "target"]
    pd.testing.assert_frame_equal(base[feats], changed[feats])


@pytest.mark.parametrize("kind", ["upgrade", "response_next_year", "next_amount"])
def test_other_kinds(kind):
    snaps = build_snapshots(_gifts(), kind=kind, threshold=1000, band=(100, 999))
    fy20 = snaps[snaps["period"] == 2020]
    if kind == "upgrade":
        assert fy20["target"].to_dict() == {"a": 1, "b": 0}
    elif kind == "response_next_year":
        # d gave in FY19, not FY20, and is still a past donor who can respond.
        assert fy20["target"].to_dict() == {"a": 1, "b": 0, "c": 0, "d": 1}
    else:
        assert fy20["target"].to_dict() == {"a": 1200.0}


def test_period_table_prior_means_previous_observed_period_and_attrition_is_unknown():
    # Biennial waves; household h2 is not observed in 2015, h3 drops out in 2017.
    totals = pd.DataFrame(
        {2011: [100.0, 50.0, 10.0], 2013: [200.0, 0.0, 20.0], 2015: [0.0, np.nan, 30.0], 2017: [10.0, 5.0, np.nan]},
        index=pd.Index(["h1", "h2", "h3"], name="household_key"),
    )
    snaps = period_snapshots(totals, kind="lapse")
    w13 = snaps[snaps["period"] == 2013]
    assert list(w13.index) == ["h1", "h3"]
    assert w13.loc["h1", "period_total_prior1"] == 100.0 and w13.loc["h1", "target"] == 1
    w15 = snaps[snaps["period"] == 2015]
    assert list(w15.index) == []
    assert set(CORE_COLUMNS) <= set(snaps.columns)


def test_scale_free_block():
    snaps = build_snapshots(_gifts(), kind="lapse", scale_free=True)
    assert set(SCALE_FREE_COLUMNS) <= set(snaps.columns)
    b = snaps[snaps["period"] == 2020].loc["b"]
    assert b["total_over_largest"] == pytest.approx(250 / 200)
    assert b["gifts_per_streak_period"] == pytest.approx(1.0)
    assert "total_over_largest" not in build_snapshots(_gifts(), kind="lapse").columns


def test_bad_arguments():
    with pytest.raises(ValueError, match="kind"):
        build_snapshots(_gifts(), kind="churn")
    with pytest.raises(ValueError, match="min_years_given"):
        build_snapshots(_gifts(), kind="lapse", min_years_given=0)
    assert build_snapshots(_gifts().iloc[:0], kind="lapse").empty
