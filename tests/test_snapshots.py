"""Tests for philanthropy.ingest.build_snapshots / period_snapshots."""

import numpy as np
import pandas as pd
import pytest

from philanthropy.ingest import build_snapshots
from philanthropy.ingest._snapshots import CADENCE_COLUMNS, CORE_COLUMNS, SCALE_FREE_COLUMNS, period_snapshots


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


def test_cadence_block():
    snaps = build_snapshots(_gifts(), kind="response_next_year", cadence=True)
    assert set(CADENCE_COLUMNS) <= set(snaps.columns)
    fy20 = snaps[snaps["period"] == 2020]
    b, d = fy20.loc["b"], fy20.loc["d"]
    # b gave FY19 (1 gift) and FY20 (2 gifts); FY18 is not in the file.
    assert b["periods_given"] == 2 and b["share_periods_given"] == 1.0
    assert b["period_total_slope"] == pytest.approx(125.0)
    assert b["gift_count_prior1"] == 1 and b["gift_count_trend"] == 1
    # d gave FY19, skipped FY20: one giving year out of two since the first.
    assert d["periods_given"] == 1 and d["share_periods_given"] == 0.5
    assert d["gift_count_prior1"] == 1 and d["gift_count_trend"] == -1
    assert "periods_given" not in build_snapshots(_gifts(), kind="lapse").columns


def test_cadence_ignores_future_gifts():
    base = build_snapshots(_gifts(), kind="response_next_year", cadence=True)
    later = pd.concat([_gifts(), pd.DataFrame(
        [("b", "2020-09-01", 10), ("b", "2020-10-01", 10), ("d", "2020-11-01", 5)],
        columns=["donor_id", "gift_date", "gift_amount"],
    )])
    changed = build_snapshots(later, kind="response_next_year", cadence=True)
    pd.testing.assert_frame_equal(base[list(CADENCE_COLUMNS)], changed[list(CADENCE_COLUMNS)])


def test_period_table_cadence_without_counts():
    totals = pd.DataFrame({1: [10.0, 0.0], 2: [0.0, 5.0], 3: [20.0, 5.0], 4: [0.0, 0.0]}, index=["x", "y"])
    snap = period_snapshots(totals, kind="lapse", cadence=True)
    x = snap[snap["period"] == 3].loc["x"]
    assert x["periods_given"] == 2 and x["share_periods_given"] == pytest.approx(2 / 3)
    assert x["period_total_slope"] == pytest.approx(5.0)
    assert "gift_count_prior1" not in snap.columns


def test_bad_arguments():
    with pytest.raises(ValueError, match="kind"):
        build_snapshots(_gifts(), kind="churn")
    with pytest.raises(ValueError, match="min_years_given"):
        build_snapshots(_gifts(), kind="lapse", min_years_given=0)
    assert build_snapshots(_gifts().iloc[:0], kind="lapse").empty


# --------------------------------------------------------------------------- #
# score_period: unlabelled rows for the year being scored
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["upgrade", "lapse", "response_next_year", "next_amount"])
def test_score_period_rows_match_the_labelled_rows(kind):
    from philanthropy.datasets import make_donor_panel

    gifts = make_donor_panel(n_donors=400, n_years=5, random_state=0)["gifts"]
    labelled = build_snapshots(gifts, kind=kind, fiscal_years=[2020], cadence=True)
    scored = build_snapshots(gifts, kind=kind, score_period=2020, cadence=True)
    assert "target" not in scored.columns and len(labelled) > 0
    if kind == "next_amount":
        # Scoring cannot condition on giving next year: everyone who gave in T.
        assert set(labelled.index) < set(scored.index)
        scored = scored.loc[labelled.index]
    pd.testing.assert_frame_equal(scored, labelled.drop(columns="target"))


def test_score_period_ignores_later_gifts_and_scores_the_last_year():
    from philanthropy.datasets import make_donor_panel

    gifts = make_donor_panel(n_donors=300, n_years=5, random_state=1)["gifts"]
    last = int(build_snapshots(gifts, kind="lapse")["period"].max()) + 1
    current = build_snapshots(gifts, kind="lapse", score_period=last)
    assert len(current) > 0 and set(current["period"]) == {last}
    trimmed = build_snapshots(gifts, kind="lapse", score_period=last - 1)
    early = gifts[gifts["gift_date"] <= gifts["gift_date"].max() - pd.DateOffset(years=1)]
    pd.testing.assert_frame_equal(trimmed, build_snapshots(early, kind="lapse", score_period=last - 1))


def test_score_period_rejects_a_future_year_and_fiscal_years():
    gifts = pd.DataFrame({"donor_id": ["a"], "gift_date": ["2020-08-01"], "gift_amount": [10.0]})
    with pytest.raises(ValueError, match="after the last fiscal year"):
        build_snapshots(gifts, kind="lapse", score_period=2030)
    with pytest.raises(ValueError, match="not both"):
        build_snapshots(gifts, kind="lapse", score_period=2021, fiscal_years=[2021])
