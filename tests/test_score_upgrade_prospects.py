"""
tests/test_score_upgrade_prospects.py
Tests for philanthropy.models.score_upgrade_prospects.

Fixture: four donor archetypes across five fiscal years (FY2021-FY2025).
"early_riser" crosses `threshold` at FY2022, "late_riser" at FY2024;
"flat_high" and "flat_low" never cross it and are the only two still
band-qualifying "today" (FY2025, the fiscal year containing the default
`as_of`).
"""

import numpy as np
import pandas as pd
import pytest

from philanthropy.models import score_upgrade_prospects

_YEARS = ["2020-08-01", "2021-08-01", "2022-08-01", "2023-08-01", "2024-08-01"]
_ARCHETYPES = {
    "flat_high": [900, 900, 900, 900, 900],
    "early_riser": [600, 1050, 1200, 1400, 1600],
    "late_riser": [300, 500, 700, 1100, 1300],
    "flat_low": [400, 400, 400, 400, 400],
}


def _archetype_gifts(n_per_group=20):
    rows = [
        {"donor_id": f"{name}_{i}", "gift_date": year, "gift_amount": amount}
        for name, amounts in _ARCHETYPES.items()
        for i in range(n_per_group)
        for year, amount in zip(_YEARS, amounts)
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Shape and basic behaviour
# --------------------------------------------------------------------------- #
def test_basic_output_shape_and_columns():
    scores, report = score_upgrade_prospects(_archetype_gifts(), random_state=0)
    assert list(scores.columns) == [
        "fiscal_year", "affinity_score", "rank", "decile", "top_reasons", "suggested_ask",
    ]
    assert scores.index.name == "donor_id"
    assert len(scores) == 40  # flat_high + flat_low, 20 each


def test_only_currently_band_qualifying_donors_are_scored():
    scores, _ = score_upgrade_prospects(_archetype_gifts(), random_state=0)
    names = {donor_id.rsplit("_", 1)[0] for donor_id in scores.index}
    assert names == {"flat_high", "flat_low"}


def test_affinity_score_bounded_0_to_100():
    scores, _ = score_upgrade_prospects(_archetype_gifts(), random_state=0)
    assert (scores["affinity_score"] >= 0).all()
    assert (scores["affinity_score"] <= 100).all()


def test_sorted_descending_with_matching_rank_and_decile():
    scores, _ = score_upgrade_prospects(_archetype_gifts(), random_state=0)
    assert list(scores["affinity_score"]) == sorted(scores["affinity_score"], reverse=True)
    assert list(scores["rank"]) == list(range(1, len(scores) + 1))
    assert scores["decile"].min() >= 1
    assert scores["decile"].max() <= 10


def test_top_reasons_are_feature_value_pairs():
    scores, _ = score_upgrade_prospects(_archetype_gifts(), random_state=0)
    reasons = scores["top_reasons"].iloc[0]
    assert len(reasons) <= 3
    for feature, value in reasons:
        assert isinstance(feature, str)
        assert np.isscalar(value)


def test_suggested_ask_is_nan_not_forced():
    # No ask-amount label exists in this data; see the docstring's "Suggested
    # ask" note for why this is left NaN rather than a fabricated number.
    scores, _ = score_upgrade_prospects(_archetype_gifts(), random_state=0)
    assert scores["suggested_ask"].isna().all()


def test_report_keys_present():
    _, report = score_upgrade_prospects(_archetype_gifts(), random_state=0)
    for key in (
        "n_training_rows", "n_training_fiscal_years", "low_data_warning",
        "low_data_message", "activity_id_match_warnings", "current_fiscal_year",
        "n_scored", "validated", "validation_fiscal_year", "n_validation_rows",
        "top_n", "model_upgrade_rate_top_n",
        "baseline_topn_fy_total_upgrade_rate", "lift_topn_fy_total",
        "baseline_giving_threshold", "baseline_gave_threshold_upgrade_rate",
        "lift_over_gave_threshold", "overall_upgrade_rate", "deciles",
        "roc_auc", "average_precision",
    ):
        assert key in report


# --------------------------------------------------------------------------- #
# Validation and errors
# --------------------------------------------------------------------------- #
def test_band_low_above_high_raises():
    with pytest.raises(ValueError):
        score_upgrade_prospects(_archetype_gifts(), band=(999, 100))


def test_no_usable_gift_rows_raises():
    gifts = pd.DataFrame({
        "donor_id": ["1"], "gift_date": ["not-a-date"], "gift_amount": [500],
    })
    with pytest.raises(ValueError, match="No usable gift rows"):
        score_upgrade_prospects(gifts)


def test_as_of_before_every_gift_raises():
    gifts = _archetype_gifts()
    with pytest.raises(ValueError, match="No gift rows on or before as_of"):
        score_upgrade_prospects(gifts, as_of="2000-01-01")


def test_no_historical_rows_raises():
    # A single fiscal year: no T with both T and T+1 resolved.
    gifts = pd.DataFrame({
        "donor_id": ["1"], "gift_date": ["2024-08-01"], "gift_amount": [500],
    })
    with pytest.raises(ValueError, match="No historical"):
        score_upgrade_prospects(gifts)


def test_low_data_warning_flagged_under_500_rows():
    _, report = score_upgrade_prospects(_archetype_gifts(), random_state=0)
    assert report["n_training_rows"] < 500
    assert report["low_data_warning"] is True
    assert report["low_data_message"] is not None
    with pytest.warns(UserWarning, match="historical donor-year rows"):
        score_upgrade_prospects(_archetype_gifts(), random_state=0)


def test_no_donors_currently_in_band_returns_empty_scores():
    # h0..h5: FY2020 (band) -> FY2021 (still below threshold, target=0 for
    # the one historical training row at T=FY2020) -> FY2022 (the current
    # year: already above threshold, so excluded from the current band).
    # "grad" only ever gives in the current year, also above threshold.
    rows = [
        {"donor_id": f"h{i}", "gift_date": "2019-08-01", "gift_amount": 300}
        for i in range(6)
    ] + [
        {"donor_id": f"h{i}", "gift_date": "2020-08-01", "gift_amount": 500}
        for i in range(6)
    ] + [
        {"donor_id": f"h{i}", "gift_date": "2021-08-01", "gift_amount": 5000}
        for i in range(6)
    ] + [
        {"donor_id": "grad", "gift_date": "2021-08-01", "gift_amount": 5000},
    ]
    gifts = pd.DataFrame(rows)
    scores, report = score_upgrade_prospects(gifts)
    assert scores.empty
    assert list(scores.columns) == [
        "fiscal_year", "affinity_score", "rank", "decile", "top_reasons", "suggested_ask",
    ]
    assert report["n_scored"] == 0


# --------------------------------------------------------------------------- #
# activities integration: the match-rate warning is captured, not recomputed
# --------------------------------------------------------------------------- #
def test_activity_match_rate_warning_is_captured_in_the_report():
    gifts = _archetype_gifts()
    # activities_to_features only checks the match rate when `donors` is
    # given (a real donor population, distinct from the gift log itself).
    donor_ids = [f"{name}_{i}" for name in _ARCHETYPES for i in range(20)]
    donors = pd.DataFrame(index=pd.Index(donor_ids, name="donor_id"))
    # Every activity contact_id is a donor id that does not exist in
    # `donors`: a 0% match rate, well under the 80% cutoff.
    activities = [
        {"contact_id": f"nobody_{i}", "activity_date": "2024-08-01", "activity_type": "event"}
        for i in range(5)
    ]
    _, report = score_upgrade_prospects(
        gifts, activities=activities, donors=donors, random_state=0
    )
    assert len(report["activity_id_match_warnings"]) >= 1
    assert "match rate" in report["activity_id_match_warnings"][0] or \
        "distinct" in report["activity_id_match_warnings"][0]


def test_activity_match_rate_warning_does_not_escape_uncaptured():
    # The warning is swallowed into the report, not left to surface on the
    # caller's own warnings filter (recomputing match-rate logic is exactly
    # what this avoids).
    gifts = _archetype_gifts()
    donor_ids = [f"{name}_{i}" for name in _ARCHETYPES for i in range(20)]
    donors = pd.DataFrame(index=pd.Index(donor_ids, name="donor_id"))
    activities = [
        {"contact_id": "nobody", "activity_date": "2024-08-01", "activity_type": "event"}
    ]
    with pytest.warns(UserWarning, match="historical donor-year rows"):
        # Only the (unrelated) low-data warning should reach here.
        score_upgrade_prospects(gifts, activities=activities, donors=donors, random_state=0)


def test_activities_columns_influence_current_row_without_crashing():
    gifts = _archetype_gifts()
    activities = [
        {"contact_id": "flat_high_0", "activity_date": "2024-08-01", "activity_type": "event"},
    ]
    scores, _ = score_upgrade_prospects(gifts, activities=activities, random_state=0)
    assert "flat_high_0" in scores.index


# --------------------------------------------------------------------------- #
# donors integration: non-numeric columns are joined but excluded from X
# --------------------------------------------------------------------------- #
def test_donors_numeric_column_used_non_numeric_ignored():
    gifts = _archetype_gifts()
    donor_ids = [f"flat_high_{i}" for i in range(20)] + [f"flat_low_{i}" for i in range(20)]
    donors = pd.DataFrame(
        {
            "wealth_rating": ["A"] * 40,
            "capacity_estimate": list(range(40)),
        },
        index=pd.Index(donor_ids, name="donor_id"),
    )
    scores, _ = score_upgrade_prospects(gifts, donors=donors, random_state=0)
    assert len(scores) == 40


# --------------------------------------------------------------------------- #
# Leakage: as_of is the hard cutoff for both halves of the function
# --------------------------------------------------------------------------- #
def test_current_row_ignores_a_gift_dated_after_as_of_in_the_same_fiscal_year():
    gifts = _archetype_gifts()
    as_of = "2024-08-01"  # inside FY2025, the fiscal year every FY2025 gift lands in

    before, report_before = score_upgrade_prospects(gifts, as_of=as_of, random_state=0)

    # One day after as_of, still within FY2025: if this leaked into the
    # current row, flat_low_0's FY total would jump from 400 to 5400, well
    # past threshold, and it would vanish from the band entirely.
    future_row = pd.DataFrame([
        {"donor_id": "flat_low_0", "gift_date": "2024-08-02", "gift_amount": 5000}
    ])
    gifts_after = pd.concat([gifts, future_row], ignore_index=True)
    after, report_after = score_upgrade_prospects(gifts_after, as_of=as_of, random_state=0)

    assert "flat_low_0" in before.index
    assert "flat_low_0" in after.index
    pd.testing.assert_frame_equal(before.sort_index(), after.sort_index())
    assert report_before["n_training_rows"] == report_after["n_training_rows"]


def test_appending_a_far_future_gift_does_not_change_training_or_scores():
    gifts = _archetype_gifts()
    as_of = "2024-08-01"

    before, report_before = score_upgrade_prospects(gifts, as_of=as_of, random_state=0)

    future_row = pd.DataFrame([
        {"donor_id": "flat_low_0", "gift_date": "2030-08-01", "gift_amount": 100_000}
    ])
    gifts_after = pd.concat([gifts, future_row], ignore_index=True)
    after, report_after = score_upgrade_prospects(gifts_after, as_of=as_of, random_state=0)

    pd.testing.assert_frame_equal(before.sort_index(), after.sort_index())
    assert report_before["n_training_rows"] == report_after["n_training_rows"]


def test_as_of_defaults_to_latest_gift_date():
    gifts = _archetype_gifts()
    explicit, _ = score_upgrade_prospects(gifts, as_of="2024-08-01", random_state=0)
    default, _ = score_upgrade_prospects(gifts, random_state=0)
    pd.testing.assert_frame_equal(explicit.sort_index(), default.sort_index())


# --------------------------------------------------------------------------- #
# F5: no upgraders / all upgraders in history (previously an IndexError deep
# inside predict_proba)
# --------------------------------------------------------------------------- #
def test_single_class_history_raises_clear_value_error():
    # Every donor stays flat below the band ceiling for every year: target is
    # 0 for every historical row, so there is nothing to learn.
    years = ["2020-08-01", "2021-08-01", "2022-08-01", "2023-08-01", "2024-08-01"]
    rows = [
        {"donor_id": f"flat_low_{i}", "gift_date": year, "gift_amount": 400}
        for i in range(20)
        for year in years
    ]
    gifts = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="only one class"):
        score_upgrade_prospects(gifts, random_state=0)


# --------------------------------------------------------------------------- #
# F5: tiny training sets (previously an sklearn ValueError from deep inside
# CalibratedClassifierCV instead of a clear, documented one)
# --------------------------------------------------------------------------- #
def test_tiny_training_set_raises_clear_value_error():
    years = ["2020-08-01", "2021-08-01", "2022-08-01"]
    rows = [
        {"donor_id": "a", "gift_date": years[0], "gift_amount": 400},
        {"donor_id": "a", "gift_date": years[1], "gift_amount": 1200},
        {"donor_id": "a", "gift_date": years[2], "gift_amount": 400},
        {"donor_id": "b", "gift_date": years[0], "gift_amount": 400},
        {"donor_id": "b", "gift_date": years[1], "gift_amount": 400},
        {"donor_id": "b", "gift_date": years[2], "gift_amount": 400},
    ]
    gifts = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="5-fold cross-validation"):
        score_upgrade_prospects(gifts, random_state=0)


# --------------------------------------------------------------------------- #
# F5: fiscal_year is not donor-specific and sits outside the training range
# at scoring time, so it must not be a model feature (still an output column)
# --------------------------------------------------------------------------- #
def test_fiscal_year_excluded_from_top_reasons():
    scores, _ = score_upgrade_prospects(_archetype_gifts(), random_state=0)
    for reasons in scores["top_reasons"]:
        assert all(feature != "fiscal_year" for feature, _ in reasons)


# --------------------------------------------------------------------------- #
# F5: top_n defaults to ~10% of the validation fold, not a fixed count, and
# the report carries deciles, roc_auc, average_precision and two baselines
# --------------------------------------------------------------------------- #
def test_default_top_n_is_roughly_ten_percent_of_validation_fold():
    _, report = score_upgrade_prospects(_archetype_gifts(), random_state=0)
    n_val = report["n_validation_rows"]
    assert report["top_n"] == max(1, round(0.1 * n_val))


def test_top_n_parameter_overrides_default():
    _, report = score_upgrade_prospects(_archetype_gifts(), top_n=3, random_state=0)
    assert report["top_n"] == 3


def test_larger_validation_fold_does_not_use_a_fixed_top_n():
    # With more rows per archetype, ~10% of the fold should exceed the old
    # hardcoded top_n of 10, proving it now scales with fold size.
    _, report = score_upgrade_prospects(_archetype_gifts(n_per_group=200), random_state=0)
    assert report["top_n"] > 10


def test_deciles_report_shape_and_coverage():
    _, report = score_upgrade_prospects(_archetype_gifts(), random_state=0)
    deciles = report["deciles"]
    assert len(deciles) == 10
    assert [d["decile"] for d in deciles] == list(range(1, 11))
    assert sum(d["n"] for d in deciles) == report["n_validation_rows"]
    for d in deciles:
        if d["n"] > 0:
            assert 0.0 <= d["actual_rate"] <= 1.0
            assert 0.0 <= d["mean_predicted"] <= 1.0


def test_roc_auc_and_average_precision_bounded():
    _, report = score_upgrade_prospects(_archetype_gifts(), random_state=0)
    assert report["roc_auc"] is None or 0.0 <= report["roc_auc"] <= 1.0
    assert report["average_precision"] is None or 0.0 <= report["average_precision"] <= 1.0


def test_baseline_giving_threshold_defaults_to_half_threshold():
    _, report = score_upgrade_prospects(_archetype_gifts(), threshold=1000.0, random_state=0)
    assert report["baseline_giving_threshold"] == 500.0


def test_baseline_giving_threshold_parameter_overrides_default():
    _, report = score_upgrade_prospects(
        _archetype_gifts(), baseline_giving_threshold=300.0, random_state=0
    )
    assert report["baseline_giving_threshold"] == 300.0


def test_two_named_baselines_and_lifts_reported():
    _, report = score_upgrade_prospects(_archetype_gifts(), random_state=0)
    for rate_key in (
        "baseline_topn_fy_total_upgrade_rate", "baseline_gave_threshold_upgrade_rate",
    ):
        assert report[rate_key] is None or 0.0 <= report[rate_key] <= 1.0
    for lift_key in ("lift_topn_fy_total", "lift_over_gave_threshold"):
        assert report[lift_key] is None or report[lift_key] >= 0.0
