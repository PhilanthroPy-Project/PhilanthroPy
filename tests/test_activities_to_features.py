"""
tests/test_activities_to_features.py
Tests for philanthropy.ingest.activities_to_features.

The point of the module is the same aggregation
philanthropy.ingest._constituent_events.constituent_events_to_features does,
generalised to an open-ended set of activity types: every type present in
the (cutoff) data gets its own count_12m / count_36m / days_since_last /
distinct block, a type with zero rows anywhere gets none of those columns at
all, and a donor with zero rows of a present type gets zeros in count_12m /
count_36m / distinct but NaN (not 0) in days_since_last, since 0 there would
read as "did it today" instead of "never". as_of is the leakage boundary
most tests below are built around.
"""

import warnings

import pandas as pd
import pytest

from philanthropy.ingest import activities_to_features


def _row(contact_id, date, activity_type, **extra):
    row = {"contact_id": contact_id, "activity_date": date, "activity_type": activity_type}
    row.update(extra)
    return row


def test_counts_within_and_outside_windows():
    rows = [
        _row("1", "2024-12-01", "event"),  # within 12m
        _row("1", "2023-06-01", "event"),  # within 36m, outside 12m
        _row("1", "2020-01-01", "event"),  # outside 36m
    ]
    feats = activities_to_features(rows, as_of="2024-12-31")

    assert feats.loc["1", "event_count_12m"] == 1
    assert feats.loc["1", "event_count_36m"] == 2
    assert feats.loc["1", "event_distinct"] == 3


def test_days_since_last_measured_from_as_of():
    rows = [_row("1", "2024-01-01", "event")]
    feats = activities_to_features(rows, as_of="2024-01-11")

    assert feats.loc["1", "event_days_since_last"] == 10


def test_donor_with_no_rows_of_a_type_gets_zero():
    rows = [
        _row("1", "2024-01-01", "event"),
        _row("2", "2024-01-01", "volunteer"),
    ]
    feats = activities_to_features(rows, as_of="2024-12-31")

    assert feats.loc["2", "event_count_12m"] == 0
    assert feats.loc["2", "event_count_36m"] == 0
    assert pd.isna(feats.loc["2", "event_days_since_last"])
    assert feats.loc["2", "event_distinct"] == 0


def test_donor_with_no_rows_of_a_type_is_nan_not_zero_in_days_since_last():
    # A donor who never did an activity of a given type has no "last time"
    # to measure days_since_last from; 0 would misread as "did it today".
    rows = [
        _row("1", "2024-01-01", "event"),
        _row("2", "2024-01-01", "volunteer"),
    ]
    feats = activities_to_features(rows, as_of="2024-12-31")

    assert pd.isna(feats.loc["2", "event_days_since_last"])
    assert feats["event_days_since_last"].dtype == "float64"


def test_type_absent_from_whole_input_produces_no_columns():
    rows = [_row("1", "2024-01-01", "event")]
    feats = activities_to_features(rows, as_of="2024-12-31")

    assert not any(col.startswith("volunteer_") for col in feats.columns)


def test_amount_column_present_yields_amount_12m():
    rows = [
        _row("1", "2024-06-01", "gift", amount=100),
        _row("1", "2020-01-01", "gift", amount=999),  # outside 12m window
    ]
    feats = activities_to_features(rows, as_of="2024-12-31")

    assert feats.loc["1", "gift_amount_12m"] == 100.0


def test_hours_column_present_yields_hours_12m():
    rows = [_row("1", "2024-06-01", "volunteer", hours=3.5)]
    feats = activities_to_features(rows, as_of="2024-12-31")

    assert feats.loc["1", "volunteer_hours_12m"] == 3.5


def test_no_amount_or_hours_column_yields_no_such_columns():
    rows = [_row("1", "2024-06-01", "event")]
    feats = activities_to_features(rows, as_of="2024-12-31")

    assert "event_amount_12m" not in feats.columns
    assert "event_hours_12m" not in feats.columns


def test_multiple_activity_types_each_get_their_own_columns():
    rows = [
        _row("1", "2024-01-01", "event"),
        _row("1", "2024-02-01", "volunteer"),
    ]
    feats = activities_to_features(rows, as_of="2024-12-31")

    for prefix in ("event", "volunteer"):
        for suffix in ("count_12m", "count_36m", "days_since_last", "distinct"):
            assert f"{prefix}_{suffix}" in feats.columns


def test_tz_aware_as_of_is_accepted():
    # A tz-aware as_of (e.g. read back from a CRM export with a UTC offset)
    # used to raise TypeError comparing tz-aware to tz-naive activity dates.
    rows = [_row("1", "2024-01-01", "event")]

    feats = activities_to_features(rows, as_of=pd.Timestamp("2024-12-31", tz="UTC"))

    assert feats.loc["1", "event_count_36m"] == 1


def test_tz_aware_as_of_agrees_with_naive_equivalent():
    rows = [_row("1", "2024-01-01", "event")]

    naive = activities_to_features(rows, as_of="2024-12-31")
    tz_aware = activities_to_features(rows, as_of=pd.Timestamp("2024-12-31", tz="UTC"))

    pd.testing.assert_frame_equal(naive, tz_aware)


def test_float_contact_id_normalises_to_integer_string():
    # pandas reads a numeric id column as float64 the moment any cell in it
    # is blank; contact_id 123 then arrives as 123.0 and used to be stringified
    # as "123.0", which can never match another donor's "123".
    df = pd.DataFrame(
        {
            "contact_id": pd.Series([123.0, None, 456.0]),
            "activity_date": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "activity_type": ["event", "event", "event"],
        }
    )

    feats = activities_to_features(df, as_of="2024-12-31")

    assert list(feats.index) == ["123", "456"]


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #
def test_window_boundary_day_is_excluded():
    # Documented behaviour: a row exactly 12 (or 36) months before as_of is
    # outside that window, not inside it (strict > against the cutoff).
    rows = [_row("1", "2023-12-31", "event")]  # exactly 12 months before as_of

    feats = activities_to_features(rows, as_of="2024-12-31")

    assert feats.loc["1", "event_count_12m"] == 0
    assert feats.loc["1", "event_count_36m"] == 1


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #
def test_rows_after_as_of_are_ignored():
    rows = [_row("1", "2024-06-01", "event")]
    before = activities_to_features(rows, as_of="2024-12-31")

    rows_with_future = rows + [_row("1", "2025-06-01", "event")]
    after = activities_to_features(rows_with_future, as_of="2024-12-31")

    pd.testing.assert_frame_equal(before, after)


def test_future_row_of_a_new_type_does_not_add_columns():
    rows = [_row("1", "2024-06-01", "event")]
    feats = activities_to_features(
        rows + [_row("1", "2025-06-01", "volunteer")], as_of="2024-12-31"
    )

    assert not any(col.startswith("volunteer_") for col in feats.columns)


def test_idempotent_on_repeat():
    rows = [
        _row("1", "2024-06-01", "event"),
        _row("2", "2023-01-01", "volunteer", hours=2),
    ]
    first = activities_to_features(rows, as_of="2024-12-31")
    second = activities_to_features(rows, as_of="2024-12-31")

    pd.testing.assert_frame_equal(first, second)


# --------------------------------------------------------------------------- #
# Match rate against donors
# --------------------------------------------------------------------------- #
def test_low_match_rate_against_donors_warns():
    rows = [_row(str(i), "2024-01-01", "event") for i in range(10)]
    donors = pd.DataFrame(index=pd.Index([str(i) for i in range(2)], name="contact_id"))

    with pytest.warns(UserWarning, match="match"):
        activities_to_features(rows, as_of="2024-12-31", donors=donors)


def test_high_match_rate_against_donors_does_not_warn():
    rows = [_row(str(i), "2024-01-01", "event") for i in range(10)]
    donors = pd.DataFrame(index=pd.Index([str(i) for i in range(10)], name="contact_id"))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        activities_to_features(rows, as_of="2024-12-31", donors=donors)


def test_donors_as_plain_iterable_of_ids():
    rows = [_row(str(i), "2024-01-01", "event") for i in range(10)]
    donors = [str(i) for i in range(2)]

    with pytest.warns(UserWarning, match="match"):
        activities_to_features(rows, as_of="2024-12-31", donors=donors)


def test_donors_as_series_of_ids():
    rows = [_row(str(i), "2024-01-01", "event") for i in range(10)]
    donors = pd.Series([str(i) for i in range(2)])

    with pytest.warns(UserWarning, match="match"):
        activities_to_features(rows, as_of="2024-12-31", donors=donors)


def test_donors_does_not_add_rows_for_donors_absent_from_activities():
    rows = [_row("1", "2024-01-01", "event")]
    donors = pd.DataFrame(index=pd.Index(["1", "2", "3"], name="contact_id"))

    feats = activities_to_features(rows, as_of="2024-12-31", donors=donors)

    assert list(feats.index) == ["1"]


# --------------------------------------------------------------------------- #
# Validation and empty input
# --------------------------------------------------------------------------- #
def test_missing_required_column_raises():
    rows = [{"contact_id": "1", "activity_date": "2024-01-01"}]

    with pytest.raises(KeyError, match="activity_type"):
        activities_to_features(rows, as_of="2024-12-31")


def test_empty_input_returns_empty_frame_with_no_columns():
    feats = activities_to_features([], as_of="2024-12-31")

    assert feats.empty
    assert feats.index.name == "contact_id"


def test_every_row_filtered_out_by_as_of_returns_empty_frame():
    rows = [_row("1", "2030-01-01", "event")]

    feats = activities_to_features(rows, as_of="2024-12-31")

    assert feats.empty


def test_blank_activity_type_is_dropped():
    rows = [
        _row("1", "2024-01-01", "event"),
        _row("1", "2024-01-02", ""),
    ]
    feats = activities_to_features(rows, as_of="2024-12-31")

    assert list(feats.columns) == [
        "event_count_12m",
        "event_count_36m",
        "event_days_since_last",
        "event_distinct",
    ]


def test_unparseable_date_row_is_dropped():
    rows = [
        _row("1", "2024-01-01", "event"),
        _row("1", "not-a-date", "event"),
    ]
    feats = activities_to_features(rows, as_of="2024-12-31")

    assert feats.loc["1", "event_distinct"] == 1


def test_dataframe_input_accepted():
    df = pd.DataFrame(
        [_row("1", "2024-01-01", "event"), _row("2", "2024-02-01", "event")]
    )
    feats = activities_to_features(df, as_of="2024-12-31")

    assert list(feats.index) == ["1", "2"]


def test_output_sorted_by_contact_id():
    rows = [_row("9", "2024-01-01", "event"), _row("1", "2024-01-01", "event")]
    feats = activities_to_features(rows, as_of="2024-12-31")

    assert list(feats.index) == ["1", "9"]
