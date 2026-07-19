"""
tests/test_ingest.py
Tests for the philanthropy.ingest UniSchema ConstituentEvent bridge.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from philanthropy.ingest import (
    constituent_events_to_features,
    read_constituent_events,
)
from philanthropy.ingest._constituent_events import _FEATURE_DTYPES


def _event(email, etype, created, *, amount=None, source="GIVECAMPUS",
           external=None, event_id=None, first=None, last=None):
    ev = {
        "constituentEmail": email,
        "eventType": etype,
        "sourceSystem": source,
        "createdAt": created,
    }
    if amount is not None:
        ev["amount"] = amount
    if external is not None:
        ev["externalConstituentId"] = external
    if event_id is not None:
        ev["eventId"] = event_id
    if first is not None:
        ev["firstName"] = first
    if last is not None:
        ev["lastName"] = last
    return ev


@pytest.fixture
def events():
    return [
        _event("ada@uni.edu", "DONATION", "2025-01-15T10:00:00Z", amount=250.0),
        _event("ada@uni.edu", "DONATION", "2025-06-01T10:00:00Z", amount=500.0),
        _event("ada@uni.edu", "EVENT_REGISTRATION", "2025-06-10T09:00:00Z"),
        _event("ada@uni.edu", "EMAIL_CLICK", "2025-06-20T09:00:00Z"),
        _event("bob@uni.edu", "EVENT_REGISTRATION", "2025-03-01T09:00:00Z"),
    ]


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def test_returns_one_row_per_constituent(events):
    feats = constituent_events_to_features(events)
    assert set(feats.index) == {"ada@uni.edu", "bob@uni.edu"}
    assert feats.index.name == "constituent_id"


def test_monetary_and_counts(events):
    feats = constituent_events_to_features(events)
    ada = feats.loc["ada@uni.edu"]
    assert ada["total_gift_amount"] == 750.0
    assert ada["gift_count"] == 2
    assert ada["event_attendance_count"] == 1
    assert ada["email_click_count"] == 1

    bob = feats.loc["bob@uni.edu"]
    assert bob["total_gift_amount"] == 0.0
    assert bob["gift_count"] == 0
    assert bob["event_attendance_count"] == 1


def test_gift_dates(events):
    feats = constituent_events_to_features(events)
    ada = feats.loc["ada@uni.edu"]
    assert ada["first_gift_date"] == pd.Timestamp("2025-01-15T10:00:00")
    assert ada["last_gift_date"] == pd.Timestamp("2025-06-01T10:00:00")
    # A donor with no donations has NaT gift dates.
    assert pd.isna(feats.loc["bob@uni.edu"]["first_gift_date"])


def test_schema_and_dtypes(events):
    feats = constituent_events_to_features(events)
    assert list(feats.columns) == list(_FEATURE_DTYPES)
    for col, dt in _FEATURE_DTYPES.items():
        assert feats[col].dtype == np.dtype(dt), col


def test_recency_features_non_negative(events):
    feats = constituent_events_to_features(events)
    assert (feats["years_active"] >= 0).all()
    assert (feats["recency_days"] >= 0).all()


def test_distinct_source_systems():
    events = [
        _event("a@x.edu", "DONATION", "2025-01-01T00:00:00Z", amount=10, source="NPSP"),
        _event("a@x.edu", "EMAIL_CLICK", "2025-02-01T00:00:00Z", source="SLATE"),
        _event("a@x.edu", "EMAIL_CLICK", "2025-03-01T00:00:00Z", source="SLATE"),
    ]
    feats = constituent_events_to_features(events)
    assert feats.loc["a@x.edu", "distinct_source_systems"] == 2


# --------------------------------------------------------------------------- #
# Identity resolution
# --------------------------------------------------------------------------- #
def test_groups_by_external_id_when_present():
    events = [
        _event("old@x.edu", "DONATION", "2025-01-01T00:00:00Z", amount=100, external="CRM-1"),
        _event("new@x.edu", "DONATION", "2025-02-01T00:00:00Z", amount=200, external="CRM-1"),
    ]
    feats = constituent_events_to_features(events)
    assert list(feats.index) == ["CRM-1"]
    assert feats.loc["CRM-1", "total_gift_amount"] == 300.0


def test_falls_back_to_email_when_external_id_blank():
    events = [
        _event("a@x.edu", "DONATION", "2025-01-01T00:00:00Z", amount=100, external=""),
        _event("a@x.edu", "DONATION", "2025-02-01T00:00:00Z", amount=100, external=None),
    ]
    feats = constituent_events_to_features(events)
    assert list(feats.index) == ["a@x.edu"]
    assert feats.loc["a@x.edu", "total_gift_amount"] == 200.0


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #
def test_deduplicates_by_event_id():
    dup = _event("a@x.edu", "DONATION", "2025-01-01T00:00:00Z", amount=500, event_id="e1")
    feats = constituent_events_to_features([dup, dict(dup)])
    assert feats.loc["a@x.edu", "total_gift_amount"] == 500.0
    assert feats.loc["a@x.edu", "gift_count"] == 1


def test_dedup_preserves_events_without_event_id():
    # Mixed: one event carries an eventId, the rest don't. None must be dropped
    # as a spurious "duplicate" of another id-less event.
    events = [
        _event("a@x.edu", "DONATION", "2025-01-01T00:00:00Z", amount=100, event_id="e1"),
        _event("a@x.edu", "DONATION", "2025-02-01T00:00:00Z", amount=200),
        _event("a@x.edu", "DONATION", "2025-03-01T00:00:00Z", amount=300),
        _event("b@x.edu", "DONATION", "2025-01-01T00:00:00Z", amount=800),
    ]
    feats = constituent_events_to_features(events)
    assert feats.loc["a@x.edu", "total_gift_amount"] == 600.0
    assert feats.loc["a@x.edu", "gift_count"] == 3
    # b@x.edu (id-less) must not vanish as a duplicate of a's id-less donation.
    assert "b@x.edu" in feats.index
    assert feats.loc["b@x.edu", "total_gift_amount"] == 800.0


def test_missing_event_type_does_not_crash():
    events = [
        {"constituentEmail": "a@x.edu", "sourceSystem": "NPSP",
         "createdAt": "2025-01-01T00:00:00Z", "amount": 10.0},  # no eventType
        _event("a@x.edu", "DONATION", "2025-02-01T00:00:00Z", amount=50),
    ]
    feats = constituent_events_to_features(events)
    # The typed donation counts; the untyped event contributes to no typed count.
    assert feats.loc["a@x.edu", "gift_count"] == 1
    assert feats.loc["a@x.edu", "total_gift_amount"] == 50.0
    assert feats.loc["a@x.edu", "event_attendance_count"] == 0


def test_reference_date_nat_falls_back_to_batch_max():
    events = [_event("a@x.edu", "DONATION", "2025-01-01T00:00:00Z", amount=10)]
    feats = constituent_events_to_features(events, reference_date=pd.NaT)
    assert feats.loc["a@x.edu", "recency_days"] == 0


def test_deduplicate_can_be_disabled():
    dup = _event("a@x.edu", "DONATION", "2025-01-01T00:00:00Z", amount=500, event_id="e1")
    feats = constituent_events_to_features([dup, dict(dup)], deduplicate=False)
    assert feats.loc["a@x.edu", "total_gift_amount"] == 1000.0


def test_donation_missing_amount_counts_as_zero():
    events = [_event("a@x.edu", "DONATION", "2025-01-01T00:00:00Z")]  # no amount
    feats = constituent_events_to_features(events)
    assert feats.loc["a@x.edu", "total_gift_amount"] == 0.0
    assert feats.loc["a@x.edu", "gift_count"] == 1


def test_empty_input_returns_typed_empty_frame():
    feats = constituent_events_to_features([])
    assert feats.empty
    assert list(feats.columns) == list(_FEATURE_DTYPES)
    assert feats.index.name == "constituent_id"


def test_unparseable_timestamp_row_dropped():
    events = [
        _event("a@x.edu", "DONATION", "not-a-date", amount=999),
        _event("a@x.edu", "DONATION", "2025-01-01T00:00:00Z", amount=100),
    ]
    feats = constituent_events_to_features(events)
    assert feats.loc["a@x.edu", "total_gift_amount"] == 100.0


def test_reference_date_shifts_recency():
    events = [_event("a@x.edu", "DONATION", "2025-01-01T00:00:00Z", amount=100)]
    early = constituent_events_to_features(events, reference_date="2025-01-01")
    later = constituent_events_to_features(events, reference_date="2025-01-31")
    assert early.loc["a@x.edu", "recency_days"] == 0
    assert later.loc["a@x.edu", "recency_days"] == 30


def test_mixed_timezone_offsets_normalized_to_utc():
    # Same instant expressed two ways: +05:00 and the equivalent UTC Z.
    events = [
        _event("a@x.edu", "DONATION", "2025-01-01T05:00:00+05:00", amount=10),
        _event("a@x.edu", "EMAIL_CLICK", "2025-01-01T00:00:00Z"),
    ]
    feats = constituent_events_to_features(events, reference_date="2025-01-01T00:00:00Z")
    # Both events land at 2025-01-01T00:00 UTC → recency 0, gift date at UTC midnight.
    assert feats.loc["a@x.edu", "recency_days"] == 0
    assert feats.loc["a@x.edu", "first_gift_date"] == pd.Timestamp("2025-01-01T00:00:00")


def test_parsing_emits_no_warning():
    import warnings

    events = [_event("a@x.edu", "DONATION", "2025-01-01T00:00:00Z", amount=10)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        constituent_events_to_features(events)


def test_carries_first_last_name_when_present():
    events = [
        _event("ada@uni.edu", "DONATION", "2025-01-01T00:00:00Z", amount=100,
               first="Ada", last="Lovelace"),
        # A later event without names must not overwrite the resolved name.
        _event("ada@uni.edu", "EVENT_REGISTRATION", "2025-02-01T00:00:00Z"),
    ]
    feats = constituent_events_to_features(events)
    assert feats.loc["ada@uni.edu", "first_name"] == "Ada"
    assert feats.loc["ada@uni.edu", "last_name"] == "Lovelace"


def test_names_absent_yields_null_column_not_crash():
    events = [_event("a@x.edu", "DONATION", "2025-01-01T00:00:00Z", amount=10)]
    feats = constituent_events_to_features(events)
    assert "first_name" in feats.columns and "last_name" in feats.columns
    assert pd.isna(feats.loc["a@x.edu", "first_name"])
    assert pd.isna(feats.loc["a@x.edu", "last_name"])


def test_mixed_currency_warns():
    events = [
        _event("a@x.edu", "DONATION", "2025-01-01T00:00:00Z", amount=100),
        _event("a@x.edu", "DONATION", "2025-02-01T00:00:00Z", amount=200),
    ]
    events[0]["currency"] = "USD"
    events[1]["currency"] = "EUR"
    with pytest.warns(UserWarning, match="mixes currencies"):
        constituent_events_to_features(events)


def test_single_currency_does_not_warn():
    events = [
        _event("a@x.edu", "DONATION", "2025-01-01T00:00:00Z", amount=100),
        _event("a@x.edu", "DONATION", "2025-02-01T00:00:00Z", amount=200),
    ]
    for ev in events:
        ev["currency"] = "USD"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        constituent_events_to_features(events)


def test_accepts_dataframe_input(events):
    from_list = constituent_events_to_features(events)
    from_df = constituent_events_to_features(pd.DataFrame(events))
    pd.testing.assert_frame_equal(from_list, from_df)


# --------------------------------------------------------------------------- #
# read_constituent_events
# --------------------------------------------------------------------------- #
def test_read_single_json_object(tmp_path):
    ev = _event("a@x.edu", "DONATION", "2025-01-01T00:00:00Z", amount=10)
    f = tmp_path / "event.json"
    f.write_text(json.dumps(ev))
    assert read_constituent_events(f) == [ev]


def test_read_json_array(tmp_path):
    evs = [
        _event("a@x.edu", "DONATION", "2025-01-01T00:00:00Z", amount=10),
        _event("b@x.edu", "DONATION", "2025-01-02T00:00:00Z", amount=20),
    ]
    f = tmp_path / "events.json"
    f.write_text(json.dumps(evs))
    assert read_constituent_events(f) == evs


def test_read_ndjson(tmp_path):
    evs = [
        _event("a@x.edu", "DONATION", "2025-01-01T00:00:00Z", amount=10),
        _event("b@x.edu", "DONATION", "2025-01-02T00:00:00Z", amount=20),
    ]
    f = tmp_path / "batch.ndjson"
    f.write_text("\n".join(json.dumps(e) for e in evs) + "\n")
    assert read_constituent_events(f) == evs


def test_read_directory_concatenates(tmp_path):
    (tmp_path / "1.json").write_text(json.dumps(
        _event("a@x.edu", "DONATION", "2025-01-01T00:00:00Z", amount=10)))
    (tmp_path / "2.ndjson").write_text(json.dumps(
        _event("b@x.edu", "DONATION", "2025-01-02T00:00:00Z", amount=20)) + "\n")
    events = read_constituent_events(tmp_path)
    assert len(events) == 2
    feats = constituent_events_to_features(events)
    assert set(feats.index) == {"a@x.edu", "b@x.edu"}


def test_read_empty_file(tmp_path):
    f = tmp_path / "empty.ndjson"
    f.write_text("")
    assert read_constituent_events(f) == []


def _write_partitioned(root, vendor, day, event_id, ev):
    # Mirror UniSchema's {prefix}/{vendor}/{yyyy}/{mm}/{dd}/{eventId}.json layout.
    d = root / vendor / "2026" / "07" / day
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{event_id}.json").write_text(json.dumps(ev))


def test_read_recurses_into_date_partitioned_egress(tmp_path):
    # UniSchema egress: events live several levels down, none in the top dir.
    _write_partitioned(tmp_path, "givecampus", "19", "e1",
                       _event("a@x.edu", "DONATION", "2026-07-19T00:00:00Z", amount=10))
    _write_partitioned(tmp_path, "npsp", "20", "e2",
                       _event("b@x.edu", "DONATION", "2026-07-20T00:00:00Z", amount=20))
    events = read_constituent_events(tmp_path)
    assert len(events) == 2
    feats = constituent_events_to_features(events)
    assert set(feats.index) == {"a@x.edu", "b@x.edu"}


def test_read_skips_manifest_sidecars(tmp_path):
    _write_partitioned(tmp_path, "npsp", "19", "e1",
                       _event("a@x.edu", "DONATION", "2026-07-19T00:00:00Z", amount=10))
    # A batch .manifest.json sidecar must be ignored (metadata, not an event).
    (tmp_path / "npsp" / "2026" / "07" / "batch.manifest.json").write_text(
        json.dumps({"batchId": "b1", "count": 1}))
    events = read_constituent_events(tmp_path)
    assert events == [_event("a@x.edu", "DONATION", "2026-07-19T00:00:00Z", amount=10)]


def test_read_mixes_nested_json_and_ndjson(tmp_path):
    _write_partitioned(tmp_path, "givecampus", "19", "e1",
                       _event("a@x.edu", "DONATION", "2026-07-19T00:00:00Z", amount=10))
    nd = tmp_path / "slate" / "2026" / "07" / "20"
    nd.mkdir(parents=True, exist_ok=True)
    (nd / "batch.ndjson").write_text("\n".join(json.dumps(e) for e in [
        _event("b@x.edu", "DONATION", "2026-07-20T00:00:00Z", amount=20),
        _event("c@x.edu", "EMAIL_CLICK", "2026-07-20T01:00:00Z"),
    ]) + "\n")
    events = read_constituent_events(tmp_path)
    assert len(events) == 3
    assert {e["constituentEmail"] for e in events} == {"a@x.edu", "b@x.edu", "c@x.edu"}


def test_read_directory_ordering_is_deterministic(tmp_path):
    for vendor, day, eid, email in [
        ("slate", "21", "e3", "c@x.edu"),
        ("givecampus", "19", "e1", "a@x.edu"),
        ("npsp", "20", "e2", "b@x.edu"),
    ]:
        _write_partitioned(tmp_path, vendor, day, eid,
                           _event(email, "DONATION", "2026-07-19T00:00:00Z", amount=1))
    order = [e["constituentEmail"] for e in read_constituent_events(tmp_path)]
    # Sorted by relative path: givecampus < npsp < slate.
    assert order == ["a@x.edu", "b@x.edu", "c@x.edu"]
    assert order == [e["constituentEmail"] for e in read_constituent_events(tmp_path)]


# --------------------------------------------------------------------------- #
# Integration: bridge output flows into an estimator
# --------------------------------------------------------------------------- #
def test_features_feed_donor_propensity_model():
    from philanthropy.models import DonorPropensityModel

    rng = np.random.default_rng(0)
    events = []
    for i in range(60):
        email = f"donor{i}@uni.edu"
        n_gifts = int(rng.integers(0, 5))
        for g in range(n_gifts):
            events.append(_event(email, "DONATION",
                                 f"2025-0{1 + g}-01T00:00:00Z",
                                 amount=float(rng.integers(50, 5000))))
        for _ in range(int(rng.integers(0, 4))):
            events.append(_event(email, "EVENT_REGISTRATION", "2025-05-01T00:00:00Z"))

    feats = constituent_events_to_features(events)
    X = feats[["total_gift_amount", "years_active", "event_attendance_count"]].to_numpy()
    y = (feats["total_gift_amount"] > feats["total_gift_amount"].median()).astype(int).to_numpy()

    model = DonorPropensityModel(n_estimators=20, random_state=0)
    model.fit(X, y)
    scores = model.predict_affinity_score(X)
    assert scores.shape == (len(feats),)
    assert ((scores >= 0) & (scores <= 100)).all()
