"""
tests/test_civicrm.py
Tests for the philanthropy.ingest CiviCRM contribution bridge.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from philanthropy.ingest import (
    civicrm_contributions_to_features,
    read_civicrm_contributions,
)
from philanthropy.ingest._civicrm import _FEATURE_DTYPES

# Header labels as a CiviCRM CSV export writes them.
_HEADER = (
    "Contribution ID,Contact ID,Contribution Date,Total Amount,Currency,"
    "Contribution Status,Financial Type,Test Mode,Email,First Name,Last Name"
)


def _row(cid, contact, date, amount, *, currency="USD", status="Completed",
         ftype="Donation", test="0", email="", first="", last=""):
    return (
        f"{cid},{contact},{date},{amount},{currency},{status},{ftype},"
        f"{test},{email},{first},{last}"
    )


def _export(*rows):
    return "\n".join((_HEADER,) + rows) + "\n"


@pytest.fixture
def contributions():
    """APIv4-spelled rows: two donors, one Failed gift, one test-mode gift."""
    return [
        {"id": "1", "contact_id": "101", "receive_date": "2025-01-15",
         "total_amount": "250.00", "currency": "USD",
         "contribution_status": "Completed", "financial_type": "Donation",
         "is_test": "0", "email": "ada@uni.edu",
         "first_name": "Ada", "last_name": "Lovelace"},
        {"id": "2", "contact_id": "101", "receive_date": "2025-06-01",
         "total_amount": "1000.00", "currency": "USD",
         "contribution_status": "Completed", "financial_type": "Member Dues",
         "is_test": "0", "email": "ada@uni.edu",
         "first_name": "Ada", "last_name": "Lovelace"},
        {"id": "3", "contact_id": "101", "receive_date": "2025-06-05",
         "total_amount": "99.00", "currency": "USD",
         "contribution_status": "Failed", "financial_type": "Donation",
         "is_test": "0", "email": "ada@uni.edu",
         "first_name": "Ada", "last_name": "Lovelace"},
        {"id": "4", "contact_id": "202", "receive_date": "2025-03-01",
         "total_amount": "5000.00", "currency": "USD",
         "contribution_status": "Completed", "financial_type": "Donation",
         "is_test": "1", "email": "grace@uni.edu",
         "first_name": "Grace", "last_name": "Hopper"},
        {"id": "5", "contact_id": "202", "receive_date": "2025-04-10",
         "total_amount": "750.00", "currency": "USD",
         "contribution_status": "Completed", "financial_type": "Donation",
         "is_test": "0", "email": "grace@uni.edu",
         "first_name": "Grace", "last_name": "Hopper"},
    ]


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def test_returns_one_row_per_contact(contributions):
    feats = civicrm_contributions_to_features(contributions)
    assert list(feats.index) == ["101", "202"]
    assert feats.index.name == "contact_id"


def test_schema_and_dtypes(contributions):
    feats = civicrm_contributions_to_features(contributions)
    assert list(feats.columns) == list(_FEATURE_DTYPES)
    for col, dtype in _FEATURE_DTYPES.items():
        assert feats[col].dtype == np.dtype(dtype), col


def test_monetary_and_counts(contributions):
    feats = civicrm_contributions_to_features(contributions)
    # 250 + 1000; the Failed row is excluded.
    assert float(feats.loc["101", "total_gift_amount"]) == 1250.0
    assert int(feats.loc["101", "gift_count"]) == 2
    assert float(feats.loc["101", "largest_gift_amount"]) == 1000.0
    # 202's 5000 gift is test-mode, so only the 750 survives.
    assert float(feats.loc["202", "total_gift_amount"]) == 750.0
    assert int(feats.loc["202", "gift_count"]) == 1


def test_gift_dates(contributions):
    feats = civicrm_contributions_to_features(contributions)
    assert feats.loc["101", "first_gift_date"] == pd.Timestamp("2025-01-15")
    assert feats.loc["101", "last_gift_date"] == pd.Timestamp("2025-06-01")


def test_distinct_financial_types(contributions):
    feats = civicrm_contributions_to_features(contributions)
    assert int(feats.loc["101", "distinct_financial_types"]) == 2
    assert int(feats.loc["202", "distinct_financial_types"]) == 1


def test_financial_type_id_used_when_label_absent():
    rows = [
        {"contact_id": "1", "receive_date": "2025-01-01", "total_amount": "10",
         "financial_type_id": "1"},
        {"contact_id": "1", "receive_date": "2025-02-01", "total_amount": "10",
         "financial_type_id": "4"},
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats = civicrm_contributions_to_features(rows)
    assert int(feats.loc["1", "distinct_financial_types"]) == 2


def test_no_financial_type_column_yields_zero():
    rows = [{"contact_id": "1", "receive_date": "2025-01-01", "total_amount": "10"}]
    feats = civicrm_contributions_to_features(rows, statuses=None)
    assert int(feats.loc["1", "distinct_financial_types"]) == 0


def test_recency_features_non_negative(contributions):
    feats = civicrm_contributions_to_features(contributions)
    assert (feats["recency_days"] >= 0).all()
    assert (feats["years_active"] >= 0.0).all()


def test_reference_date_shifts_recency(contributions):
    early = civicrm_contributions_to_features(contributions,
                                              reference_date="2025-06-01")
    late = civicrm_contributions_to_features(contributions,
                                             reference_date="2026-06-01")
    assert int(late.loc["101", "recency_days"]) - \
        int(early.loc["101", "recency_days"]) == 365


def test_reference_date_nat_falls_back_to_batch_max(contributions):
    default = civicrm_contributions_to_features(contributions)
    nat = civicrm_contributions_to_features(contributions, reference_date=pd.NaT)
    assert nat["recency_days"].equals(default["recency_days"])


def test_carries_identity_fields(contributions):
    feats = civicrm_contributions_to_features(contributions)
    assert feats.loc["101", "constituent_email"] == "ada@uni.edu"
    assert feats.loc["101", "first_name"] == "Ada"
    assert feats.loc["202", "last_name"] == "Hopper"


def test_identity_fields_absent_yields_null_column_not_crash():
    rows = [{"contact_id": "1", "receive_date": "2025-01-01", "total_amount": "10"}]
    feats = civicrm_contributions_to_features(rows, statuses=None)
    assert feats.loc["1", "constituent_email"] is None
    assert feats.loc["1", "first_name"] is None


# --------------------------------------------------------------------------- #
# Status / test-mode filtering — the reason this bridge exists
# --------------------------------------------------------------------------- #
def test_statuses_none_counts_every_row(contributions):
    feats = civicrm_contributions_to_features(contributions, statuses=None)
    # The Failed 99 is now included; the test-mode row still is not.
    assert float(feats.loc["101", "total_gift_amount"]) == 1349.0


def test_statuses_can_select_other_statuses():
    rows = [
        {"contact_id": "1", "receive_date": "2025-01-01", "total_amount": "10",
         "contribution_status": "Completed"},
        {"contact_id": "1", "receive_date": "2025-02-01", "total_amount": "40",
         "contribution_status": "Pending"},
    ]
    feats = civicrm_contributions_to_features(rows, statuses=("Completed", "Pending"))
    assert float(feats.loc["1", "total_gift_amount"]) == 50.0


def test_status_match_is_case_insensitive():
    rows = [{"contact_id": "1", "receive_date": "2025-01-01",
             "total_amount": "10", "contribution_status": " completed "}]
    feats = civicrm_contributions_to_features(rows)
    assert int(feats.loc["1", "gift_count"]) == 1


def test_missing_status_column_warns():
    rows = [{"contact_id": "1", "receive_date": "2025-01-01", "total_amount": "10"}]
    with pytest.warns(UserWarning, match="no contribution_status column"):
        civicrm_contributions_to_features(rows)


def test_statuses_none_does_not_warn_about_missing_status():
    rows = [{"contact_id": "1", "receive_date": "2025-01-01", "total_amount": "10"}]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        civicrm_contributions_to_features(rows, statuses=None)


def test_numeric_is_test_column_is_dropped():
    rows = pd.DataFrame({
        "contact_id": ["1", "1"],
        "receive_date": ["2025-01-01", "2025-02-01"],
        "total_amount": [10.0, 999.0],
        "is_test": [0, 1],
    })
    feats = civicrm_contributions_to_features(rows, statuses=None)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


def test_boolean_is_test_column_is_dropped():
    rows = pd.DataFrame({
        "contact_id": ["1", "1"],
        "receive_date": ["2025-01-01", "2025-02-01"],
        "total_amount": [10.0, 999.0],
        "is_test": [False, True],
    })
    feats = civicrm_contributions_to_features(rows, statuses=None)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


def test_textual_test_mode_column_is_dropped():
    rows = [
        {"contact_id": "1", "Contribution Date": "2025-01-01",
         "Total Amount": "10", "Test Mode": "No"},
        {"contact_id": "1", "Contribution Date": "2025-02-01",
         "Total Amount": "999", "Test Mode": "Yes"},
    ]
    feats = civicrm_contributions_to_features(rows, statuses=None)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


def test_every_row_filtered_out_returns_typed_empty_frame():
    rows = [{"contact_id": "1", "receive_date": "2025-01-01",
             "total_amount": "10", "contribution_status": "Refunded"}]
    feats = civicrm_contributions_to_features(rows)
    assert feats.empty
    assert list(feats.columns) == list(_FEATURE_DTYPES)


# --------------------------------------------------------------------------- #
# Header normalisation
# --------------------------------------------------------------------------- #
def test_export_labels_and_apiv4_names_agree(contributions):
    labelled = [
        {"Contribution ID": r["id"], "Contact ID": r["contact_id"],
         "Contribution Date": r["receive_date"], "Total Amount": r["total_amount"],
         "Currency": r["currency"], "Contribution Status": r["contribution_status"],
         "Financial Type": r["financial_type"], "Test Mode": r["is_test"],
         "Email": r["email"], "First Name": r["first_name"],
         "Last Name": r["last_name"]}
        for r in contributions
    ]
    assert civicrm_contributions_to_features(labelled).equals(
        civicrm_contributions_to_features(contributions)
    )


def test_apiv4_pseudo_fields_are_recognised():
    rows = [{
        "contact_id": 101, "receive_date": "2025-01-15T00:00:00Z",
        "total_amount": 250.0, "contribution_status_id:name": "Completed",
        "financial_type_id:label": "Donation",
    }, {
        "contact_id": 101, "receive_date": "2025-02-15T00:00:00Z",
        "total_amount": 40.0, "contribution_status_id:name": "Failed",
        "financial_type_id:label": "Donation",
    }]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        feats = civicrm_contributions_to_features(rows)
    assert float(feats.loc["101", "total_gift_amount"]) == 250.0
    assert int(feats.loc["101", "distinct_financial_types"]) == 1


def test_date_received_label_is_accepted():
    rows = [{"Contact ID": "1", "Date Received": "2025-01-01",
             "Amount": "10", "Contribution Status": "Completed"}]
    feats = civicrm_contributions_to_features(rows)
    assert feats.loc["1", "first_gift_date"] == pd.Timestamp("2025-01-01")


def test_duplicate_header_collapses_to_the_first():
    # A wide export carrying both "Amount" and "Total Amount" must stay 1-D.
    rows = [{"Contact ID": "1", "Receive Date": "2025-01-01",
             "Amount": "10", "Total Amount": "999"}]
    feats = civicrm_contributions_to_features(rows, statuses=None)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


def test_missing_required_column_raises_keyerror():
    rows = [{"contact_id": "1", "total_amount": "10"}]
    with pytest.raises(KeyError, match="receive_date"):
        civicrm_contributions_to_features(rows)


# --------------------------------------------------------------------------- #
# Value coercion
# --------------------------------------------------------------------------- #
def test_currency_formatted_amounts_parse():
    rows = [{"contact_id": "1", "receive_date": "2025-01-01",
             "total_amount": "$1,250.50"}]
    feats = civicrm_contributions_to_features(rows, statuses=None)
    assert float(feats.loc["1", "total_gift_amount"]) == 1250.50


def test_numeric_amount_column_passes_through():
    rows = pd.DataFrame({
        "contact_id": ["1"], "receive_date": ["2025-01-01"],
        "total_amount": [1250.5],
    })
    feats = civicrm_contributions_to_features(rows, statuses=None)
    assert float(feats.loc["1", "total_gift_amount"]) == 1250.5


def test_unparseable_amount_counts_as_zero():
    rows = [
        {"contact_id": "1", "receive_date": "2025-01-01", "total_amount": "n/a"},
        {"contact_id": "1", "receive_date": "2025-02-01", "total_amount": "10"},
    ]
    feats = civicrm_contributions_to_features(rows, statuses=None)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0
    assert int(feats.loc["1", "gift_count"]) == 2


def test_locale_formatted_date_parses_month_first():
    rows = [{"contact_id": "1", "receive_date": "03/01/2025", "total_amount": "10"}]
    feats = civicrm_contributions_to_features(rows, statuses=None)
    assert feats.loc["1", "first_gift_date"] == pd.Timestamp("2025-03-01")


def test_already_parsed_datetime_column_passes_through():
    # The documented escape hatch for a dd/mm/yyyy site: hand in real datetimes.
    rows = pd.DataFrame({
        "contact_id": ["1"],
        "receive_date": pd.to_datetime(["03/01/2025"], dayfirst=True),
        "total_amount": [10.0],
    })
    feats = civicrm_contributions_to_features(rows, statuses=None)
    assert feats.loc["1", "first_gift_date"] == pd.Timestamp("2025-01-03")


def test_timezone_aware_datetime_column_normalises_to_naive_utc():
    rows = pd.DataFrame({
        "contact_id": ["1"],
        "receive_date": pd.to_datetime(["2025-01-01 00:00:00"]).tz_localize(
            "America/Chicago"
        ),
        "total_amount": [10.0],
    })
    feats = civicrm_contributions_to_features(rows, statuses=None)
    assert feats.loc["1", "first_gift_date"] == pd.Timestamp("2025-01-01 06:00:00")


def test_timezone_aware_dates_normalise_to_naive_utc():
    rows = [{"contact_id": "1", "receive_date": "2025-01-01T23:00:00-05:00",
             "total_amount": "10"}]
    feats = civicrm_contributions_to_features(rows, statuses=None)
    assert feats.loc["1", "first_gift_date"] == pd.Timestamp("2025-01-02 04:00:00")


def test_unparseable_date_row_dropped():
    rows = [
        {"contact_id": "1", "receive_date": "not-a-date", "total_amount": "999"},
        {"contact_id": "1", "receive_date": "2025-01-01", "total_amount": "10"},
    ]
    feats = civicrm_contributions_to_features(rows, statuses=None)
    assert int(feats.loc["1", "gift_count"]) == 1
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


def test_blank_contact_id_row_dropped():
    rows = [
        {"contact_id": "  ", "receive_date": "2025-01-01", "total_amount": "999"},
        {"contact_id": "1", "receive_date": "2025-01-01", "total_amount": "10"},
    ]
    feats = civicrm_contributions_to_features(rows, statuses=None)
    assert list(feats.index) == ["1"]


def test_parsing_emits_no_warning(contributions):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        civicrm_contributions_to_features(contributions)


# --------------------------------------------------------------------------- #
# Deduplication and currency
# --------------------------------------------------------------------------- #
def test_deduplicates_by_contribution_id(contributions):
    once = civicrm_contributions_to_features(contributions)
    twice = civicrm_contributions_to_features(contributions + contributions)
    assert twice["total_gift_amount"].equals(once["total_gift_amount"])
    assert twice["gift_count"].equals(once["gift_count"])


def test_dedup_preserves_rows_without_a_contribution_id():
    rows = [
        {"id": "1", "contact_id": "1", "receive_date": "2025-01-01",
         "total_amount": "10"},
        {"contact_id": "1", "receive_date": "2025-02-01", "total_amount": "20"},
        {"contact_id": "1", "receive_date": "2025-03-01", "total_amount": "30"},
    ]
    feats = civicrm_contributions_to_features(rows, statuses=None)
    assert int(feats.loc["1", "gift_count"]) == 3
    assert float(feats.loc["1", "total_gift_amount"]) == 60.0


def test_mixed_currency_warns(contributions):
    rows = contributions + [
        {"id": "6", "contact_id": "303", "receive_date": "2025-05-01",
         "total_amount": "100.00", "currency": "EUR",
         "contribution_status": "Completed", "financial_type": "Donation",
         "is_test": "0"},
    ]
    with pytest.warns(UserWarning, match="mixes currencies"):
        civicrm_contributions_to_features(rows)


def test_single_currency_does_not_warn(contributions):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        civicrm_contributions_to_features(contributions)


# --------------------------------------------------------------------------- #
# Empty inputs
# --------------------------------------------------------------------------- #
def test_empty_list_returns_typed_empty_frame():
    feats = civicrm_contributions_to_features([])
    assert feats.empty
    assert list(feats.columns) == list(_FEATURE_DTYPES)
    assert feats.index.name == "contact_id"
    for col, dtype in _FEATURE_DTYPES.items():
        assert feats[col].dtype == np.dtype(dtype), col


def test_empty_dataframe_with_columns_returns_typed_empty_frame():
    feats = civicrm_contributions_to_features(
        pd.DataFrame(columns=["contact_id", "receive_date", "total_amount"])
    )
    assert feats.empty
    assert list(feats.columns) == list(_FEATURE_DTYPES)


# --------------------------------------------------------------------------- #
# read_civicrm_contributions
# --------------------------------------------------------------------------- #
def test_read_single_csv(tmp_path):
    f = tmp_path / "contributions.csv"
    f.write_text(_export(
        _row(1, 101, "2025-01-15", "250.00", email="ada@uni.edu"),
        _row(2, 202, "2025-03-01", "5000.00", email="grace@uni.edu"),
    ))
    df = read_civicrm_contributions(f)
    assert len(df) == 2
    assert "total_amount" in df.columns and "receive_date" in df.columns
    assert list(df["contact_id"]) == ["101", "202"]


def test_read_is_lossless_and_features_applies_the_policy(tmp_path):
    f = tmp_path / "contributions.csv"
    f.write_text(_export(
        _row(1, 101, "2025-01-15", "250.00"),
        _row(2, 101, "2025-02-15", "99.00", status="Refunded"),
        _row(3, 101, "2025-03-15", "999.00", test="1"),
    ))
    df = read_civicrm_contributions(f)
    assert len(df) == 3, "reading must not drop rows"
    feats = civicrm_contributions_to_features(df)
    assert float(feats.loc["101", "total_gift_amount"]) == 250.0


def test_read_strips_the_excel_bom(tmp_path):
    f = tmp_path / "excel.csv"
    f.write_text(_export(_row(1, 101, "2025-01-15", "250.00")),
                 encoding="utf-8-sig")
    df = read_civicrm_contributions(f)
    assert "contact_id" in df.columns


def test_read_directory_concatenates_in_sorted_order(tmp_path):
    (tmp_path / "2025-02.csv").write_text(_export(
        _row(2, 202, "2025-02-01", "20.00")))
    (tmp_path / "2025-01.csv").write_text(_export(
        _row(1, 101, "2025-01-01", "10.00")))
    df = read_civicrm_contributions(tmp_path)
    assert list(df["contact_id"]) == ["101", "202"]


def test_read_recurses_into_subdirectories(tmp_path):
    nested = tmp_path / "2025" / "q1"
    nested.mkdir(parents=True)
    (nested / "jan.csv").write_text(_export(_row(1, 101, "2025-01-01", "10.00")))
    (tmp_path / "top.csv").write_text(_export(_row(2, 202, "2025-02-01", "20.00")))
    df = read_civicrm_contributions(tmp_path)
    assert set(df["contact_id"]) == {"101", "202"}


def test_read_directory_with_no_csv_returns_empty_frame(tmp_path):
    (tmp_path / "notes.txt").write_text("not an export")
    df = read_civicrm_contributions(tmp_path)
    assert df.empty
    assert civicrm_contributions_to_features(df).empty


def test_read_does_not_follow_symlinks(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.csv").write_text(_export(_row(9, 999, "2025-01-01", "1.00")))
    exports = tmp_path / "exports"
    exports.mkdir()
    (exports / "real.csv").write_text(_export(_row(1, 101, "2025-01-01", "10.00")))
    try:
        (exports / "link.csv").symlink_to(outside / "secret.csv")
    except (OSError, NotImplementedError):  # pragma: no cover - platform-dependent
        pytest.skip("symlinks unavailable on this platform")
    df = read_civicrm_contributions(exports)
    assert list(df["contact_id"]) == ["101"]


def test_read_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_civicrm_contributions(tmp_path / "nope.csv")


# --------------------------------------------------------------------------- #
# Integration: bridge output flows into an estimator
# --------------------------------------------------------------------------- #
def test_features_feed_donor_propensity_model():
    from philanthropy.datasets import generate_synthetic_donor_data
    from philanthropy.models import DonorPropensityModel

    rng = np.random.default_rng(0)
    rows = []
    contribution_id = 0
    for donor in range(40):
        for gift in range(int(rng.integers(1, 5))):
            contribution_id += 1
            rows.append({
                "id": str(contribution_id),
                "contact_id": str(1000 + donor),
                "receive_date": f"2025-0{1 + gift}-15",
                "total_amount": f"{rng.uniform(25, 5000):.2f}",
                "currency": "USD",
                "contribution_status": "Completed",
                "financial_type": "Donation",
                "is_test": "0",
            })
    feats = civicrm_contributions_to_features(rows, reference_date="2026-01-01")

    cols = ["total_gift_amount", "years_active", "gift_count"]
    history = generate_synthetic_donor_data(n_samples=300, random_state=0)
    model = DonorPropensityModel(n_estimators=25, random_state=0).fit(
        history[["total_gift_amount", "years_active", "event_attendance_count"]]
        .to_numpy(),
        history["is_major_donor"].to_numpy(),
    )
    scores = model.predict_affinity_score(feats[cols].to_numpy())
    assert scores.shape == (len(feats),)
    assert np.isfinite(scores).all()
