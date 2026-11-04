"""
Tests for the philanthropy.ingest Neon CRM donation bridge.
"""

import warnings

import pandas as pd
import pytest

from philanthropy.ingest import (
    DEFAULT_EXCLUDED_NEON_DONATION_TYPES,
    neon_donations_to_features,
    read_neon_donations,
)
from philanthropy.ingest._civicrm import _FEATURE_DTYPES

_HEADER = "Account ID,Donation Date,Donation Amount,Donation Type,Fund,Email,First Name,Last Name"


def _row(contact, date, amount, donation_type, *, fund="General",
         email="", first="", last=""):
    return f"{contact},{date},{amount},{donation_type},{fund},{email},{first},{last}"


def _export(*rows):
    return "\n".join((_HEADER,) + rows) + "\n"


@pytest.fixture
def donations():
    return [
        {"Account ID": "88", "Donation Date": "2025-01-10",
         "Donation Amount": "1200.00", "Donation Type": "Pledge",
         "Fund": "General", "Email": "ada@amc.edu",
         "First Name": "Ada", "Last Name": "Lovelace"},
        {"Account ID": "88", "Donation Date": "2025-02-10",
         "Donation Amount": "100.00", "Donation Type": "Pledge Payment",
         "Fund": "General", "Email": "ada@amc.edu",
         "First Name": "Ada", "Last Name": "Lovelace"},
        {"Account ID": "88", "Donation Date": "2025-03-10",
         "Donation Amount": "100.00", "Donation Type": "Donation",
         "Fund": "Research", "Email": "ada@amc.edu",
         "First Name": "Ada", "Last Name": "Lovelace"},
    ]


def test_pledge_is_excluded_and_pledge_payment_is_kept(donations):
    feats = neon_donations_to_features(donations)
    assert float(feats.loc["88", "total_gift_amount"]) == 200.0
    assert int(feats.loc["88", "gift_count"]) == 2


@pytest.mark.parametrize("donation_type", ["Pledge", "Matching Pledge"])
def test_default_excluded_types_are_dropped(donation_type):
    rows = [
        {"Account ID": "1", "Donation Date": "2025-01-01",
         "Donation Amount": "500.00", "Donation Type": donation_type},
        {"Account ID": "1", "Donation Date": "2025-01-02",
         "Donation Amount": "25.00", "Donation Type": "Donation"},
    ]
    feats = neon_donations_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 25.0


@pytest.mark.parametrize(
    "donation_type",
    ["pledge", "PLEDGE", "  Pledge  ", "MatchingPledge", "matching_pledge"],
)
def test_exclusion_matching_ignores_case_spacing_and_punctuation(donation_type):
    rows = [
        {"Account ID": "1", "Donation Date": "2025-01-01",
         "Donation Amount": "500.00", "Donation Type": donation_type},
        {"Account ID": "1", "Donation Date": "2025-01-02",
         "Donation Amount": "25.00", "Donation Type": "Donation"},
    ]
    feats = neon_donations_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 25.0


@pytest.mark.parametrize("donation_type", ["Donation", "Pledge Payment"])
def test_payment_types_are_kept(donation_type):
    rows = [{"Account ID": "1", "Donation Date": "2025-01-01",
             "Donation Amount": "10.00", "Donation Type": donation_type}]
    feats = neon_donations_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


def test_custom_exclude_set_replaces_the_default(donations):
    feats = neon_donations_to_features(
        donations, exclude_donation_types=("Pledge Payment",)
    )
    assert float(feats.loc["88", "total_gift_amount"]) == 1300.0


def test_exclude_none_counts_every_row(donations):
    feats = neon_donations_to_features(donations, exclude_donation_types=None)
    assert float(feats.loc["88", "total_gift_amount"]) == 1400.0


def test_empty_exclude_sequence_counts_every_row(donations):
    feats = neon_donations_to_features(donations, exclude_donation_types=())
    assert float(feats.loc["88", "total_gift_amount"]) == 1400.0


def test_missing_donation_type_column_warns():
    rows = [{"Account ID": "1", "Donation Date": "2025-01-01",
             "Donation Amount": "10.00"}]
    with pytest.warns(UserWarning, match="no donation type column"):
        feats = neon_donations_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


def test_exclude_none_does_not_warn_about_a_missing_donation_type_column():
    rows = [{"Account ID": "1", "Donation Date": "2025-01-01",
             "Donation Amount": "10.00"}]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        neon_donations_to_features(rows, exclude_donation_types=None)


def test_default_excluded_set_is_documented_and_non_empty():
    assert "Pledge" in DEFAULT_EXCLUDED_NEON_DONATION_TYPES
    assert "Matching Pledge" in DEFAULT_EXCLUDED_NEON_DONATION_TYPES


def test_api_field_names_are_accepted():
    rows = [
        {"linkedAccountId": "7", "date": "2025-05-01", "amount": "300.00",
         "type": "Donation"},
        {"linkedAccountId": "7", "date": "2025-05-02", "amount": "999.00",
         "type": "Pledge"},
    ]
    feats = neon_donations_to_features(rows)
    assert float(feats.loc["7", "total_gift_amount"]) == 300.0


def test_export_labels_and_api_names_agree(donations):
    api = [
        {"linkedAccountId": t["Account ID"], "date": t["Donation Date"],
         "amount": t["Donation Amount"], "type": t["Donation Type"],
         "fundName": t["Fund"], "email": t["Email"],
         "first_name": t["First Name"], "last_name": t["Last Name"]}
        for t in donations
    ]
    pd.testing.assert_frame_equal(
        neon_donations_to_features(donations),
        neon_donations_to_features(api),
    )


def test_currency_formatted_amounts_parse():
    rows = [{"Account ID": "1", "Donation Date": "2025-01-01",
             "Donation Amount": "$1,250.00", "Donation Type": "Donation"}]
    feats = neon_donations_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 1250.0


def test_schema_and_dtypes(donations):
    feats = neon_donations_to_features(donations)
    assert list(feats.columns) == list(_FEATURE_DTYPES)
    assert feats.index.name == "contact_id"


def test_fund_feeds_distinct_financial_types(donations):
    feats = neon_donations_to_features(donations)
    assert int(feats.loc["88", "distinct_financial_types"]) == 2


def test_carries_identity_fields(donations):
    feats = neon_donations_to_features(donations)
    assert feats.loc["88", "constituent_email"] == "ada@amc.edu"
    assert feats.loc["88", "first_name"] == "Ada"
    assert feats.loc["88", "last_name"] == "Lovelace"


def test_reference_date_shifts_recency(donations):
    feats = neon_donations_to_features(donations, reference_date="2025-12-31")
    assert int(feats.loc["88", "recency_days"]) == 296


def test_empty_input_returns_a_typed_empty_frame():
    feats = neon_donations_to_features([])
    assert feats.empty
    assert list(feats.columns) == list(_FEATURE_DTYPES)


def test_every_row_filtered_out_returns_a_typed_empty_frame():
    rows = [{"Account ID": "1", "Donation Date": "2025-01-01",
             "Donation Amount": "500.00", "Donation Type": "Pledge"}]
    feats = neon_donations_to_features(rows)
    assert feats.empty
    assert list(feats.columns) == list(_FEATURE_DTYPES)
    assert feats.index.name == "contact_id"


def test_missing_required_column_names_the_neon_fields():
    rows = [{"Account ID": "1", "Donation Type": "Donation"}]
    with pytest.raises(KeyError) as excinfo:
        neon_donations_to_features(rows)
    message = str(excinfo.value)
    assert "Neon" in message
    assert "donation date" in message
    assert "CiviCRM" not in message


def test_read_single_csv(tmp_path):
    path = tmp_path / "donations.csv"
    path.write_text(_export(_row("88", "2025-01-10", "1200.00", "Pledge"),
                            _row("88", "2025-02-10", "100.00", "Pledge Payment")))
    raw = read_neon_donations(path)
    assert len(raw) == 2
    feats = neon_donations_to_features(raw)
    assert float(feats.loc["88", "total_gift_amount"]) == 100.0


def test_read_directory_concatenates(tmp_path):
    (tmp_path / "jan.csv").write_text(
        _export(_row("88", "2025-01-10", "100.00", "Donation"))
    )
    (tmp_path / "feb.csv").write_text(
        _export(_row("88", "2025-02-10", "250.00", "Donation"))
    )
    feats = neon_donations_to_features(read_neon_donations(tmp_path))
    assert float(feats.loc["88", "total_gift_amount"]) == 350.0
