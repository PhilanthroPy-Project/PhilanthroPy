"""
tests/test_donorperfect.py
Tests for the philanthropy.ingest DonorPerfect gift bridge.

The point of the module is one filter, but DonorPerfect's own field for it
is not called "gift type": in DonorPerfect a Pledge record (record_type='P')
and a split gift's Main total (record_type='M') both duplicate money that is
recorded again elsewhere (as the pledge's payments, or as the split's own
entries), so a naive sum counts each of those dollars twice. Most of what
follows checks that the Pledge and Main rows leave and the regular gift rows
(record_type='G', whether a plain gift, a pledge payment, or a split entry)
stay.
"""

import warnings

import pandas as pd
import pytest

from philanthropy.ingest import (
    DEFAULT_EXCLUDED_RECORD_TYPES,
    donorperfect_gifts_to_features,
    read_donorperfect_gifts,
)
from philanthropy.ingest._civicrm import _FEATURE_DTYPES

# Header labels as a DonorPerfect "Gift/Pledge Transactions" export writes them.
_HEADER = (
    "Donor ID,Gift Date,Gift Amount,Record Type,GL Code,Email,"
    "First Name,Last Name"
)


def _row(contact, date, amount, record_type, *, gl_code="General",
         email="", first="", last=""):
    return f"{contact},{date},{amount},{record_type},{gl_code},{email},{first},{last}"


def _export(*rows):
    return "\n".join((_HEADER,) + rows) + "\n"


@pytest.fixture
def gifts():
    """One donor with a $1,200 pledge paid in two $100 payments, and one with
    a $50 split gift Main total plus its own $50 split entry."""
    return [
        {"Donor ID": "88", "Gift Date": "2025-01-10",
         "Gift Amount": "1200.00", "Record Type": "P", "GL Code": "General",
         "Email": "ada@amc.edu", "First Name": "Ada", "Last Name": "Lovelace"},
        {"Donor ID": "88", "Gift Date": "2025-02-10",
         "Gift Amount": "100.00", "Record Type": "G", "GL Code": "General",
         "Email": "ada@amc.edu", "First Name": "Ada", "Last Name": "Lovelace"},
        {"Donor ID": "88", "Gift Date": "2025-03-10",
         "Gift Amount": "100.00", "Record Type": "G", "GL Code": "General",
         "Email": "ada@amc.edu", "First Name": "Ada", "Last Name": "Lovelace"},
        {"Donor ID": "91", "Gift Date": "2025-02-01",
         "Gift Amount": "50.00", "Record Type": "M",
         "GL Code": "Research", "Email": "grace@amc.edu",
         "First Name": "Grace", "Last Name": "Hopper"},
        {"Donor ID": "91", "Gift Date": "2025-02-01",
         "Gift Amount": "50.00", "Record Type": "G",
         "GL Code": "Capital", "Email": "grace@amc.edu",
         "First Name": "Grace", "Last Name": "Hopper"},
    ]


# --------------------------------------------------------------------------- #
# The commitment/split-versus-money filter
# --------------------------------------------------------------------------- #
def test_pledge_is_excluded_and_its_payments_are_not(gifts):
    feats = donorperfect_gifts_to_features(gifts)
    # 1200 (the promise) + 100 + 100 would be 1400; only the payments count.
    assert float(feats.loc["88", "total_gift_amount"]) == 200.0
    assert int(feats.loc["88", "gift_count"]) == 2


def test_split_main_total_is_excluded_and_its_split_is_not(gifts):
    feats = donorperfect_gifts_to_features(gifts)
    # 50 (the Main total) + 50 (the split) would be 100; only the split counts.
    assert float(feats.loc["91", "total_gift_amount"]) == 50.0
    assert int(feats.loc["91", "gift_count"]) == 1


@pytest.mark.parametrize("record_type", ["P", "M"])
def test_every_default_excluded_type_is_dropped(record_type):
    rows = [{"Donor ID": "1", "Gift Date": "2025-01-01",
             "Gift Amount": "500.00", "Record Type": record_type},
            {"Donor ID": "1", "Gift Date": "2025-01-02",
             "Gift Amount": "10.00", "Record Type": "G"}]
    feats = donorperfect_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


def test_regular_gift_type_is_kept():
    rows = [{"Donor ID": "1", "Gift Date": "2025-01-01",
             "Gift Amount": "10.00", "Record Type": "G"}]
    feats = donorperfect_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


@pytest.mark.parametrize("record_type", ["p", "P", "  P  ", "m", "M"])
def test_exclusion_matching_ignores_case_and_spacing(record_type):
    rows = [{"Donor ID": "1", "Gift Date": "2025-01-01",
             "Gift Amount": "500.00", "Record Type": record_type},
            {"Donor ID": "1", "Gift Date": "2025-01-02",
             "Gift Amount": "10.00", "Record Type": "G"}]
    feats = donorperfect_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


def test_blank_record_type_is_kept():
    """An unlabelled row cannot be shown to be a commitment, so it stays."""
    rows = [{"Donor ID": "1", "Gift Date": "2025-01-01",
             "Gift Amount": "10.00", "Record Type": ""},
            {"Donor ID": "1", "Gift Date": "2025-01-02",
             "Gift Amount": "5.00", "Record Type": None}]
    feats = donorperfect_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 15.0
    assert int(feats.loc["1", "gift_count"]) == 2


def test_custom_exclude_set_replaces_the_default(gifts):
    feats = donorperfect_gifts_to_features(gifts, exclude_record_types=("G",))
    # Pledge is no longer excluded; the two G rows are.
    assert float(feats.loc["88", "total_gift_amount"]) == 1200.0


def test_exclude_none_counts_every_row(gifts):
    feats = donorperfect_gifts_to_features(gifts, exclude_record_types=None)
    assert float(feats.loc["88", "total_gift_amount"]) == 1400.0
    assert int(feats.loc["88", "gift_count"]) == 3


def test_empty_exclude_sequence_counts_every_row(gifts):
    feats = donorperfect_gifts_to_features(gifts, exclude_record_types=())
    assert float(feats.loc["88", "total_gift_amount"]) == 1400.0


def test_missing_record_type_column_warns():
    rows = [{"Donor ID": "1", "Gift Date": "2025-01-01", "Gift Amount": "10.00"}]
    with pytest.warns(UserWarning, match="no record type column"):
        feats = donorperfect_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


def test_exclude_none_does_not_warn_about_a_missing_record_type_column():
    rows = [{"Donor ID": "1", "Gift Date": "2025-01-01", "Gift Amount": "10.00"}]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        donorperfect_gifts_to_features(rows, exclude_record_types=None)


def test_default_excluded_set_is_documented_and_non_empty():
    assert "P" in DEFAULT_EXCLUDED_RECORD_TYPES
    assert "M" in DEFAULT_EXCLUDED_RECORD_TYPES


# --------------------------------------------------------------------------- #
# Header dialects
# --------------------------------------------------------------------------- #
def test_db_column_names_are_accepted():
    """The DB/API field names are DONOR_ID / GIFT_DATE / AMOUNT / RECORD_TYPE."""
    rows = [{"DONOR_ID": "7", "GIFT_DATE": "2025-05-01", "AMOUNT": "300.00",
             "RECORD_TYPE": "G"},
            {"DONOR_ID": "7", "GIFT_DATE": "2025-05-02", "AMOUNT": "999.00",
             "RECORD_TYPE": "P"}]
    feats = donorperfect_gifts_to_features(rows)
    assert float(feats.loc["7", "total_gift_amount"]) == 300.0


def test_export_labels_and_db_names_agree(gifts):
    db = [
        {"DONOR_ID": g["Donor ID"], "GIFT_DATE": g["Gift Date"],
         "AMOUNT": g["Gift Amount"], "RECORD_TYPE": g["Record Type"],
         "GL_CODE": g["GL Code"], "email": g["Email"],
         "first_name": g["First Name"], "last_name": g["Last Name"]}
        for g in gifts
    ]
    pd.testing.assert_frame_equal(
        donorperfect_gifts_to_features(gifts),
        donorperfect_gifts_to_features(db),
    )


def test_currency_formatted_amounts_parse():
    rows = [{"Donor ID": "1", "Gift Date": "2025-01-01",
             "Gift Amount": "$1,250.00", "Record Type": "G"}]
    feats = donorperfect_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 1250.0


# --------------------------------------------------------------------------- #
# Schema, identity and recency
# --------------------------------------------------------------------------- #
def test_schema_and_dtypes(gifts):
    feats = donorperfect_gifts_to_features(gifts)
    assert list(feats.columns) == list(_FEATURE_DTYPES)
    assert feats.index.name == "contact_id"


def test_gl_code_feeds_distinct_financial_types(gifts):
    feats = donorperfect_gifts_to_features(gifts, exclude_record_types=None)
    # Donor 91's two rows sit on Research and Capital.
    assert int(feats.loc["91", "distinct_financial_types"]) == 2
    assert int(feats.loc["88", "distinct_financial_types"]) == 1


def test_carries_identity_fields(gifts):
    feats = donorperfect_gifts_to_features(gifts)
    assert feats.loc["88", "constituent_email"] == "ada@amc.edu"
    assert feats.loc["88", "first_name"] == "Ada"
    assert feats.loc["88", "last_name"] == "Lovelace"


def test_reference_date_shifts_recency(gifts):
    feats = donorperfect_gifts_to_features(gifts, reference_date="2025-12-31")
    assert int(feats.loc["88", "recency_days"]) == 296


def test_recency_anchors_on_the_batch_by_default(gifts):
    feats = donorperfect_gifts_to_features(gifts)
    # Latest surviving gift is donor 88's 2025-03-10 payment.
    assert int(feats.loc["88", "recency_days"]) == 0


def test_empty_input_returns_a_typed_empty_frame():
    feats = donorperfect_gifts_to_features([])
    assert feats.empty
    assert list(feats.columns) == list(_FEATURE_DTYPES)


def test_every_row_filtered_out_returns_a_typed_empty_frame():
    rows = [{"Donor ID": "1", "Gift Date": "2025-01-01",
             "Gift Amount": "500.00", "Record Type": "P"}]
    feats = donorperfect_gifts_to_features(rows)
    assert feats.empty
    assert list(feats.columns) == list(_FEATURE_DTYPES)
    assert feats.index.name == "contact_id"


def test_missing_required_column_names_the_donorperfect_fields():
    rows = [{"Donor ID": "1", "Record Type": "G"}]
    with pytest.raises(KeyError) as excinfo:
        donorperfect_gifts_to_features(rows)
    message = str(excinfo.value)
    assert "DonorPerfect" in message
    assert "Gift Date" in message
    # A CiviCRM-worded error here would read as "you used the wrong reader".
    assert "CiviCRM" not in message


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def test_read_single_csv(tmp_path):
    path = tmp_path / "gifts.csv"
    path.write_text(_export(_row("88", "2025-01-10", "1200.00", "P"),
                            _row("88", "2025-02-10", "100.00", "G")))
    raw = read_donorperfect_gifts(path)
    assert len(raw) == 2
    # Reading is lossless: the pledge row is still there.
    feats = donorperfect_gifts_to_features(raw)
    assert float(feats.loc["88", "total_gift_amount"]) == 100.0


def test_read_directory_concatenates(tmp_path):
    (tmp_path / "jan.csv").write_text(
        _export(_row("88", "2025-01-10", "100.00", "G"))
    )
    (tmp_path / "feb.csv").write_text(
        _export(_row("88", "2025-02-10", "250.00", "G"))
    )
    feats = donorperfect_gifts_to_features(read_donorperfect_gifts(tmp_path))
    assert float(feats.loc["88", "total_gift_amount"]) == 350.0


def test_read_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_donorperfect_gifts(tmp_path / "nope.csv")


def test_numeric_looking_donor_ids_stay_text(tmp_path):
    """dtype=str on the read keeps 0088 from becoming 88, and a blank in the
    column from turning the whole thing into floats."""
    path = tmp_path / "gifts.csv"
    path.write_text(_export(_row("0088", "2025-01-10", "100.00", "G"),
                            _row("", "2025-01-11", "50.00", "G")))
    feats = donorperfect_gifts_to_features(read_donorperfect_gifts(path))
    assert list(feats.index) == ["0088"]
