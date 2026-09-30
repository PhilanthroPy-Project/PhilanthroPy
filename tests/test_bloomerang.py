"""
tests/test_bloomerang.py
Tests for the philanthropy.ingest Bloomerang transaction bridge.

The point of the module is one filter: in Bloomerang a pledge and the
payments against it are separate transaction records (as are a recurring
donation's schedule and its charged instalments), so a naive sum counts
every committed dollar twice. Most of what follows checks that the
commitment rows leave and the payment rows stay, in each spelling Bloomerang
writes them in.
"""

import warnings

import pandas as pd
import pytest

from philanthropy.ingest import (
    DEFAULT_EXCLUDED_ENTRY_TYPES,
    bloomerang_transactions_to_features,
    read_bloomerang_transactions,
)
from philanthropy.ingest._civicrm import _FEATURE_DTYPES

# Header labels as a Bloomerang report/CSV export writes them.
_HEADER = (
    "Account Number,Date,Amount,Transaction Type,Fund,Email,"
    "First Name,Last Name"
)


def _row(contact, date, amount, entry_type, *, fund="General",
         email="", first="", last=""):
    return f"{contact},{date},{amount},{entry_type},{fund},{email},{first},{last}"


def _export(*rows):
    return "\n".join((_HEADER,) + rows) + "\n"


@pytest.fixture
def transactions():
    """One donor with a $1,200 pledge paid in two $100 instalments, and one
    with a recurring donation schedule plus one charged payment against it."""
    return [
        {"Account Number": "88", "Date": "2025-01-10",
         "Amount": "1200.00", "Transaction Type": "Pledge", "Fund": "General",
         "Email": "ada@amc.edu", "First Name": "Ada", "Last Name": "Lovelace"},
        {"Account Number": "88", "Date": "2025-02-10",
         "Amount": "100.00", "Transaction Type": "PledgePayment", "Fund": "General",
         "Email": "ada@amc.edu", "First Name": "Ada", "Last Name": "Lovelace"},
        {"Account Number": "88", "Date": "2025-03-10",
         "Amount": "100.00", "Transaction Type": "PledgePayment", "Fund": "General",
         "Email": "ada@amc.edu", "First Name": "Ada", "Last Name": "Lovelace"},
        {"Account Number": "91", "Date": "2025-02-01",
         "Amount": "50.00", "Transaction Type": "Recurring Donation",
         "Fund": "Research", "Email": "grace@amc.edu",
         "First Name": "Grace", "Last Name": "Hopper"},
        {"Account Number": "91", "Date": "2025-03-01",
         "Amount": "50.00", "Transaction Type": "RecurringDonationPayment",
         "Fund": "Capital", "Email": "grace@amc.edu",
         "First Name": "Grace", "Last Name": "Hopper"},
    ]


# --------------------------------------------------------------------------- #
# The commitment-versus-payment filter
# --------------------------------------------------------------------------- #
def test_pledge_is_excluded_and_its_payments_are_not(transactions):
    feats = bloomerang_transactions_to_features(transactions)
    # 1200 (the promise) + 100 + 100 would be 1400; only the payments count.
    assert float(feats.loc["88", "total_gift_amount"]) == 200.0
    assert int(feats.loc["88", "gift_count"]) == 2


def test_recurring_donation_schedule_is_excluded_and_its_payment_is_not(transactions):
    feats = bloomerang_transactions_to_features(transactions)
    assert float(feats.loc["91", "total_gift_amount"]) == 50.0
    assert int(feats.loc["91", "gift_count"]) == 1


@pytest.mark.parametrize("entry_type", ["Pledge", "Recurring Donation"])
def test_every_default_excluded_type_is_dropped(entry_type):
    rows = [{"Account Number": "1", "Date": "2025-01-01",
             "Amount": "500.00", "Transaction Type": entry_type},
            {"Account Number": "1", "Date": "2025-01-02",
             "Amount": "10.00", "Transaction Type": "Donation"}]
    feats = bloomerang_transactions_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


@pytest.mark.parametrize(
    "entry_type",
    ["Donation", "PledgePayment", "Pledge Payment",
     "RecurringDonationPayment", "Recurring Donation Payment"],
)
def test_payment_and_outright_types_are_kept(entry_type):
    rows = [{"Account Number": "1", "Date": "2025-01-01",
             "Amount": "10.00", "Transaction Type": entry_type}]
    feats = bloomerang_transactions_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


@pytest.mark.parametrize(
    "entry_type",
    ["pledge", "PLEDGE", "  Pledge  ", "RecurringDonation",
     "recurring_donation", "RECURRING DONATION"],
)
def test_exclusion_matching_ignores_case_spacing_and_punctuation(entry_type):
    """The REST API writes RecurringDonation where a report writes Recurring
    Donation; the default set names one spelling and matches both."""
    rows = [{"Account Number": "1", "Date": "2025-01-01",
             "Amount": "500.00", "Transaction Type": entry_type},
            {"Account Number": "1", "Date": "2025-01-02",
             "Amount": "10.00", "Transaction Type": "Donation"}]
    feats = bloomerang_transactions_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


def test_a_pay_type_is_not_matched_by_its_commitment_prefix():
    """'RecurringDonationPayment' must not be swept up by 'Recurring Donation'."""
    rows = [{"Account Number": "1", "Date": "2025-01-01",
             "Amount": "25.00", "Transaction Type": "RecurringDonationPayment"}]
    feats = bloomerang_transactions_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 25.0


def test_blank_entry_type_is_kept():
    """An unlabelled row cannot be shown to be a commitment, so it stays."""
    rows = [{"Account Number": "1", "Date": "2025-01-01",
             "Amount": "10.00", "Transaction Type": ""},
            {"Account Number": "1", "Date": "2025-01-02",
             "Amount": "5.00", "Transaction Type": None}]
    feats = bloomerang_transactions_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 15.0
    assert int(feats.loc["1", "gift_count"]) == 2


def test_custom_exclude_set_replaces_the_default(transactions):
    feats = bloomerang_transactions_to_features(
        transactions, exclude_entry_types=("PledgePayment",)
    )
    # Pledge is no longer excluded; the two PledgePayment rows are.
    assert float(feats.loc["88", "total_gift_amount"]) == 1200.0


def test_exclude_none_counts_every_row(transactions):
    feats = bloomerang_transactions_to_features(transactions, exclude_entry_types=None)
    assert float(feats.loc["88", "total_gift_amount"]) == 1400.0
    assert int(feats.loc["88", "gift_count"]) == 3


def test_empty_exclude_sequence_counts_every_row(transactions):
    feats = bloomerang_transactions_to_features(transactions, exclude_entry_types=())
    assert float(feats.loc["88", "total_gift_amount"]) == 1400.0


def test_missing_entry_type_column_warns():
    rows = [{"Account Number": "1", "Date": "2025-01-01", "Amount": "10.00"}]
    with pytest.warns(UserWarning, match="no entry type column"):
        feats = bloomerang_transactions_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


def test_exclude_none_does_not_warn_about_a_missing_entry_type_column():
    rows = [{"Account Number": "1", "Date": "2025-01-01", "Amount": "10.00"}]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bloomerang_transactions_to_features(rows, exclude_entry_types=None)


def test_default_excluded_set_is_documented_and_non_empty():
    assert "Pledge" in DEFAULT_EXCLUDED_ENTRY_TYPES
    assert "Recurring Donation" in DEFAULT_EXCLUDED_ENTRY_TYPES


# --------------------------------------------------------------------------- #
# Header dialects
# --------------------------------------------------------------------------- #
def test_rest_api_field_names_are_accepted():
    """Bloomerang's REST API returns AccountId / Date / Amount / EntryType."""
    rows = [{"AccountId": "7", "Date": "2025-05-01", "Amount": "300.00",
             "EntryType": "Donation"},
            {"AccountId": "7", "Date": "2025-05-02", "Amount": "999.00",
             "EntryType": "Pledge"}]
    feats = bloomerang_transactions_to_features(rows)
    assert float(feats.loc["7", "total_gift_amount"]) == 300.0


def test_export_labels_and_api_names_agree(transactions):
    api = [
        {"AccountId": t["Account Number"], "Date": t["Date"],
         "Amount": t["Amount"], "EntryType": t["Transaction Type"],
         "FundName": t["Fund"], "email": t["Email"],
         "first_name": t["First Name"], "last_name": t["Last Name"]}
        for t in transactions
    ]
    pd.testing.assert_frame_equal(
        bloomerang_transactions_to_features(transactions),
        bloomerang_transactions_to_features(api),
    )


def test_currency_formatted_amounts_parse():
    rows = [{"Account Number": "1", "Date": "2025-01-01",
             "Amount": "$1,250.00", "Transaction Type": "Donation"}]
    feats = bloomerang_transactions_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 1250.0


# --------------------------------------------------------------------------- #
# Schema, identity and recency
# --------------------------------------------------------------------------- #
def test_schema_and_dtypes(transactions):
    feats = bloomerang_transactions_to_features(transactions)
    assert list(feats.columns) == list(_FEATURE_DTYPES)
    assert feats.index.name == "contact_id"


def test_fund_feeds_distinct_financial_types(transactions):
    feats = bloomerang_transactions_to_features(transactions, exclude_entry_types=None)
    # Donor 91's two rows sit on Research and Capital.
    assert int(feats.loc["91", "distinct_financial_types"]) == 2
    assert int(feats.loc["88", "distinct_financial_types"]) == 1


def test_carries_identity_fields(transactions):
    feats = bloomerang_transactions_to_features(transactions)
    assert feats.loc["88", "constituent_email"] == "ada@amc.edu"
    assert feats.loc["88", "first_name"] == "Ada"
    assert feats.loc["88", "last_name"] == "Lovelace"


def test_reference_date_shifts_recency(transactions):
    feats = bloomerang_transactions_to_features(transactions, reference_date="2025-12-31")
    assert int(feats.loc["88", "recency_days"]) == 296


def test_recency_anchors_on_the_batch_by_default(transactions):
    feats = bloomerang_transactions_to_features(transactions)
    # Latest surviving transaction is donor 88's 2025-03-10 payment.
    assert int(feats.loc["88", "recency_days"]) == 0


def test_empty_input_returns_a_typed_empty_frame():
    feats = bloomerang_transactions_to_features([])
    assert feats.empty
    assert list(feats.columns) == list(_FEATURE_DTYPES)


def test_every_row_filtered_out_returns_a_typed_empty_frame():
    rows = [{"Account Number": "1", "Date": "2025-01-01",
             "Amount": "500.00", "Transaction Type": "Pledge"}]
    feats = bloomerang_transactions_to_features(rows)
    assert feats.empty
    assert list(feats.columns) == list(_FEATURE_DTYPES)
    assert feats.index.name == "contact_id"


def test_missing_required_column_names_the_bloomerang_fields():
    rows = [{"Account Number": "1", "Transaction Type": "Donation"}]
    with pytest.raises(KeyError) as excinfo:
        bloomerang_transactions_to_features(rows)
    message = str(excinfo.value)
    assert "Bloomerang" in message
    assert "Date" in message
    # A CiviCRM-worded error here would read as "you used the wrong reader".
    assert "CiviCRM" not in message


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def test_read_single_csv(tmp_path):
    path = tmp_path / "transactions.csv"
    path.write_text(_export(_row("88", "2025-01-10", "1200.00", "Pledge"),
                            _row("88", "2025-02-10", "100.00", "PledgePayment")))
    raw = read_bloomerang_transactions(path)
    assert len(raw) == 2
    # Reading is lossless: the pledge row is still there.
    feats = bloomerang_transactions_to_features(raw)
    assert float(feats.loc["88", "total_gift_amount"]) == 100.0


def test_read_directory_concatenates(tmp_path):
    (tmp_path / "jan.csv").write_text(
        _export(_row("88", "2025-01-10", "100.00", "Donation"))
    )
    (tmp_path / "feb.csv").write_text(
        _export(_row("88", "2025-02-10", "250.00", "Donation"))
    )
    feats = bloomerang_transactions_to_features(read_bloomerang_transactions(tmp_path))
    assert float(feats.loc["88", "total_gift_amount"]) == 350.0


def test_read_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_bloomerang_transactions(tmp_path / "nope.csv")


def test_numeric_looking_account_numbers_stay_text(tmp_path):
    """dtype=str on the read keeps 0088 from becoming 88, and a blank in the
    column from turning the whole thing into floats."""
    path = tmp_path / "transactions.csv"
    path.write_text(_export(_row("0088", "2025-01-10", "100.00", "Donation"),
                            _row("", "2025-01-11", "50.00", "Donation")))
    feats = bloomerang_transactions_to_features(read_bloomerang_transactions(path))
    assert list(feats.index) == ["0088"]
