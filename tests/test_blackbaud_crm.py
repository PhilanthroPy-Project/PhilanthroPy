"""
tests/test_blackbaud_crm.py
Tests for the philanthropy.ingest Blackbaud CRM revenue bridge.

The point of the module is one filter: in Blackbaud CRM a pledge and the
payments against it are separate Revenue records, distinguished only by their
transaction type. A naive sum counts every committed dollar twice. Most of
what follows checks that the commitment rows leave and the Payment rows stay,
in each spelling the export labels and database columns write them in.
"""

import warnings

import pandas as pd
import pytest

from philanthropy.ingest import (
    DEFAULT_INCLUDED_REVENUE_TYPES,
    blackbaud_crm_revenue_to_features,
    read_blackbaud_crm_revenue,
)
from philanthropy.ingest._civicrm import _FEATURE_DTYPES

# Header labels as the acknowledgement export format writes them. The format
# writes a combined "Constituent name" and a single "Email address" column,
# not separate first/last name fields.
_HEADER = (
    "Constituent ID,Constituent name,Revenue Type,Date,Amount,"
    "Application type,Designation name,Email address"
)


def _row(contact, name, date, amount, revenue_type, *, application_type="Donation",
         designation_name="Annual Fund", email_address=""):
    return (
        f"{contact},{name},{date},{amount},{revenue_type},"
        f"{application_type},{designation_name},{email_address}"
    )


def _export(*rows):
    return "\n".join((_HEADER,) + rows) + "\n"


@pytest.fixture
def revenue():
    """One donor with a $1,200 pledge paid in two $100 instalments (each
    payment applied to the Pledge, the shape Blackbaud CRM writes for a
    pledge payment), and one donor with a pledge that has not been paid."""
    return [
        {"Constituent ID": "88", "Constituent name": "Ada Lovelace",
         "Date": "2025-01-10", "Amount": "1200.00", "Revenue Type": "Pledge",
         "Application type": "Pledge", "Designation name": "Annual Fund",
         "Email address": "ada@amc.edu"},
        {"Constituent ID": "88", "Constituent name": "Ada Lovelace",
         "Date": "2025-02-10", "Amount": "100.00", "Revenue Type": "Payment",
         "Application type": "Pledge", "Designation name": "Annual Fund",
         "Email address": "ada@amc.edu"},
        {"Constituent ID": "88", "Constituent name": "Ada Lovelace",
         "Date": "2025-03-10", "Amount": "100.00", "Revenue Type": "Payment",
         "Application type": "Pledge", "Designation name": "Annual Fund",
         "Email address": "ada@amc.edu"},
        {"Constituent ID": "91", "Constituent name": "Grace Hopper",
         "Date": "2025-02-01", "Amount": "600.00", "Revenue Type": "Pledge",
         "Application type": "Pledge", "Designation name": "Research",
         "Email address": "grace@amc.edu"},
    ]


# --------------------------------------------------------------------------- #
# The commitment-versus-payment filter
# --------------------------------------------------------------------------- #
def test_pledge_is_excluded_and_its_payments_are_not(revenue):
    feats = blackbaud_crm_revenue_to_features(revenue)
    # 1200 (the promise) + 100 + 100 would be 1400; only the payments count.
    assert float(feats.loc["88", "total_gift_amount"]) == 200.0
    assert int(feats.loc["88", "gift_count"]) == 2


def test_donor_with_only_a_pledge_is_absent(revenue):
    feats = blackbaud_crm_revenue_to_features(revenue)
    assert "91" not in feats.index


@pytest.mark.parametrize(
    "revenue_type",
    ["Pledge", "Recurring gift", "Matching gift claim", "Planned gift",
     "Order", "Grant award", "Auction donation", "Donor challenge claim",
     "Pending Gift", "Deposit", "Adjustment deposit", "Adjustment payment",
     "Adjustment transfer in", "Adjustment transfer out",
     "Membership installment plan", "Write off", "Sold stock",
     "Sold property", "Refund", "Deposit Correction Short",
     "Deposit Correction Over", "Payout", "Sold Gift In Kind",
     "Computer check", "Deleted Revenue", "Invoice", "Credit memo",
     "Journal entry batch", "Charge", "AP Payment", "Credit", "Reversal",
     "System Transaction"]
)
def test_every_non_payment_type_is_dropped(revenue_type):
    """The allowlist keeps only Payment; every other documented type leaves."""
    rows = [{"Constituent ID": "1", "Date": "2025-01-01",
             "Amount": "500.00", "Revenue Type": revenue_type},
            {"Constituent ID": "1", "Date": "2025-01-02",
             "Amount": "10.00", "Revenue Type": "Payment"}]
    feats = blackbaud_crm_revenue_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


@pytest.mark.parametrize(
    "revenue_type",
    ["Payment", "payment", "PAYMENT", "  Payment  ", "Payment."]
)
def test_inclusion_matching_ignores_case_spacing_and_punctuation(revenue_type):
    rows = [{"Constituent ID": "1", "Date": "2025-01-01",
             "Amount": "10.00", "Revenue Type": revenue_type}]
    feats = blackbaud_crm_revenue_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


def test_payment_applied_to_a_pledge_is_kept():
    """The filter operates on the transaction type, not the application type: a
    Payment applied to a Pledge is money received and must count."""
    rows = [{"Constituent ID": "1", "Date": "2025-01-01",
             "Amount": "100.00", "Revenue Type": "Payment",
             "Application type": "Pledge"}]
    feats = blackbaud_crm_revenue_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 100.0


def test_pledge_applied_to_donation_is_still_dropped():
    """A Pledge with an Application type of Donation is still the promise, not the
    money: the application type does not override the transaction type."""
    rows = [{"Constituent ID": "1", "Date": "2025-01-01",
             "Amount": "500.00", "Revenue Type": "Pledge",
             "Application type": "Donation"}]
    feats = blackbaud_crm_revenue_to_features(rows)
    assert "1" not in feats.index


def test_blank_gift_type_is_kept():
    """An unlabelled row cannot be shown to be a commitment, so it stays."""
    rows = [{"Constituent ID": "1", "Date": "2025-01-01",
             "Amount": "10.00", "Revenue Type": ""},
            {"Constituent ID": "1", "Date": "2025-01-02",
             "Amount": "5.00", "Revenue Type": None}]
    feats = blackbaud_crm_revenue_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 15.0
    assert int(feats.loc["1", "gift_count"]) == 2


def test_custom_include_set_replaces_the_default(revenue):
    feats = blackbaud_crm_revenue_to_features(
        revenue, include_revenue_types=("Pledge",)
    )
    # Payment is no longer included; the two Pledge rows are.
    assert float(feats.loc["88", "total_gift_amount"]) == 1200.0
    assert float(feats.loc["91", "total_gift_amount"]) == 600.0


def test_include_none_counts_every_row(revenue):
    feats = blackbaud_crm_revenue_to_features(revenue, include_revenue_types=None)
    assert float(feats.loc["88", "total_gift_amount"]) == 1400.0
    assert int(feats.loc["88", "gift_count"]) == 3


def test_empty_include_sequence_counts_every_row(revenue):
    feats = blackbaud_crm_revenue_to_features(revenue, include_revenue_types=())
    assert float(feats.loc["88", "total_gift_amount"]) == 1400.0


def test_missing_revenue_type_column_warns():
    rows = [{"Constituent ID": "1", "Date": "2025-01-01", "Amount": "10.00"}]
    with pytest.warns(UserWarning, match="no revenue type column"):
        feats = blackbaud_crm_revenue_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


def test_include_none_does_not_warn_about_a_missing_revenue_type_column():
    rows = [{"Constituent ID": "1", "Date": "2025-01-01", "Amount": "10.00"}]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        blackbaud_crm_revenue_to_features(rows, include_revenue_types=None)


def test_default_included_set_is_documented_and_non_empty():
    assert "Payment" in DEFAULT_INCLUDED_REVENUE_TYPES


def test_every_row_filtered_out_returns_a_typed_empty_frame():
    rows = [{"Constituent ID": "1", "Date": "2025-01-01",
             "Amount": "500.00", "Revenue Type": "Pledge"}]
    feats = blackbaud_crm_revenue_to_features(rows)
    assert feats.empty
    assert list(feats.columns) == list(_FEATURE_DTYPES)
    assert feats.index.name == "contact_id"


# --------------------------------------------------------------------------- #
# Header dialects
# --------------------------------------------------------------------------- #
def test_database_column_names_are_accepted():
    """A raw database export spells them CONSTITUENTID / TYPECODE / DATE /
    TRANSACTIONAMOUNT / DESIGNATIONID. Only the string label columns
    (TYPE, APPLICATION) are supported for the filter; a numeric TYPECODE
    is not translated."""
    rows = [{"CONSTITUENTID": "7", "DATE": "2025-05-01",
             "TRANSACTIONAMOUNT": "300.00", "TYPE": "Payment"},
            {"CONSTITUENTID": "7", "DATE": "2025-05-02",
             "TRANSACTIONAMOUNT": "999.00", "TYPE": "Pledge"}]
    feats = blackbaud_crm_revenue_to_features(rows)
    assert float(feats.loc["7", "total_gift_amount"]) == 300.0


def test_short_export_labels_are_accepted():
    """The acknowledgement format abbreviates to "Revenue Type" and
    "Designation name"; both are the same fields under the hood."""
    rows = [{"Constituent ID": "7", "Date": "2025-05-01",
             "Amount": "300.00", "Revenue Type": "Payment"}]
    feats = blackbaud_crm_revenue_to_features(rows)
    assert float(feats.loc["7", "total_gift_amount"]) == 300.0


def test_lookup_id_is_accepted_as_the_donor_key():
    rows = [{"Constituent ID": "8-000123", "Date": "2025-05-01",
             "Amount": "300.00", "Revenue Type": "Payment"}]
    feats = blackbaud_crm_revenue_to_features(rows)
    assert float(feats.loc["8-000123", "total_gift_amount"]) == 300.0


def test_export_labels_and_database_columns_agree(revenue):
    """The export labels and the raw column names must produce the same
    feature frame. Identity fields are excluded from the comparison because
    the export writes a combined name and the database has no single email
    column on the revenue side."""
    db = [
        {"CONSTITUENTID": r["Constituent ID"], "DATE": r["Date"],
         "TRANSACTIONAMOUNT": r["Amount"], "TYPE": r["Revenue Type"],
         "APPLICATION": r["Application type"],
         "DESIGNATION_NAME": r["Designation name"]}
        for r in revenue
    ]
    labels_only = [
        {"Constituent ID": r["Constituent ID"], "Date": r["Date"],
         "Amount": r["Amount"], "Revenue Type": r["Revenue Type"],
         "Application type": r["Application type"],
         "Designation name": r["Designation name"]}
        for r in revenue
    ]
    pd.testing.assert_frame_equal(
        blackbaud_crm_revenue_to_features(labels_only),
        blackbaud_crm_revenue_to_features(db),
    )


def test_currency_formatted_amounts_parse():
    rows = [{"Constituent ID": "1", "Date": "2025-01-01",
             "Amount": "$1,250.00", "Revenue Type": "Payment"}]
    feats = blackbaud_crm_revenue_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 1250.0


# --------------------------------------------------------------------------- #
# Schema, identity and recency
# --------------------------------------------------------------------------- #
def test_schema_and_dtypes(revenue):
    feats = blackbaud_crm_revenue_to_features(revenue)
    assert list(feats.columns) == list(_FEATURE_DTYPES)
    assert feats.index.name == "contact_id"


def test_designation_feeds_distinct_financial_types(revenue):
    feats = blackbaud_crm_revenue_to_features(revenue, include_revenue_types=None)
    # Donor 88's rows all sit on Annual Fund; donor 91's on Research.
    assert int(feats.loc["88", "distinct_financial_types"]) == 1
    assert int(feats.loc["91", "distinct_financial_types"]) == 1


def test_carries_constituent_email(revenue):
    """The acknowledgement export writes a single 'Email address' column.
    First and last name are not separate fields in this format, so they are
    not asserted here."""
    feats = blackbaud_crm_revenue_to_features(revenue)
    assert feats.loc["88", "constituent_email"] == "ada@amc.edu"


def test_reference_date_shifts_recency(revenue):
    feats = blackbaud_crm_revenue_to_features(revenue, reference_date="2025-12-31")
    assert int(feats.loc["88", "recency_days"]) == 296


def test_recency_anchors_on_the_batch_by_default(revenue):
    feats = blackbaud_crm_revenue_to_features(revenue)
    # Latest surviving row is donor 88's 2025-03-10 Payment.
    assert int(feats.loc["88", "recency_days"]) == 0


def test_empty_input_returns_a_typed_empty_frame():
    feats = blackbaud_crm_revenue_to_features([])
    assert feats.empty
    assert list(feats.columns) == list(_FEATURE_DTYPES)


def test_missing_required_column_names_the_blackbaud_fields():
    rows = [{"Constituent ID": "1", "Revenue Type": "Payment"}]
    with pytest.raises(KeyError) as excinfo:
        blackbaud_crm_revenue_to_features(rows)
    message = str(excinfo.value)
    assert "Blackbaud CRM" in message
    assert "Date" in message
    # A CiviCRM-worded error here would read as "you used the wrong reader".
    assert "CiviCRM" not in message


def test_name_only_export_raises_rather_than_merging_donors():
    """A name is not an ID: two donors named "Ada Lovelace" exist. An export
    with only a name column has no contact_id and must raise, not silently
    key on the name."""
    rows = [{"Constituent name": "Ada Lovelace", "Date": "2025-01-01",
             "Amount": "100.00", "Revenue Type": "Payment"}]
    with pytest.raises(KeyError) as excinfo:
        blackbaud_crm_revenue_to_features(rows)
    assert "Constituent ID" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def test_read_single_csv(tmp_path):
    path = tmp_path / "revenue.csv"
    path.write_text(_export(
        _row("88", "Ada Lovelace", "Pledge", "2025-01-10", "1200.00" ),
        _row("88", "Ada Lovelace", "Payment", "2025-02-10", "100.00")
    ))
    raw = read_blackbaud_crm_revenue(path)
    assert len(raw) == 2
    # Reading is lossless: the pledge row is still there.
    feats = blackbaud_crm_revenue_to_features(raw)
    assert float(feats.loc["88", "total_gift_amount"]) == 100.0


def test_read_directory_concatenates(tmp_path):
    (tmp_path / "jan.csv").write_text(
        _export(_row("88", "Ada Lovelace", "Payment", "2025-01-10", "100.00"))
    )
    (tmp_path / "feb.csv").write_text(
        _export(_row("88", "Ada Lovelace", "Payment", "2025-02-10", "250.00"))
    )
    feats = blackbaud_crm_revenue_to_features(read_blackbaud_crm_revenue(tmp_path))
    assert float(feats.loc["88", "total_gift_amount"]) == 350.0


def test_read_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_blackbaud_crm_revenue(tmp_path / "nope.csv")


def test_numeric_looking_constituent_ids_stay_text(tmp_path):
    """dtype=str on the read keeps 0088 from becoming 88, and a blank in the
    column from turning the whole thing into floats."""
    path = tmp_path / "revenue.csv"
    path.write_text(_export(
        _row("0088", "Ada Lovelace", "Payment", "2025-01-10", "100.00"),
        _row("", "Unknown", "Payment", "2025-01-11", "50.00")
    ))
    feats = blackbaud_crm_revenue_to_features(read_blackbaud_crm_revenue(path))
    assert list(feats.index) == ["0088"]