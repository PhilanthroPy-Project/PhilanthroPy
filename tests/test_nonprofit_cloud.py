"""
tests/test_nonprofit_cloud.py
Tests for the philanthropy.ingest Salesforce Nonprofit Cloud GiftTransaction
bridge.

The point of the module is an allowlist on ``Status``: a GiftTransaction that a
GiftCommitment generated automatically starts at ``Unpaid`` and only becomes
``Paid`` when the money actually arrives. A ``Canceled``, ``Failed``,
``Fully Refunded``, ``Written Off`` or ``Pending`` transaction was never received
either, and an org can add its own custom non-paid statuses. Only rows whose
status is a documented received status (``Paid``) are counted, so anything else
is dropped by default. Most of what follows checks that Paid rows stay, everything
else leaves, and that the export's header spellings are all accepted.
"""

import warnings

import pandas as pd
import pytest

from philanthropy.ingest import (
    DEFAULT_INCLUDED_GIFT_TRANSACTION_STATUSES,
    nonprofit_cloud_gifts_to_features,
    read_nonprofit_cloud_gifts,
)
from philanthropy.ingest._civicrm import _FEATURE_DTYPES


# Header labels as a Salesforce report export writes them.
_HEADER = "Donor ID,Transaction Date,Current Amount,Status,Gift Type,Email,First Name,Last Name"


def _row(donor, date, amount, status, *, gift_type="Individual",
         email="", first="", last=""):
    return f"{donor},{date},{amount},{status},{gift_type},{email},{first},{last}"


def _export(*rows):
    return "\n".join((_HEADER,) + rows) + "\n"


@pytest.fixture
def gifts():
    """One donor with a monthly GiftCommitment's generated schedule: two
    Paid receipts and one Unpaid future instalment, all $100; one donor with
    only a still-Unpaid instalment."""
    return [
        {"Donor ID": "88", "Transaction Date": "2025-01-10",
         "Current Amount": "100.00", "Status": "Paid",
         "Gift Type": "Individual",
         "Email": "ada@amc.edu", "First Name": "Ada", "Last Name": "Lovelace"},
        {"Donor ID": "88", "Transaction Date": "2025-02-10",
         "Current Amount": "100.00", "Status": "Paid",
         "Gift Type": "Organizational",
         "Email": "ada@amc.edu", "First Name": "Ada", "Last Name": "Lovelace"},
        {"Donor ID": "88", "Transaction Date": "2025-03-10",
         "Current Amount": "100.00", "Status": "Unpaid",
         "Gift Type": "Organizational",
         "Email": "ada@amc.edu", "First Name": "Ada", "Last Name": "Lovelace"},
        {"Donor ID": "91", "Transaction Date": "2025-03-01",
         "Current Amount": "50.00", "Status": "Unpaid",
         "Gift Type": "Individual",
         "Email": "grace@amc.edu", "First Name": "Grace", "Last Name": "Hopper"},
    ]


# --------------------------------------------------------------------------- #
# The Paid allowlist
# --------------------------------------------------------------------------- #
def test_unpaid_instalment_is_excluded_and_paid_is_not(gifts):
    feats = nonprofit_cloud_gifts_to_features(gifts)
    # 100 + 100 + 100 would be 300; only the 2 Paid rows count
    assert float(feats.loc["88", "total_gift_amount"]) == 200.0
    assert int(feats.loc["88", "gift_count"]) == 2


def test_donor_with_only_unpaid_rows_is_absent(gifts):
    feats = nonprofit_cloud_gifts_to_features(gifts)
    assert "91" not in feats.index


def test_every_row_filtered_out_returns_a_typed_empty_frame():
    rows = [{"Donor ID": "1", "Transaction Date": "2025-01-01",
             "Current Amount": "500.00", "Status": "Unpaid"}]
    feats = nonprofit_cloud_gifts_to_features(rows)
    assert feats.empty
    assert list(feats.columns) == list(_FEATURE_DTYPES)
    assert feats.index.name == "contact_id"


def test_every_default_included_status_is_kept():
    rows = [{"Donor ID": "1", "Transaction Date": "2025-01-01",
             "Current Amount": "500.00", "Status": "Unpaid"},
            {"Donor ID": "1", "Transaction Date": "2025-01-02",
             "Current Amount": "10.00", "Status": "Paid"}]
    feats = nonprofit_cloud_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


@pytest.mark.parametrize("status", [
    "Unpaid", "Pending", "Failed", "Canceled", "Fully Refunded", "Written Off"
])
def test_non_paid_statuses_are_excluded(status):
    """Everything except a documented received status is dropped by default:
    an Unpaid, Pending, Failed, Cancelled, Fully Refunded, or Written Off
    installment was never received or was unwound."""
    rows = [{"Donor ID": "1", "Transaction Date": "2025-01-01",
             "Current Amount": "10.00", "Status": status}]
    feats = nonprofit_cloud_gifts_to_features(rows)
    assert "1" not in feats.index


def test_custom_org_specific_status_is_excluded():
    """The allowlist offers a safety property in which an org-defined status
    this module has never heard of is dropped rather than silently summed,
    because an unknown status is not proof the money arrived."""
    rows = [{"Donor ID": "1", "Transaction Date": "2025-01-01",
             "Current Amount": "10.00", "Status": "In Review"}]
    feats = nonprofit_cloud_gifts_to_features(rows)
    assert "1" not in feats.index


@pytest.mark.parametrize("status", ["PAID", "paid", "  Paid  "])
def test_inclusion_matching_ignores_case_and_spacing(status):
    rows = [{"Donor ID": "1", "Transaction Date": "2025-01-01",
             "Current Amount": "500.00", "Status": "Unpaid"},
            {"Donor ID": "1", "Transaction Date": "2025-01-02",
             "Current Amount": "10.00", "Status": status}]
    feats = nonprofit_cloud_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


def test_a_status_only_containing_paid_as_a_substring_is_not_included():
    """'Partially Paid' must not be swept into the 'Paid' default:
    the matching key is compared for equality, not membership."""
    rows = [{"Donor ID": "1", "Transaction Date": "2025-01-01",
             "Current Amount": "25.00", "Status": "Partially Paid"}]
    feats = nonprofit_cloud_gifts_to_features(rows)
    assert "1" not in feats.index


def test_blank_status_is_kept():
    """An unlabelled row is kept. The export did not name a status, so it
    cannot be shown to be an expected-but-unpaid commitment."""
    rows = [{"Donor ID": "1", "Transaction Date": "2025-01-01",
             "Current Amount": "10.00", "Status": ""},
            {"Donor ID": "1", "Transaction Date": "2025-01-02",
             "Current Amount": "5.00", "Status": None}]
    feats = nonprofit_cloud_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 15.0
    assert int(feats.loc["1", "gift_count"]) == 2


def test_custom_include_set_replaces_the_default(gifts):
    feats = nonprofit_cloud_gifts_to_features(gifts, include_statuses=("Unpaid",))
    # Paid is no longer included; the 2 Unpaid rows are.
    assert float(feats.loc["88", "total_gift_amount"]) == 100.0
    assert float(feats.loc["91", "total_gift_amount"]) == 50.0


def test_include_none_counts_every_row(gifts):
    feats = nonprofit_cloud_gifts_to_features(gifts, include_statuses=None)
    assert float(feats.loc["88", "total_gift_amount"]) == 300.0
    assert int(feats.loc["88", "gift_count"]) == 3


def test_empty_include_sequence_counts_every_row(gifts):
    feats = nonprofit_cloud_gifts_to_features(gifts, include_statuses=())
    assert float(feats.loc["88", "total_gift_amount"]) == 300.0


def test_missing_status_column_warns():
    rows = [{"Donor ID": "1", "Transaction Date": "2025-01-01", "Current Amount": "10.00"}]
    with pytest.warns(UserWarning, match="no Status column"):
        feats = nonprofit_cloud_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


def test_include_none_does_not_warn_about_a_missing_status_column():
    rows = [{"Donor ID": "1", "Transaction Date": "2025-01-01", "Current Amount": "10.00"}]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nonprofit_cloud_gifts_to_features(rows, include_statuses=None)


def test_default_included_set_is_documented_and_non_empty():
    assert "Paid" in DEFAULT_INCLUDED_GIFT_TRANSACTION_STATUSES


# --------------------------------------------------------------------------- #
# Header dialects
# --------------------------------------------------------------------------- #
def test_raw_api_field_names_are_accepted():
    """The GiftTransaction API returns DonorId / TransactionDate /
    CurrentAmount / Status / GiftType."""
    rows = [{"DonorId": "7", "TransactionDate": "2025-05-01",
             "CurrentAmount": "300.00", "Status": "Paid"},
            {"DonorId": "7", "TransactionDate": "2025-05-02",
             "CurrentAmount": "999.00", "Status": "Unpaid"}]
    feats = nonprofit_cloud_gifts_to_features(rows)
    assert float(feats.loc["7", "total_gift_amount"]) == 300.0


def test_donor_id_is_accepted_as_the_donor_key():
    rows = [{"Donor ID": "7", "Transaction Date": "2025-05-01",
             "Current Amount": "300.00", "Status": "Paid"}]
    feats = nonprofit_cloud_gifts_to_features(rows)
    assert float(feats.loc["7", "total_gift_amount"]) == 300.0


def test_raw_api_donor_id_is_accepted_as_the_donor_key():
    rows = [{"DonorId": "7", "TransactionDate": "2025-05-01",
             "CurrentAmount": "300.00", "Status": "Paid"}]
    feats = nonprofit_cloud_gifts_to_features(rows)
    assert float(feats.loc["7", "total_gift_amount"]) == 300.0


def test_donor_id_wins_over_contact_id_when_both_are_present():
    """Nonprofit Cloud keys a GiftTransaction to an Account (the
    household/organization) via DonorId and to a Contact (the
    individual associated with the transaction) via ContactId.
    DonorId maps to the Account DMO and ContactId to the Individual
    DMO. For the donor-level frame this module produces, the Account
    is the intended key regardless of which column comes first in
    the export."""
    rows_donor_first = [
        {"Donor ID": "7", "Contact ID": "999", "Transaction Date": "2025-05-01",
         "Current Amount": "300.00", "Status": "Paid"}
    ]
    rows_contact_first = [
        {"Contact ID": "999", "Donor ID": "7", "Transaction Date": "2025-05-01",
         "Current Amount": "300.00", "Status": "Paid"}
    ]
    for rows in (rows_donor_first, rows_contact_first):
        feats = nonprofit_cloud_gifts_to_features(rows)
        assert "7" in feats.index
        assert "999" not in feats.index


def test_export_labels_and_api_names_agree(gifts):
    api = [
        {"DonorId": g["Donor ID"], "TransactionDate": g["Transaction Date"],
         "CurrentAmount": g["Current Amount"], "Status": g["Status"],
         "GiftType": g["Gift Type"], "email": g["Email"],
         "first_name": g["First Name"], "last_name": g["Last Name"]}
        for g in gifts
    ]
    pd.testing.assert_frame_equal(
        nonprofit_cloud_gifts_to_features(gifts),
        nonprofit_cloud_gifts_to_features(api),
    )


def test_currency_formatted_amounts_parse():
    rows = [{"Donor ID": "1", "Transaction Date": "2025-01-01",
             "Current Amount": "$1,250.00", "Status": "Paid"}]
    feats = nonprofit_cloud_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 1250.0


def test_name_only_export_raises_error_instead_of_merging_donors():
    """A name is not an ID: two donors sharing a name would be merged
    into one row, and one donor under two spellings would be split. An
    export with only a name column has no ``contact_id`` after
    normalisation and must raise an error, instead of silently keying on the name."""
    rows = [{"Donor Name": "Ada Lovelace", "Transaction Date": "2025-01-01",
             "Current Amount": "$1,250.00", "Status": "Paid"}]
    with pytest.raises(KeyError) as excinfo:
        nonprofit_cloud_gifts_to_features(rows)
    assert "Donor ID" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Schema, identity and recency
# --------------------------------------------------------------------------- #
def test_schema_and_dtypes(gifts):
    feats = nonprofit_cloud_gifts_to_features(gifts)
    assert list(feats.columns) == list(_FEATURE_DTYPES)
    assert feats.index.name == "contact_id"


def test_carries_identity_fields(gifts):
    feats = nonprofit_cloud_gifts_to_features(gifts)
    assert feats.loc["88", "constituent_email"] == "ada@amc.edu"
    assert feats.loc["88", "first_name"] == "Ada"
    assert feats.loc["88", "last_name"] == "Lovelace"


def test_reference_date_shifts_recency(gifts):
    feats = nonprofit_cloud_gifts_to_features(gifts, reference_date="2025-12-31")
    assert int(feats.loc["88", "recency_days"]) == 324


def test_recency_anchors_on_the_batch_by_default(gifts):
    feats = nonprofit_cloud_gifts_to_features(gifts)
    # Latest surviving row is donor 88's 2025-02-10 Paid instalment.
    assert int(feats.loc["88", "recency_days"]) == 0


def test_empty_input_returns_a_typed_empty_frame():
    feats = nonprofit_cloud_gifts_to_features([])
    assert feats.empty
    assert list(feats.columns) == list(_FEATURE_DTYPES)


def test_missing_required_column_names_the_nonprofit_cloud_fields():
    rows = [{"Donor ID": "1", "Status": "Paid"}]
    with pytest.raises(KeyError) as excinfo:
        nonprofit_cloud_gifts_to_features(rows)
    message = str(excinfo.value)
    assert "Nonprofit Cloud" in message
    assert "Transaction Date" in message
    # A CiviCRM-worded error here would read as "you used the wrong reader".
    assert "CiviCRM" not in message


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def test_read_single_csv(tmp_path):
    path = tmp_path / "gifts.csv"
    path.write_text(_export(_row("88", "2025-01-10", "100.00", "Unpaid"),
                            _row("88", "2025-02-10", "100.00", "Paid")))
    raw = read_nonprofit_cloud_gifts(path)
    assert len(raw) == 2
    # Reading is lossless: the Unpaid row is still there.
    feats = nonprofit_cloud_gifts_to_features(raw)
    assert float(feats.loc["88", "total_gift_amount"]) == 100.0


def test_read_directory_concatenates(tmp_path):
    (tmp_path / "jan.csv").write_text(
        _export(_row("88", "2025-01-10", "100.00", "Paid"))
    )
    (tmp_path / "feb.csv").write_text(
        _export(_row("88", "2025-02-10", "250.00", "Paid"))
    )
    feats = nonprofit_cloud_gifts_to_features(read_nonprofit_cloud_gifts(tmp_path))
    assert float(feats.loc["88", "total_gift_amount"]) == 350.0


def test_read_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_nonprofit_cloud_gifts(tmp_path / "nope.csv")


def test_numeric_looking_contact_ids_stay_text(tmp_path):
    """dtype=str on the read keeps 0088 from becoming 88, and a blank in the
    column from turning the whole thing into floats."""
    path = tmp_path / "gifts.csv"
    path.write_text(_export(_row("0088", "2025-01-10", "100.00", "Paid"),
                            _row("", "2025-01-11", "50.00", "Paid")))
    feats = nonprofit_cloud_gifts_to_features(read_nonprofit_cloud_gifts(path))
    assert list(feats.index) == ["0088"]