"""
tests/test_raisers_edge.py
Tests for the philanthropy.ingest Blackbaud Raiser's Edge gift bridge.

The point of the module is one filter: in Raiser's Edge a pledge and the
payments against it are separate gift records, so a naive sum counts every
committed dollar twice. Most of what follows checks that the commitment rows
leave and the payment rows stay, in each spelling the two Raiser's Edge
generations write them in.
"""

import warnings

import pandas as pd
import pytest

from philanthropy.ingest import (
    DEFAULT_EXCLUDED_GIFT_TYPES,
    raisers_edge_gifts_to_features,
    read_raisers_edge_gifts,
)
from philanthropy.ingest._civicrm import _FEATURE_DTYPES

# Header labels as the Raiser's Edge desktop Export module writes them.
_HEADER = (
    "Constituent ID,Gift Date,Gift Amount,Gift Type,Fund,Email,"
    "First Name,Last Name"
)


def _row(contact, date, amount, gift_type, *, fund="Annual Fund",
         email="", first="", last=""):
    return f"{contact},{date},{amount},{gift_type},{fund},{email},{first},{last}"


def _export(*rows):
    return "\n".join((_HEADER,) + rows) + "\n"


@pytest.fixture
def gifts():
    """One donor with a $1,200 pledge paid in two $100 instalments, and one
    with a recurring gift template plus one payment against it."""
    return [
        {"Constituent ID": "88", "Gift Date": "2025-01-10",
         "Gift Amount": "1200.00", "Gift Type": "Pledge", "Fund": "Annual Fund",
         "Email": "ada@amc.edu", "First Name": "Ada", "Last Name": "Lovelace"},
        {"Constituent ID": "88", "Gift Date": "2025-02-10",
         "Gift Amount": "100.00", "Gift Type": "Pay-Cash", "Fund": "Annual Fund",
         "Email": "ada@amc.edu", "First Name": "Ada", "Last Name": "Lovelace"},
        {"Constituent ID": "88", "Gift Date": "2025-03-10",
         "Gift Amount": "100.00", "Gift Type": "Pay-Cash", "Fund": "Annual Fund",
         "Email": "ada@amc.edu", "First Name": "Ada", "Last Name": "Lovelace"},
        {"Constituent ID": "91", "Gift Date": "2025-02-01",
         "Gift Amount": "50.00", "Gift Type": "Recurring Gift", "Fund": "Research",
         "Email": "grace@amc.edu", "First Name": "Grace", "Last Name": "Hopper"},
        {"Constituent ID": "91", "Gift Date": "2025-03-01",
         "Gift Amount": "50.00", "Gift Type": "Recurring Gift Pay-Cash",
         "Fund": "Capital", "Email": "grace@amc.edu",
         "First Name": "Grace", "Last Name": "Hopper"},
    ]


# --------------------------------------------------------------------------- #
# The commitment-versus-payment filter
# --------------------------------------------------------------------------- #
def test_pledge_is_excluded_and_its_payments_are_not(gifts):
    feats = raisers_edge_gifts_to_features(gifts)
    # 1200 (the promise) + 100 + 100 would be 1400; only the payments count.
    assert float(feats.loc["88", "total_gift_amount"]) == 200.0
    assert int(feats.loc["88", "gift_count"]) == 2


def test_recurring_gift_template_is_excluded_and_its_payment_is_not(gifts):
    feats = raisers_edge_gifts_to_features(gifts)
    assert float(feats.loc["91", "total_gift_amount"]) == 50.0
    assert int(feats.loc["91", "gift_count"]) == 1


@pytest.mark.parametrize(
    "gift_type",
    ["Pledge", "Matching Gift Pledge", "MG Pledge", "Recurring Gift",
     "Amendment", "Adjustment", "General Ledger Reversal", "Write Off",
     "Pledge Write Off", "Matching Gift Write Off", "MG Write Off"],
)
def test_every_default_excluded_type_is_dropped(gift_type):
    rows = [{"Constituent ID": "1", "Gift Date": "2025-01-01",
             "Gift Amount": "500.00", "Gift Type": gift_type},
            {"Constituent ID": "1", "Gift Date": "2025-01-02",
             "Gift Amount": "10.00", "Gift Type": "Cash"}]
    feats = raisers_edge_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


@pytest.mark.parametrize(
    "gift_type",
    ["Cash", "Pay-Cash", "MG Pay-Cash", "Recurring Gift Pay-Cash",
     "Gift-in-Kind", "Stock/Property", "Other", "Donation", "PledgePayment",
     "MatchingGiftPayment", "RecurringGiftPayment", "SoldStock", "PlannedGift"],
)
def test_payment_and_outright_types_are_kept(gift_type):
    rows = [{"Constituent ID": "1", "Gift Date": "2025-01-01",
             "Gift Amount": "10.00", "Gift Type": gift_type}]
    feats = raisers_edge_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


@pytest.mark.parametrize(
    "gift_type",
    ["MatchingGiftPledge", "RecurringGift", "PledgeWriteOff",
     "MatchingGiftWriteOff", "GeneralLedgerReversal", "recurring_gift",
     "RECURRING GIFT", "  Pledge  "],
)
def test_exclusion_matching_ignores_case_spacing_and_punctuation(gift_type):
    """The RE NXT SKY API writes RecurringGift where the desktop writes
    Recurring Gift; the default set names one spelling and matches both."""
    rows = [{"Constituent ID": "1", "Gift Date": "2025-01-01",
             "Gift Amount": "500.00", "Gift Type": gift_type},
            {"Constituent ID": "1", "Gift Date": "2025-01-02",
             "Gift Amount": "10.00", "Gift Type": "Cash"}]
    feats = raisers_edge_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


def test_a_pay_type_is_not_matched_by_its_commitment_prefix():
    """'Recurring Gift Pay-Cash' must not be swept up by 'Recurring Gift'."""
    rows = [{"Constituent ID": "1", "Gift Date": "2025-01-01",
             "Gift Amount": "25.00", "Gift Type": "Recurring Gift Pay-Cash"}]
    feats = raisers_edge_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 25.0


def test_blank_gift_type_is_kept():
    """An unlabelled row cannot be shown to be a commitment, so it stays."""
    rows = [{"Constituent ID": "1", "Gift Date": "2025-01-01",
             "Gift Amount": "10.00", "Gift Type": ""},
            {"Constituent ID": "1", "Gift Date": "2025-01-02",
             "Gift Amount": "5.00", "Gift Type": None}]
    feats = raisers_edge_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 15.0
    assert int(feats.loc["1", "gift_count"]) == 2


def test_custom_exclude_set_replaces_the_default(gifts):
    feats = raisers_edge_gifts_to_features(gifts, exclude_gift_types=("Pay-Cash",))
    # Pledge is no longer excluded; the two Pay-Cash payments are.
    assert float(feats.loc["88", "total_gift_amount"]) == 1200.0


def test_exclude_none_counts_every_row(gifts):
    feats = raisers_edge_gifts_to_features(gifts, exclude_gift_types=None)
    assert float(feats.loc["88", "total_gift_amount"]) == 1400.0
    assert int(feats.loc["88", "gift_count"]) == 3


def test_empty_exclude_sequence_counts_every_row(gifts):
    feats = raisers_edge_gifts_to_features(gifts, exclude_gift_types=())
    assert float(feats.loc["88", "total_gift_amount"]) == 1400.0


def test_missing_gift_type_column_warns():
    rows = [{"Constituent ID": "1", "Gift Date": "2025-01-01",
             "Gift Amount": "10.00"}]
    with pytest.warns(UserWarning, match="no gift type column"):
        feats = raisers_edge_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


def test_exclude_none_does_not_warn_about_a_missing_gift_type_column():
    rows = [{"Constituent ID": "1", "Gift Date": "2025-01-01",
             "Gift Amount": "10.00"}]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        raisers_edge_gifts_to_features(rows, exclude_gift_types=None)


def test_default_excluded_set_is_documented_and_non_empty():
    assert "Pledge" in DEFAULT_EXCLUDED_GIFT_TYPES
    assert "Recurring Gift" in DEFAULT_EXCLUDED_GIFT_TYPES


def test_the_abbreviated_matching_gift_commitment_is_excluded_too():
    """'MG Pledge' is not 'Matching Gift Pledge' with the punctuation removed,
    so collapsing case and punctuation does not make one match the other. The
    desktop writes the abbreviated form, and it is the commitment half of the
    'MG Pay-Cash' the payment set keeps."""
    rows = [{"Constituent ID": "1", "Gift Date": "2025-01-01",
             "Gift Amount": "500.00", "Gift Type": "MG Pledge"},
            {"Constituent ID": "1", "Gift Date": "2025-01-02",
             "Gift Amount": "10.00", "Gift Type": "MG Pay-Cash"}]
    feats = raisers_edge_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


# --------------------------------------------------------------------------- #
# Header dialects
# --------------------------------------------------------------------------- #
def test_sky_api_field_names_are_accepted():
    """RE NXT's SKY API returns constituent_id / date / amount / type."""
    rows = [{"constituent_id": "7", "date": "2025-05-01", "amount": "300.00",
             "type": "Donation"},
            {"constituent_id": "7", "date": "2025-05-02", "amount": "999.00",
             "type": "Pledge"}]
    feats = raisers_edge_gifts_to_features(rows)
    assert float(feats.loc["7", "total_gift_amount"]) == 300.0


def test_re7_database_column_names_are_accepted():
    """The RE7 tables spell them CONSTIT_ID / DTE / AMOUNT / TYPE."""
    rows = [{"CONSTIT_ID": "7", "DTE": "2025-05-01", "AMOUNT": "300.00",
             "TYPE": "Cash"}]
    feats = raisers_edge_gifts_to_features(rows)
    assert float(feats.loc["7", "total_gift_amount"]) == 300.0


def test_export_labels_and_sky_names_agree(gifts):
    sky = [
        {"constituent_id": g["Constituent ID"], "date": g["Gift Date"],
         "amount": g["Gift Amount"], "type": g["Gift Type"],
         "fund": g["Fund"], "email": g["Email"],
         "first_name": g["First Name"], "last_name": g["Last Name"]}
        for g in gifts
    ]
    pd.testing.assert_frame_equal(
        raisers_edge_gifts_to_features(gifts),
        raisers_edge_gifts_to_features(sky),
    )


@pytest.mark.parametrize("amount_label", ["Gift Amount", "Amount", "Pledge amt", "Value"])
def test_every_amount_label_the_guide_names_is_accepted(amount_label):
    """The Gift Records Guide renames the amount field per gift type: Amount on
    a Cash gift, Pledge amt on a Pledge, Value on a Gift-in-Kind."""
    rows = [{"Constituent ID": "1", "Gift Date": "2025-01-01",
             amount_label: "42.00", "Gift Type": "Cash"}]
    feats = raisers_edge_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 42.0


@pytest.mark.parametrize("date_label", ["Gift Date", "Pledged on", "date"])
def test_every_date_label_the_guide_names_is_accepted(date_label):
    rows = [{"Constituent ID": "1", date_label: "2025-01-01",
             "Gift Amount": "42.00", "Gift Type": "Cash"}]
    feats = raisers_edge_gifts_to_features(rows)
    assert feats.loc["1", "last_gift_date"] == pd.Timestamp("2025-01-01")


def test_currency_formatted_amounts_parse():
    rows = [{"Constituent ID": "1", "Gift Date": "2025-01-01",
             "Gift Amount": "$1,250.00", "Gift Type": "Cash"}]
    feats = raisers_edge_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 1250.0


# --------------------------------------------------------------------------- #
# Schema, identity and recency
# --------------------------------------------------------------------------- #
def test_schema_and_dtypes(gifts):
    feats = raisers_edge_gifts_to_features(gifts)
    assert list(feats.columns) == list(_FEATURE_DTYPES)
    assert feats.index.name == "contact_id"


def test_fund_feeds_distinct_financial_types(gifts):
    feats = raisers_edge_gifts_to_features(gifts, exclude_gift_types=None)
    # Donor 91's two rows sit on Research and Capital.
    assert int(feats.loc["91", "distinct_financial_types"]) == 2
    assert int(feats.loc["88", "distinct_financial_types"]) == 1


def test_carries_identity_fields(gifts):
    feats = raisers_edge_gifts_to_features(gifts)
    assert feats.loc["88", "constituent_email"] == "ada@amc.edu"
    assert feats.loc["88", "first_name"] == "Ada"
    assert feats.loc["88", "last_name"] == "Lovelace"


def test_reference_date_shifts_recency(gifts):
    feats = raisers_edge_gifts_to_features(gifts, reference_date="2025-12-31")
    assert int(feats.loc["88", "recency_days"]) == 296


def test_recency_anchors_on_the_batch_by_default(gifts):
    feats = raisers_edge_gifts_to_features(gifts)
    # Latest surviving gift is donor 88's 2025-03-10 payment.
    assert int(feats.loc["88", "recency_days"]) == 0


def test_empty_input_returns_a_typed_empty_frame():
    feats = raisers_edge_gifts_to_features([])
    assert feats.empty
    assert list(feats.columns) == list(_FEATURE_DTYPES)


def test_every_row_filtered_out_returns_a_typed_empty_frame():
    rows = [{"Constituent ID": "1", "Gift Date": "2025-01-01",
             "Gift Amount": "500.00", "Gift Type": "Pledge"}]
    feats = raisers_edge_gifts_to_features(rows)
    assert feats.empty
    assert list(feats.columns) == list(_FEATURE_DTYPES)
    assert feats.index.name == "contact_id"


def test_missing_required_column_names_the_raisers_edge_fields():
    rows = [{"Constituent ID": "1", "Gift Type": "Cash"}]
    with pytest.raises(KeyError) as excinfo:
        raisers_edge_gifts_to_features(rows)
    message = str(excinfo.value)
    assert "Raiser's Edge" in message
    assert "Gift Date" in message
    # A CiviCRM-worded error here would read as "you used the wrong reader".
    assert "CiviCRM" not in message


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def test_read_single_csv(tmp_path):
    path = tmp_path / "gifts.csv"
    path.write_text(_export(_row("88", "2025-01-10", "1200.00", "Pledge"),
                            _row("88", "2025-02-10", "100.00", "Pay-Cash")))
    raw = read_raisers_edge_gifts(path)
    assert len(raw) == 2
    # Reading is lossless: the pledge row is still there.
    feats = raisers_edge_gifts_to_features(raw)
    assert float(feats.loc["88", "total_gift_amount"]) == 100.0


def test_read_directory_concatenates(tmp_path):
    (tmp_path / "jan.csv").write_text(
        _export(_row("88", "2025-01-10", "100.00", "Cash"))
    )
    (tmp_path / "feb.csv").write_text(
        _export(_row("88", "2025-02-10", "250.00", "Cash"))
    )
    feats = raisers_edge_gifts_to_features(read_raisers_edge_gifts(tmp_path))
    assert float(feats.loc["88", "total_gift_amount"]) == 350.0


def test_read_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_raisers_edge_gifts(tmp_path / "nope.csv")


def test_numeric_looking_constituent_ids_stay_text(tmp_path):
    """dtype=str on the read keeps 0088 from becoming 88, and a blank in the
    column from turning the whole thing into floats."""
    path = tmp_path / "gifts.csv"
    path.write_text(_export(_row("0088", "2025-01-10", "100.00", "Cash"),
                            _row("", "2025-01-11", "50.00", "Cash")))
    feats = raisers_edge_gifts_to_features(read_raisers_edge_gifts(path))
    assert list(feats.index) == ["0088"]
