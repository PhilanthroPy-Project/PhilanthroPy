"""
tests/test_ellucian_advance.py
Tests for the philanthropy.ingest Ellucian CRM Advance gift bridge.
"""

import warnings

import pytest

from philanthropy.ingest import (
    DEFAULT_EXCLUDED_ADVANCE_TRANSACTION_TYPES,
    ellucian_advance_gifts_to_features,
    read_ellucian_advance_gifts,
)
from philanthropy.ingest._civicrm import _FEATURE_DTYPES

_HEADER = "Entity ID,Date,Legal,Type,Allocation,Email,First Name,Last Name"


def _row(contact, date, amount, gift_type, *, allocation="Annual Fund",
         email="", first="", last=""):
    return (
        f"{contact},{date},{amount},{gift_type},{allocation},"
        f"{email},{first},{last}"
    )


def _export(*rows):
    return "\n".join((_HEADER,) + rows) + "\n"


@pytest.fixture
def gifts():
    return [
        {"Entity ID": "88", "Date": "2025-01-10", "Legal": "1200.00",
         "Type": "Straight Pledge (Primary)", "Allocation": "Annual Fund",
         "Email": "ada@univ.edu", "First Name": "Ada", "Last Name": "Lovelace"},
        {"Entity ID": "88", "Date": "2025-02-10", "Legal": "100.00",
         "Type": "Pledge Payment (Primary)", "Allocation": "Annual Fund",
         "Email": "ada@univ.edu", "First Name": "Ada", "Last Name": "Lovelace"},
        {"Entity ID": "88", "Date": "2025-03-10", "Legal": "100.00",
         "Type": "Gift", "Allocation": "Scholarship",
         "Email": "ada@univ.edu", "First Name": "Ada", "Last Name": "Lovelace"},
    ]


def test_pledge_is_excluded_and_payments_are_kept(gifts):
    feats = ellucian_advance_gifts_to_features(gifts)
    assert float(feats.loc["88", "total_gift_amount"]) == 200.0
    assert int(feats.loc["88", "gift_count"]) == 2


@pytest.mark.parametrize(
    "gift_type",
    [f"{t} (Primary)" for t in DEFAULT_EXCLUDED_ADVANCE_TRANSACTION_TYPES],
)
def test_default_excluded_types_are_dropped(gift_type):
    rows = [
        {"Entity ID": "1", "Date": "2025-01-01", "Legal": "500.00",
         "Type": gift_type},
        {"Entity ID": "1", "Date": "2025-01-02", "Legal": "10.00",
         "Type": "Gift"},
    ]
    feats = ellucian_advance_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


@pytest.mark.parametrize(
    "gift_type",
    ["Gift", "Outright Gift (Primary)", "Pledge Payment (Joint)",
     "Realized Bequest (Primary)", "Grant (Primary)"],
)
def test_payment_and_outright_types_are_kept(gift_type):
    rows = [
        {"Entity ID": "1", "Date": "2025-01-01", "Legal": "10.00",
         "Type": gift_type},
    ]
    feats = ellucian_advance_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


@pytest.mark.parametrize(
    "gift_type",
    ["StraightPledge", "straight_pledge", "  STRAIGHT PLEDGE  ",
     "Straight Pledge (Covering Account)"],
)
def test_exclusion_matching_ignores_case_spacing_and_punctuation(gift_type):
    rows = [
        {"Entity ID": "1", "Date": "2025-01-01", "Legal": "500.00",
         "Type": gift_type},
        {"Entity ID": "1", "Date": "2025-01-02", "Legal": "10.00",
         "Type": "Gift"},
    ]
    feats = ellucian_advance_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


def test_exclude_none_counts_every_row(gifts):
    feats = ellucian_advance_gifts_to_features(
        gifts, exclude_transaction_types=None
    )
    assert float(feats.loc["88", "total_gift_amount"]) == 1400.0
    assert int(feats.loc["88", "gift_count"]) == 3


def test_custom_exclude_set_replaces_the_default(gifts):
    feats = ellucian_advance_gifts_to_features(
        gifts, exclude_transaction_types=("Pledge Payment",)
    )
    assert float(feats.loc["88", "total_gift_amount"]) == 1300.0


def test_missing_type_column_warns():
    rows = [{"Entity ID": "1", "Date": "2025-01-01", "Legal": "10.00"}]
    with pytest.warns(UserWarning, match="no Type column"):
        feats = ellucian_advance_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


def test_exclude_none_does_not_warn_about_missing_type_column():
    rows = [{"Entity ID": "1", "Date": "2025-01-01", "Legal": "10.00"}]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ellucian_advance_gifts_to_features(
            rows, exclude_transaction_types=None
        )


def test_default_excluded_set_is_documented_and_non_empty():
    assert "Straight Pledge" in DEFAULT_EXCLUDED_ADVANCE_TRANSACTION_TYPES
    # Grant is both a pledge type and a received-gift type in Advance, so the
    # Type column alone cannot say which; it is deliberately not excluded.
    assert "Grant" not in DEFAULT_EXCLUDED_ADVANCE_TRANSACTION_TYPES


def test_snake_case_field_names_are_accepted():
    rows = [
        {"id_number": "7", "date": "2025-05-01",
         "legal_amount": "300.00", "transaction_type": "Gift",
         "allocation": "Scholarship"},
        {"id_number": "7", "date": "2025-05-02",
         "legal_amount": "999.00", "transaction_type": "Straight Pledge",
         "allocation": "Scholarship"},
    ]
    feats = ellucian_advance_gifts_to_features(rows)
    assert float(feats.loc["7", "total_gift_amount"]) == 300.0


@pytest.mark.parametrize("id_label", ["Entity ID", "ID Number"])
def test_every_entity_id_alias_is_accepted(id_label):
    rows = [{id_label: "1", "Date": "2025-01-01", "Legal": "42.00",
             "Type": "Gift"}]
    feats = ellucian_advance_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 42.0


@pytest.mark.parametrize("amount_label", ["Legal", "Legal Amount", "Amount"])
def test_every_amount_alias_is_accepted(amount_label):
    rows = [{"Entity ID": "1", "Date": "2025-01-01", amount_label: "42.00",
             "Type": "Gift"}]
    feats = ellucian_advance_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 42.0


def test_bare_id_column_is_not_taken_as_the_entity():
    # A bare "ID" in a transaction export may be the receipt number, so it
    # must not key donors; the entity column wins and "ID" is ignored.
    rows = [
        {"ID": "R1", "Entity ID": "1", "Date": "2025-01-01", "Legal": "5.00",
         "Type": "Gift"},
        {"ID": "R2", "Entity ID": "1", "Date": "2025-01-02", "Legal": "7.00",
         "Type": "Gift"},
    ]
    feats = ellucian_advance_gifts_to_features(rows)
    assert list(feats.index) == ["1"]
    assert float(feats.loc["1", "total_gift_amount"]) == 12.0


def test_soft_credit_is_not_summed():
    rows = [{"Entity ID": "1", "Date": "2025-01-01", "Legal": "100.00",
             "Credit": "900.00", "Type": "Gift"}]
    feats = ellucian_advance_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 100.0


def test_receipt_number_after_the_date_is_stripped():
    rows = [
        {"Entity ID": "1", "Date": "01/10/2025 #0001234", "Legal": "10.00",
         "Type": "Gift (Primary)"},
        {"Entity ID": "1", "Date": "03/15/2025 #0001301", "Legal": "20.00",
         "Type": "Gift (Primary)"},
    ]
    feats = ellucian_advance_gifts_to_features(
        rows, reference_date="2025-03-25"
    )
    assert int(feats.loc["1", "gift_count"]) == 2
    assert int(feats.loc["1", "recency_days"]) == 10


def test_currency_formatted_amounts_parse():
    rows = [{"Entity ID": "1", "Date": "2025-01-01", "Legal": "$1,250.00",
             "Type": "Gift"}]
    feats = ellucian_advance_gifts_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 1250.0


def test_schema_identity_and_allocation(gifts):
    feats = ellucian_advance_gifts_to_features(gifts)
    assert list(feats.columns) == list(_FEATURE_DTYPES)
    assert feats.index.name == "contact_id"
    assert feats.loc["88", "constituent_email"] == "ada@univ.edu"
    assert feats.loc["88", "first_name"] == "Ada"
    assert feats.loc["88", "last_name"] == "Lovelace"
    assert int(feats.loc["88", "distinct_financial_types"]) == 2


def test_reference_date_shifts_recency(gifts):
    feats = ellucian_advance_gifts_to_features(
        gifts, reference_date="2025-12-31"
    )
    assert int(feats.loc["88", "recency_days"]) == 296


def test_empty_input_returns_a_typed_empty_frame():
    feats = ellucian_advance_gifts_to_features([])
    assert feats.empty
    assert list(feats.columns) == list(_FEATURE_DTYPES)


def test_every_row_filtered_out_returns_a_typed_empty_frame():
    rows = [{"Entity ID": "1", "Date": "2025-01-01", "Legal": "500.00",
             "Type": "Straight Pledge (Primary)"}]
    feats = ellucian_advance_gifts_to_features(rows)
    assert feats.empty
    assert list(feats.columns) == list(_FEATURE_DTYPES)
    assert feats.index.name == "contact_id"


def test_missing_required_column_names_the_advance_fields():
    rows = [{"Entity ID": "1", "Type": "Gift"}]
    with pytest.raises(KeyError) as excinfo:
        ellucian_advance_gifts_to_features(rows)
    message = str(excinfo.value)
    assert "Ellucian CRM Advance" in message
    assert "Legal" in message
    assert "CiviCRM" not in message


def test_read_single_csv(tmp_path):
    path = tmp_path / "gifts.csv"
    path.write_text(
        _export(
            _row("88", "01/10/2025 #0001234", "1200.00",
                 "Straight Pledge (Primary)"),
            _row("88", "02/10/2025 #0001301", "100.00",
                 "Pledge Payment (Primary)"),
        )
    )
    raw = read_ellucian_advance_gifts(path)
    assert len(raw) == 2
    feats = ellucian_advance_gifts_to_features(raw)
    assert float(feats.loc["88", "total_gift_amount"]) == 100.0


def test_read_directory_concatenates(tmp_path):
    (tmp_path / "jan.csv").write_text(
        _export(_row("88", "2025-01-10", "100.00", "Gift"))
    )
    (tmp_path / "feb.csv").write_text(
        _export(_row("88", "2025-02-10", "250.00", "Gift"))
    )
    feats = ellucian_advance_gifts_to_features(
        read_ellucian_advance_gifts(tmp_path)
    )
    assert float(feats.loc["88", "total_gift_amount"]) == 350.0


def test_read_gift_rows_strips_receipt_suffix():
    from philanthropy.ingest import read_gift_rows

    rows = [
        {"Entity ID": "88", "Date": "01/10/2025 #0001234",
         "Legal": "1200.00", "Type": "Straight Pledge (Primary)"},
        {"Entity ID": "88", "Date": "02/10/2025 #0001301",
         "Legal": "100.00", "Type": "Pledge Payment (Primary)"},
    ]
    out = read_gift_rows(rows, source="ellucian_advance")
    assert out["gift_amount"].tolist() == [100.0]


def test_soft_credit_only_row_is_not_a_gift():
    rows = [
        {"Entity ID": "1", "Date": "2020-01-01", "Legal": "100", "Credit": "100", "Type": "Gift (Primary)"},
        {"Entity ID": "1", "Date": "2025-01-01", "Legal": "", "Credit": "500", "Type": "Gift (Joint)"},
        {"Entity ID": "1", "Date": "2025-02-01", "Legal": "0", "Credit": "500", "Type": "Gift (Joint)"},
    ]
    feats = ellucian_advance_gifts_to_features(rows)
    assert int(feats.loc["1", "gift_count"]) == 1
    assert str(feats.loc["1", "last_gift_date"].date()) == "2020-01-01"
