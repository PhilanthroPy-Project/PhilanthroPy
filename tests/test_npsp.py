"""
tests/test_npsp.py
Tests for the philanthropy.ingest Salesforce NPSP Opportunity bridge.

The point of the module is one filter: NPSP's Recurring Donations feature can
carry both a ``Pledged`` Opportunity for an instalment and the ``Closed Won``
Opportunity recording its receipt, so a naive sum counts the instalment
twice. Most of what follows checks that the ``Pledged`` rows leave and the
closed/won rows stay, in each spelling NPSP writes the export's headers in.
"""

import warnings

import pandas as pd
import pytest

from philanthropy.ingest import (
    DEFAULT_EXCLUDED_STAGES,
    npsp_opportunities_to_features,
    read_npsp_opportunities,
)
from philanthropy.ingest._civicrm import _FEATURE_DTYPES

# Header labels as a Salesforce report export writes them.
_HEADER = "Account ID,Close Date,Amount,Stage,Record Type,Email,First Name,Last Name"


def _row(contact, date, amount, stage, *, record_type="Donation",
         email="", first="", last=""):
    return f"{contact},{date},{amount},{stage},{record_type},{email},{first},{last}"


def _export(*rows):
    return "\n".join((_HEADER,) + rows) + "\n"


@pytest.fixture
def opportunities():
    """One donor with a Recurring Donation instalment that carries both a
    Pledged and a Closed Won row for the same $100, plus one already-received
    instalment; one donor with only a still-open Pledged instalment."""
    return [
        {"Account ID": "88", "Close Date": "2025-01-10", "Amount": "100.00",
         "Stage": "Pledged", "Record Type": "Donation",
         "Email": "ada@amc.edu", "First Name": "Ada", "Last Name": "Lovelace"},
        {"Account ID": "88", "Close Date": "2025-01-10", "Amount": "100.00",
         "Stage": "Closed Won", "Record Type": "Donation",
         "Email": "ada@amc.edu", "First Name": "Ada", "Last Name": "Lovelace"},
        {"Account ID": "88", "Close Date": "2025-02-10", "Amount": "100.00",
         "Stage": "Closed Won", "Record Type": "Recurring Donation",
         "Email": "ada@amc.edu", "First Name": "Ada", "Last Name": "Lovelace"},
        {"Account ID": "91", "Close Date": "2025-03-01", "Amount": "50.00",
         "Stage": "Pledged", "Record Type": "Donation",
         "Email": "grace@amc.edu", "First Name": "Grace", "Last Name": "Hopper"},
    ]


# --------------------------------------------------------------------------- #
# The pledged-versus-closed-won filter
# --------------------------------------------------------------------------- #
def test_pledged_instalment_is_excluded_and_closed_won_is_not(opportunities):
    feats = npsp_opportunities_to_features(opportunities)
    # 100 (the pledge) + 100 + 100 would be 300; only the closed/won rows count.
    assert float(feats.loc["88", "total_gift_amount"]) == 200.0
    assert int(feats.loc["88", "gift_count"]) == 2


def test_every_row_pledged_returns_a_typed_empty_frame_for_that_donor(opportunities):
    feats = npsp_opportunities_to_features(opportunities)
    assert "91" not in feats.index


def test_every_row_filtered_out_returns_a_typed_empty_frame():
    rows = [{"Account ID": "1", "Close Date": "2025-01-01",
             "Amount": "500.00", "Stage": "Pledged"}]
    feats = npsp_opportunities_to_features(rows)
    assert feats.empty
    assert list(feats.columns) == list(_FEATURE_DTYPES)
    assert feats.index.name == "contact_id"


@pytest.mark.parametrize("stage", ["Pledged"])
def test_every_default_excluded_stage_is_dropped(stage):
    rows = [{"Account ID": "1", "Close Date": "2025-01-01",
             "Amount": "500.00", "Stage": stage},
            {"Account ID": "1", "Close Date": "2025-01-02",
             "Amount": "10.00", "Stage": "Closed Won"}]
    feats = npsp_opportunities_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


@pytest.mark.parametrize("stage", ["Closed Won", "Posted", "Prospecting", "Awarded"])
def test_closed_and_other_open_stages_are_kept(stage):
    rows = [{"Account ID": "1", "Close Date": "2025-01-01",
             "Amount": "10.00", "Stage": stage}]
    feats = npsp_opportunities_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


@pytest.mark.parametrize("stage", ["PLEDGED", "pledged", "  Pledged  "])
def test_exclusion_matching_ignores_case_and_spacing(stage):
    rows = [{"Account ID": "1", "Close Date": "2025-01-01",
             "Amount": "500.00", "Stage": stage},
            {"Account ID": "1", "Close Date": "2025-01-02",
             "Amount": "10.00", "Stage": "Closed Won"}]
    feats = npsp_opportunities_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


def test_a_stage_only_containing_pledged_as_a_substring_is_not_excluded():
    """'Not Pledged Yet' must not be swept up by the 'Pledged' default: the
    matching key is compared for equality, not membership."""
    rows = [{"Account ID": "1", "Close Date": "2025-01-01",
             "Amount": "25.00", "Stage": "Not Pledged Yet"}]
    feats = npsp_opportunities_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 25.0


def test_blank_stage_is_kept():
    """An unlabelled row cannot be shown to be a pledge, so it stays."""
    rows = [{"Account ID": "1", "Close Date": "2025-01-01",
             "Amount": "10.00", "Stage": ""},
            {"Account ID": "1", "Close Date": "2025-01-02",
             "Amount": "5.00", "Stage": None}]
    feats = npsp_opportunities_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 15.0
    assert int(feats.loc["1", "gift_count"]) == 2


def test_custom_exclude_set_replaces_the_default(opportunities):
    feats = npsp_opportunities_to_features(opportunities, exclude_stages=("Closed Won",))
    # Pledged is no longer excluded; the two Closed Won rows are.
    assert float(feats.loc["88", "total_gift_amount"]) == 100.0


def test_exclude_none_counts_every_row(opportunities):
    feats = npsp_opportunities_to_features(opportunities, exclude_stages=None)
    assert float(feats.loc["88", "total_gift_amount"]) == 300.0
    assert int(feats.loc["88", "gift_count"]) == 3


def test_empty_exclude_sequence_counts_every_row(opportunities):
    feats = npsp_opportunities_to_features(opportunities, exclude_stages=())
    assert float(feats.loc["88", "total_gift_amount"]) == 300.0


def test_missing_stage_column_warns():
    rows = [{"Account ID": "1", "Close Date": "2025-01-01", "Amount": "10.00"}]
    with pytest.warns(UserWarning, match="no Stage column"):
        feats = npsp_opportunities_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 10.0


def test_exclude_none_does_not_warn_about_a_missing_stage_column():
    rows = [{"Account ID": "1", "Close Date": "2025-01-01", "Amount": "10.00"}]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        npsp_opportunities_to_features(rows, exclude_stages=None)


def test_default_excluded_set_is_documented_and_non_empty():
    assert "Pledged" in DEFAULT_EXCLUDED_STAGES


# --------------------------------------------------------------------------- #
# Header dialects
# --------------------------------------------------------------------------- #
def test_raw_api_field_names_are_accepted():
    """The Opportunity API returns AccountId / CloseDate / Amount / StageName."""
    rows = [{"AccountId": "7", "CloseDate": "2025-05-01", "Amount": "300.00",
             "StageName": "Closed Won"},
            {"AccountId": "7", "CloseDate": "2025-05-02", "Amount": "999.00",
             "StageName": "Pledged"}]
    feats = npsp_opportunities_to_features(rows)
    assert float(feats.loc["7", "total_gift_amount"]) == 300.0


def test_data_import_template_headers_are_accepted():
    """NPSP's own Data Import tool spells them Donation Date / Donation
    Amount / Donation Stage."""
    rows = [{"Account ID": "7", "Donation Date": "2025-05-01",
             "Donation Amount": "300.00", "Donation Stage": "Closed Won"}]
    feats = npsp_opportunities_to_features(rows)
    assert float(feats.loc["7", "total_gift_amount"]) == 300.0


def test_primary_contact_is_accepted_as_the_donor_key():
    rows = [{"Primary Contact": "7", "Close Date": "2025-05-01",
             "Amount": "300.00", "Stage": "Closed Won"}]
    feats = npsp_opportunities_to_features(rows)
    assert float(feats.loc["7", "total_gift_amount"]) == 300.0


def test_export_labels_and_api_names_agree(opportunities):
    api = [
        {"AccountId": o["Account ID"], "CloseDate": o["Close Date"],
         "Amount": o["Amount"], "StageName": o["Stage"],
         "RecordType.Name": o["Record Type"], "email": o["Email"],
         "first_name": o["First Name"], "last_name": o["Last Name"]}
        for o in opportunities
    ]
    pd.testing.assert_frame_equal(
        npsp_opportunities_to_features(opportunities),
        npsp_opportunities_to_features(api),
    )


def test_currency_formatted_amounts_parse():
    rows = [{"Account ID": "1", "Close Date": "2025-01-01",
             "Amount": "$1,250.00", "Stage": "Closed Won"}]
    feats = npsp_opportunities_to_features(rows)
    assert float(feats.loc["1", "total_gift_amount"]) == 1250.0


# --------------------------------------------------------------------------- #
# Schema, identity and recency
# --------------------------------------------------------------------------- #
def test_schema_and_dtypes(opportunities):
    feats = npsp_opportunities_to_features(opportunities)
    assert list(feats.columns) == list(_FEATURE_DTYPES)
    assert feats.index.name == "contact_id"


def test_record_type_feeds_distinct_financial_types(opportunities):
    feats = npsp_opportunities_to_features(opportunities, exclude_stages=None)
    # Donor 88's three rows sit on Donation, Donation and Recurring Donation.
    assert int(feats.loc["88", "distinct_financial_types"]) == 2


def test_carries_identity_fields(opportunities):
    feats = npsp_opportunities_to_features(opportunities)
    assert feats.loc["88", "constituent_email"] == "ada@amc.edu"
    assert feats.loc["88", "first_name"] == "Ada"
    assert feats.loc["88", "last_name"] == "Lovelace"


def test_reference_date_shifts_recency(opportunities):
    feats = npsp_opportunities_to_features(opportunities, reference_date="2025-12-31")
    assert int(feats.loc["88", "recency_days"]) == 324


def test_recency_anchors_on_the_batch_by_default(opportunities):
    feats = npsp_opportunities_to_features(opportunities)
    # Latest surviving row is donor 88's 2025-02-10 Closed Won instalment.
    assert int(feats.loc["88", "recency_days"]) == 0


def test_empty_input_returns_a_typed_empty_frame():
    feats = npsp_opportunities_to_features([])
    assert feats.empty
    assert list(feats.columns) == list(_FEATURE_DTYPES)


def test_missing_required_column_names_the_npsp_fields():
    rows = [{"Account ID": "1", "Stage": "Closed Won"}]
    with pytest.raises(KeyError) as excinfo:
        npsp_opportunities_to_features(rows)
    message = str(excinfo.value)
    assert "NPSP" in message
    assert "Close Date" in message
    # A CiviCRM-worded error here would read as "you used the wrong reader".
    assert "CiviCRM" not in message


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def test_read_single_csv(tmp_path):
    path = tmp_path / "opportunities.csv"
    path.write_text(_export(_row("88", "2025-01-10", "100.00", "Pledged"),
                            _row("88", "2025-02-10", "100.00", "Closed Won")))
    raw = read_npsp_opportunities(path)
    assert len(raw) == 2
    # Reading is lossless: the pledged row is still there.
    feats = npsp_opportunities_to_features(raw)
    assert float(feats.loc["88", "total_gift_amount"]) == 100.0


def test_read_directory_concatenates(tmp_path):
    (tmp_path / "jan.csv").write_text(
        _export(_row("88", "2025-01-10", "100.00", "Closed Won"))
    )
    (tmp_path / "feb.csv").write_text(
        _export(_row("88", "2025-02-10", "250.00", "Closed Won"))
    )
    feats = npsp_opportunities_to_features(read_npsp_opportunities(tmp_path))
    assert float(feats.loc["88", "total_gift_amount"]) == 350.0


def test_read_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_npsp_opportunities(tmp_path / "nope.csv")


def test_numeric_looking_account_ids_stay_text(tmp_path):
    """dtype=str on the read keeps 0088 from becoming 88, and a blank in the
    column from turning the whole thing into floats."""
    path = tmp_path / "opportunities.csv"
    path.write_text(_export(_row("0088", "2025-01-10", "100.00", "Closed Won"),
                            _row("", "2025-01-11", "50.00", "Closed Won")))
    feats = npsp_opportunities_to_features(read_npsp_opportunities(path))
    assert list(feats.index) == ["0088"]
