"""
tests/test_read_gifts.py
Tests for philanthropy.ingest.read_gifts, the preset-registry entry point
over the CiviCRM, Raiser's Edge, NPSP, Bloomerang and DonorPerfect gift
bridges.

The point of this module is that `read_gifts(x, source=name)` is exactly
equivalent to calling that source's own reader-and-aggregator pair directly;
each preset's own filtering behaviour is already covered by
tests/test_civicrm.py, tests/test_raisers_edge.py, tests/test_npsp.py,
tests/test_bloomerang.py and tests/test_donorperfect.py.
"""

import pandas as pd
import pytest

from philanthropy.ingest import (
    GIFT_SOURCES,
    bloomerang_transactions_to_features,
    civicrm_contributions_to_features,
    donorperfect_gifts_to_features,
    npsp_opportunities_to_features,
    raisers_edge_gifts_to_features,
    read_bloomerang_transactions,
    read_civicrm_contributions,
    read_donorperfect_gifts,
    read_gifts,
    read_npsp_opportunities,
    read_raisers_edge_gifts,
    little_green_light_gifts_to_features,
read_little_green_light_gifts,
)


def _write_csv(tmp_path, header, *rows):
    path = tmp_path / "gifts.csv"
    path.write_text("\n".join((header,) + rows) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Path input: read_gifts matches the two-step reader + aggregator call
# --------------------------------------------------------------------------- #
def test_civicrm_path_matches_direct_call(tmp_path):
    path = _write_csv(
        tmp_path,
        "Contact ID,Contribution Date,Total Amount,Contribution Status",
        "101,2025-01-15,250.00,Completed",
    )
    via_registry = read_gifts(path, source="civicrm")
    direct = civicrm_contributions_to_features(read_civicrm_contributions(path))
    pd.testing.assert_frame_equal(via_registry, direct)


def test_raisers_edge_path_matches_direct_call(tmp_path):
    path = _write_csv(
        tmp_path,
        "Constituent ID,Gift Date,Gift Amount,Gift Type",
        "88,2025-01-10,1200.00,Pledge",
        "88,2025-02-10,100.00,Pay-Cash",
    )
    via_registry = read_gifts(path, source="raisers_edge")
    direct = raisers_edge_gifts_to_features(read_raisers_edge_gifts(path))
    pd.testing.assert_frame_equal(via_registry, direct)


def test_npsp_path_matches_direct_call(tmp_path):
    path = _write_csv(
        tmp_path,
        "Account ID,Close Date,Amount,Stage",
        "88,2025-01-10,100.00,Pledged",
        "88,2025-02-10,100.00,Closed Won",
    )
    via_registry = read_gifts(path, source="npsp")
    direct = npsp_opportunities_to_features(read_npsp_opportunities(path))
    pd.testing.assert_frame_equal(via_registry, direct)


def test_bloomerang_path_matches_direct_call(tmp_path):
    path = _write_csv(
        tmp_path,
        "Account Number,Date,Amount,Transaction Type",
        "88,2025-01-10,1200.00,Pledge",
        "88,2025-02-10,100.00,PledgePayment",
    )
    via_registry = read_gifts(path, source="bloomerang")
    direct = bloomerang_transactions_to_features(read_bloomerang_transactions(path))
    pd.testing.assert_frame_equal(via_registry, direct)


def test_donorperfect_path_matches_direct_call(tmp_path):
    path = _write_csv(
        tmp_path,
        "Donor ID,Gift Date,Gift Amount,Record Type",
        "88,2025-01-10,1200.00,P",
        "88,2025-02-10,100.00,G",
    )
    via_registry = read_gifts(path, source="donorperfect")
    direct = donorperfect_gifts_to_features(read_donorperfect_gifts(path))
    pd.testing.assert_frame_equal(via_registry, direct)


# --------------------------------------------------------------------------- #
# In-memory input: no file read, straight to the aggregator
# --------------------------------------------------------------------------- #
def test_dataframe_input_skips_reading_and_goes_straight_to_the_aggregator():
    df = pd.DataFrame(
        [{"Account ID": "7", "Close Date": "2025-05-01",
          "Amount": "300.00", "Stage": "Closed Won"}]
    )
    via_registry = read_gifts(df, source="npsp")
    direct = npsp_opportunities_to_features(df)
    pd.testing.assert_frame_equal(via_registry, direct)


def test_iterable_of_mappings_input():
    rows = [
        {"Contact ID": "101", "Contribution Date": "2025-01-15",
         "Total Amount": "250.00", "Contribution Status": "Completed"},
    ]
    via_registry = read_gifts(rows, source="civicrm")
    direct = civicrm_contributions_to_features(rows)
    pd.testing.assert_frame_equal(via_registry, direct)


# --------------------------------------------------------------------------- #
# Unknown source
# --------------------------------------------------------------------------- #
def test_unknown_source_raises_a_clear_error():
    with pytest.raises(ValueError, match="salesforce_classic"):
        read_gifts([], source="salesforce_classic")


def test_gift_sources_lists_the_six_presets():
    assert set(GIFT_SOURCES) == {
        "civicrm", "raisers_edge", "npsp", "bloomerang", "donorperfect",  "little_green_light",
    }


# --------------------------------------------------------------------------- #
# kwarg passthrough to the underlying aggregator
# --------------------------------------------------------------------------- #
def test_source_specific_kwarg_reaches_the_underlying_aggregator():
    rows = [
        {"Account ID": "88", "Close Date": "2025-01-10", "Amount": "100.00",
         "Stage": "Pledged"},
        {"Account ID": "88", "Close Date": "2025-02-10", "Amount": "100.00",
         "Stage": "Closed Won"},
    ]
    # Default keeps Closed Won only: the Pledged row does not count.
    default = read_gifts(rows, source="npsp")
    assert float(default.loc["88", "total_gift_amount"]) == 100.0
    # include_stages=None disables the filter and sums both rows.
    unfiltered = read_gifts(rows, source="npsp", include_stages=None)
    assert float(unfiltered.loc["88", "total_gift_amount"]) == 200.0


def test_bloomerang_kwarg_reaches_the_underlying_aggregator():
    rows = [
        {"Account Number": "88", "Date": "2025-01-10", "Amount": "1200.00",
         "Transaction Type": "Pledge"},
        {"Account Number": "88", "Date": "2025-02-10", "Amount": "100.00",
         "Transaction Type": "PledgePayment"},
    ]
    # Default excludes the Pledge; only the payment counts.
    default = read_gifts(rows, source="bloomerang")
    assert float(default.loc["88", "total_gift_amount"]) == 100.0
    # exclude_entry_types=None disables the filter and sums both rows.
    unfiltered = read_gifts(rows, source="bloomerang", exclude_entry_types=None)
    assert float(unfiltered.loc["88", "total_gift_amount"]) == 1300.0


def test_donorperfect_kwarg_reaches_the_underlying_aggregator():
    rows = [
        {"Donor ID": "88", "Gift Date": "2025-01-10", "Gift Amount": "1200.00",
         "Record Type": "P"},
        {"Donor ID": "88", "Gift Date": "2025-02-10", "Gift Amount": "100.00",
         "Record Type": "G"},
    ]
    # Default excludes the Pledge; only the gift counts.
    default = read_gifts(rows, source="donorperfect")
    assert float(default.loc["88", "total_gift_amount"]) == 100.0
    # exclude_record_types=None disables the filter and sums both rows.
    unfiltered = read_gifts(rows, source="donorperfect", exclude_record_types=None)
    assert float(unfiltered.loc["88", "total_gift_amount"]) == 1300.0

def test_little_green_light_path_matches_direct_call(tmp_path):
    path = _write_csv(
        tmp_path,
        "LGL Constituent ID,Gift date,Amount,Gift type",
        "88,2025-01-10,1200.00,Pledge",
        "88,2025-02-10,100.00,Gift",
    )
    via_registry = read_gifts(path, source="little_green_light")
    direct = little_green_light_gifts_to_features(
        read_little_green_light_gifts(path)
    )
    pd.testing.assert_frame_equal(via_registry, direct)
    
def test_little_green_light_kwarg_reaches_the_underlying_aggregator():
    rows = [
        {
            "LGL Constituent ID": "88",
            "Gift date": "2025-01-10",
            "Amount": "1200.00",
            "Gift type": "Pledge",
        },
        {
            "LGL Constituent ID": "88",
            "Gift date": "2025-02-10",
            "Amount": "100.00",
            "Gift type": "Gift",
        },
    ]

    # Default excludes the pledge; only the received gift counts.
    default = read_gifts(rows, source="little_green_light")
    assert float(default.loc["88", "total_gift_amount"]) == 100.0

    # None disables the pledge filter and sums both rows.
    unfiltered = read_gifts(
        rows,
        source="little_green_light",
        exclude_gift_types=None,
    )
    assert float(unfiltered.loc["88", "total_gift_amount"]) == 1300.0