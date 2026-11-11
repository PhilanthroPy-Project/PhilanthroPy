"""
tests/test_read_gift_rows.py
"""

import pandas as pd
import pytest

from philanthropy.ingest import (
    GIFT_SOURCES,
    build_snapshots,
    civicrm_contributions_to_features,
    read_gift_rows,
    read_gifts,
)

# Per source: rows read_gifts counts (two donors, two years) plus one it
# drops (a pledge, an open stage, an unpaid schedule, a cancelled or test
# contribution) and one with no parseable date.
_EXPORTS = {
    "civicrm": (
        "Contribution ID,Contact ID,Contribution Date,Total Amount,Contribution Status,Is Test",
        "1,1,2024-01-10,1000,Cancelled,0",
        "2,1,2024-02-10,100,Completed,0",
        "2,1,2024-02-10,100,Completed,0",
        "3,2,2025-02-10,50,Completed,1",
        "4,2,2025-03-10,60,Completed,0",
        "5,2,not a date,70,Completed,0",
    ),
    "raisers_edge": (
        "Constituent ID,Gift Date,Gift Amount,Gift Type",
        "1,2024-01-10,1000,Pledge",
        "1,2024-02-10,100,Pay-Cash",
        "2,2025-03-10,60,Cash",
        "2,not a date,70,Cash",
    ),
    "npsp": (
        "Account ID,Close Date,Amount,Stage",
        "1,2024-01-10,1000,Pledged",
        "1,2024-02-10,100,Closed Won",
        "2,2025-03-10,60,Closed Won",
        "2,not a date,70,Closed Won",
    ),
    "nonprofit_cloud": (
        "Donor ID,Transaction Date,Amount,Status",
        "1,2024-01-10,1000,Unpaid",
        "1,2024-02-10,100,Paid",
        "2,2025-03-10,60,Paid",
        "2,not a date,70,Paid",
    ),
    "bloomerang": (
        "Account Number,Date,Amount,Transaction Type",
        "1,2024-01-10,1000,Pledge",
        "1,2024-02-10,100,PledgePayment",
        "2,2025-03-10,60,Donation",
        "2,not a date,70,Donation",
    ),
    "donorperfect": (
        "Donor ID,Gift Date,Gift Amount,Record Type",
        "1,2024-01-10,1000,P",
        "1,2024-02-10,100,G",
        "2,2025-03-10,60,G",
        "2,not a date,70,G",
    ),
    "blackbaud_crm": (
        "Constituent ID,Date,Amount,Revenue Type",
        "1,2024-01-10,1000,Pledge",
        "1,2024-02-10,100,Payment",
        "2,2025-03-10,60,Payment",
        "2,not a date,70,Payment",
    ),
    "little_green_light": (
        "LGL Constituent ID,Gift date,Amount,Gift type",
        "1,2024-01-10,1000,Pledge",
        "1,2024-02-10,100,Gift",
        "2,2025-03-10,60,Gift",
        "2,not a date,70,Gift",
    ),
    "ellucian_advance": (
        "Entity ID,Date,Legal,Type",
        "1,2024-01-10,1000,Straight Pledge (Primary)",
        "1,2024-02-10,100,Pledge Payment",
        "2,2025-03-10,60,Outright Gift",
        "2,not a date,70,Outright Gift",
    ),
    "neon": (
        "Account ID,Donation Date,Donation Amount,Donation Type",
        "1,2024-01-10,1000,Pledge",
        "1,2024-02-10,100,Pledge Payment",
        "2,2025-03-10,60,Donation",
        "2,not a date,70,Donation",
    ),
}


def _write(tmp_path, source):
    path = tmp_path / f"{source}.csv"
    path.write_text("\n".join(_EXPORTS[source]) + "\n")
    return path


def test_every_gift_source_is_covered():
    from philanthropy.ingest._read_gifts import _ROW_FILTERS

    assert set(_ROW_FILTERS) == set(GIFT_SOURCES) == set(_EXPORTS)


@pytest.mark.parametrize("source", GIFT_SOURCES)
def test_rolling_the_rows_up_gives_read_gifts(tmp_path, source):
    path = _write(tmp_path, source)
    rows = read_gift_rows(path, source=source)
    assert list(rows.columns) == ["donor_id", "gift_date", "gift_amount"]
    assert rows["donor_id"].tolist() == ["1", "2"]
    assert rows["gift_amount"].tolist() == [100.0, 60.0]

    rolled = civicrm_contributions_to_features(
        rows.rename(columns={"donor_id": "contact_id", "gift_date": "receive_date",
                             "gift_amount": "total_amount"}),
        statuses=None,
    )
    pd.testing.assert_frame_equal(rolled, read_gifts(path, source=source))


def test_filter_option_reaches_the_source_filter(tmp_path):
    path = _write(tmp_path, "raisers_edge")
    rows = read_gift_rows(path, source="raisers_edge", exclude_gift_types=None)
    assert rows["gift_amount"].tolist() == [1000.0, 100.0, 60.0]
    civi = read_gift_rows(_write(tmp_path, "civicrm"), source="civicrm", statuses=None)
    assert 1000.0 in civi["gift_amount"].tolist()


def test_unknown_source_and_unknown_option_raise(tmp_path):
    with pytest.raises(ValueError, match="Unknown gift source"):
        read_gift_rows([], source="not_a_crm")
    with pytest.raises(TypeError):
        read_gift_rows(_write(tmp_path, "npsp"), source="npsp", reference_date="2025-01-01")


def test_empty_input_keeps_the_columns():
    rows = read_gift_rows([], source="bloomerang")
    assert rows.empty and list(rows.columns) == ["donor_id", "gift_date", "gift_amount"]


def test_rows_feed_build_snapshots():
    from philanthropy.datasets import make_donor_panel

    gifts = make_donor_panel(n_donors=200, n_years=4, random_state=0)["gifts"]
    export = pd.DataFrame({
        "Constituent ID": gifts["donor_id"].astype(str),
        "Gift Date": gifts["gift_date"].dt.strftime("%Y-%m-%d"),
        "Gift Amount": gifts["gift_amount"].astype(str),
        "Gift Type": "Cash",
    })
    rows = read_gift_rows(export, source="raisers_edge")
    snaps = build_snapshots(rows, kind="response_next_year")
    expected = build_snapshots(gifts.assign(donor_id=gifts["donor_id"].astype(str)), kind="response_next_year")
    assert len(snaps) > 0
    pd.testing.assert_frame_equal(snaps, expected, check_dtype=False, check_index_type=False)
