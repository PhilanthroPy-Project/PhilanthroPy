"""
tests/test_donorschoose.py
===========================
Unit tests for the DonorsChoose (ICPSR 37898) local-file reader. No network
access: every test reads a tiny CSV fixture written in the same column
layout as a DonorsChoose donations export.
"""

import pandas as pd

from philanthropy.datasets import load_donorschoose

_HEADER = "Donor ID,Donation Amount,Donation Received Date,Project ID"


def _write_fixture(path):
    path.write_text(
        _HEADER
        + "\n"
        + "d1,500.00,2016-03-01,p1\n"
        + "d1,600.00,2017-02-15,p2\n"
        + "d2,25.00,2015-09-10,p3\n"
    )


def test_returns_expected_columns_and_dtypes(tmp_path):
    csv_path = tmp_path / "donations.csv"
    _write_fixture(csv_path)

    gifts = load_donorschoose(str(csv_path))

    assert list(gifts.columns) == ["donor_id", "gift_date", "gift_amount"]
    assert pd.api.types.is_datetime64_any_dtype(gifts["gift_date"])
    assert gifts["gift_amount"].dtype == "float64"


def test_ignores_extra_columns(tmp_path):
    csv_path = tmp_path / "donations.csv"
    _write_fixture(csv_path)

    gifts = load_donorschoose(str(csv_path))

    assert gifts.shape == (3, 3)


def test_values_match_source_rows(tmp_path):
    csv_path = tmp_path / "donations.csv"
    _write_fixture(csv_path)

    gifts = load_donorschoose(str(csv_path))

    d1 = gifts[gifts["donor_id"] == "d1"].sort_values("gift_date")
    assert list(d1["gift_amount"]) == [500.0, 600.0]
    assert d1["gift_date"].iloc[0] == pd.Timestamp("2016-03-01")
