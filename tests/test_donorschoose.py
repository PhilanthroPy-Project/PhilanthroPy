"""
tests/test_donorschoose.py
===========================
Unit tests for the DonorsChoose (ICPSR 37898, DS0001) local-file reader. No
network access: every test reads a tiny TSV fixture written in the real
study's column layout (DONOR_ID, AMOUNT, CREATED_MONTH month-resolution
dates, Yes/No flags with ICPSR's trailing-space padding, DONOR_TYPE).
"""

import pandas as pd

from philanthropy.datasets import load_donorschoose

_HEADER = (
    "DONATION_ID\tAMOUNT\tCREATED_MONTH\t"
    "PAYMENT_INCLUDED_CAMPAIGN_GIFT_1\tPAYMENT_INCLUDED_WEB_PURCHASED_1\t"
    "PAYMENT_WAS_MATCHED\tTHANK_YOU_PACKET_MAILED\tIS_TEACHER_REFERRED\t"
    "DONOR_TYPE\tPROJECT_ID\tDONOR_ID"
)

_ROWS = [
    "a1\t500\t2016-03\tYes\tNo \tNo \tNo \tf\tcitizen donor\tp1\td1",
    "a2\t600\t2017-02\tNo \tNo \tYes\tNo \tf\tcitizen donor\tp2\td1",
    "a3\t-5\t2015-09\tNo \tNo \tNo \tNo \tf\tteacher      \tp3\td2",
]


def _write_fixture(path):
    path.write_text("\n".join([_HEADER, *_ROWS]) + "\n")


def test_returns_expected_columns_and_dtypes(tmp_path):
    csv_path = tmp_path / "donations.tsv"
    _write_fixture(csv_path)

    gifts = load_donorschoose(str(csv_path))

    assert list(gifts.columns) == [
        "donor_id",
        "gift_date",
        "gift_amount",
        "donor_type",
        "included_campaign_gift_card",
        "included_web_purchased_gift_card",
        "was_matched",
    ]
    assert pd.api.types.is_datetime64_any_dtype(gifts["gift_date"])
    assert gifts["gift_amount"].dtype == "float64"


def test_month_resolution_dates_land_on_first_of_month(tmp_path):
    csv_path = tmp_path / "donations.tsv"
    _write_fixture(csv_path)

    gifts = load_donorschoose(str(csv_path))

    assert gifts["gift_date"].iloc[0] == pd.Timestamp("2016-03-01")
    assert gifts["gift_date"].iloc[1] == pd.Timestamp("2017-02-01")


def test_padded_yes_no_flags_and_donor_type_are_stripped(tmp_path):
    csv_path = tmp_path / "donations.tsv"
    _write_fixture(csv_path)

    gifts = load_donorschoose(str(csv_path))

    assert gifts["donor_type"].tolist() == [
        "citizen donor",
        "citizen donor",
        "teacher",
    ]
    assert gifts["included_campaign_gift_card"].tolist() == [True, False, False]
    assert gifts["was_matched"].tolist() == [False, True, False]


def test_non_positive_amounts_pass_through_unfiltered(tmp_path):
    csv_path = tmp_path / "donations.tsv"
    _write_fixture(csv_path)

    gifts = load_donorschoose(str(csv_path))

    assert gifts.shape == (3, 7)
    assert gifts["gift_amount"].tolist() == [500.0, 600.0, -5.0]
