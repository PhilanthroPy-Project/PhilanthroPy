"""
tests/test_ciob.py
==================
Unit tests for the vendored CIOB "Official Fundraising by City Agencies" loader.
"""

import pandas as pd

from philanthropy.datasets import load_ciob_fundraising

EXPECTED_COLUMNS = ["year", "agency", "name_of_not_for_profit"]


def test_returns_dataframe():
    assert isinstance(load_ciob_fundraising(), pd.DataFrame)


def test_shape_and_columns():
    df = load_ciob_fundraising()
    assert df.shape == (2336, 3)
    assert list(df.columns) == EXPECTED_COLUMNS


def test_year_range():
    df = load_ciob_fundraising()
    assert df["year"].dtype == "int64"
    assert df["year"].between(2019, 2024).all()


def test_registry_content():
    df = load_ciob_fundraising()
    # An affiliation registry: many agencies, many distinct nonprofits.
    assert df["agency"].nunique() > 1
    assert df["name_of_not_for_profit"].nunique() > 100
