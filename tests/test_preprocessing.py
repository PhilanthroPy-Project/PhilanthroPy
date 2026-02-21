"""
tests/test_preprocessing.py
"""

import pandas as pd
import pytest
from philanthropy.preprocessing import CRMCleaner, FiscalYearTransformer


def test_crm_cleaner_fit_transform(donor_df):
    cleaner = CRMCleaner()
    result = cleaner.fit_transform(donor_df)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == donor_df.shape


def test_fiscal_year_transformer_adds_columns(donor_df):
    transformer = FiscalYearTransformer(date_col="gift_date", fiscal_year_start=7)
    result = transformer.fit_transform(donor_df)
    assert "fiscal_year" in result.columns
    assert "fiscal_quarter" in result.columns


def test_fiscal_year_logic():
    df = pd.DataFrame({"gift_date": ["2023-07-01", "2023-06-30"]})
    transformer = FiscalYearTransformer(date_col="gift_date", fiscal_year_start=7)
    result = transformer.fit_transform(df)
    assert result.loc[0, "fiscal_year"] == 2024
    assert result.loc[1, "fiscal_year"] == 2023


def test_invalid_fiscal_year_start():
    transformer = FiscalYearTransformer(fiscal_year_start=13)
    with pytest.raises(ValueError, match="fiscal_year_start"):
        transformer.fit(pd.DataFrame({"gift_date": ["2023-01-01"]}))
