"""
tests/test_preprocessing_properties.py
======================================
Property-based tests for PhilanthroPy transformers using Hypothesis.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings
from philanthropy.preprocessing import (
    FiscalYearTransformer,
    WealthScreeningImputer,
    SolicitationWindowTransformer
)

@st.composite
def df_strategy(draw):
    """Generates a random DataFrame with gift dates and numeric values."""
    n_samples = draw(st.integers(min_value=1, max_value=50))
    data = {
        "gift_date": pd.to_datetime(draw(st.lists(st.dates(), min_size=n_samples, max_size=n_samples))),
        "amount": draw(st.lists(st.one_of(st.floats(min_value=0, max_value=1e6), st.just(np.nan)), min_size=n_samples, max_size=n_samples)),
        "days_since_last_gift": draw(st.lists(st.integers(min_value=0, max_value=5000), min_size=n_samples, max_size=n_samples)),
        "net_worth": draw(st.lists(st.one_of(st.floats(min_value=0, max_value=1e10), st.just(np.nan)), min_size=n_samples, max_size=n_samples)),
    }
    return pd.DataFrame(data)

@settings(max_examples=20, deadline=None)
@given(df_strategy())
def test_fiscal_year_transformer_properties(df):
    """FiscalYearTransformer should never change the number of rows or contain years < 1900."""
    t = FiscalYearTransformer(fiscal_year_start=7).set_output(transform="pandas")
    out = t.fit_transform(df)
    assert len(out) == len(df)
    assert "fiscal_year" in out.columns
    assert "fiscal_quarter" in out.columns
    # Basic sanity: years shouldn't be crazy
    valid_years = out["fiscal_year"].dropna()
    if not valid_years.empty:
        assert (valid_years >= 1900).all()
        assert (valid_years <= 2100).all()

@settings(max_examples=20, deadline=None)
@given(df_strategy())
def test_wealth_imputer_properties(df):
    """WealthScreeningImputer should eliminate all NaNs in specified columns."""
    wealth_cols = ["net_worth"]
    imputer = WealthScreeningImputer(wealth_cols=wealth_cols, strategy="median").set_output(transform="pandas")
    out = imputer.fit_transform(df)
    assert len(out) == len(df)
    # Check if NaNs are gone in the output
    assert not out["net_worth"].isna().any()

@settings(max_examples=20, deadline=None)
@given(df_strategy())
def test_solicitation_window_properties(df):
    """SolicitationWindowTransformer should return binary values (0 or 1)."""
    t = SolicitationWindowTransformer(days_since_last_gift_col="days_since_last_gift", min_days=100, max_days=200).set_output(transform="pandas")
    out = t.fit_transform(df)
    assert len(out) == len(df)
    # The new column
    is_in_window = out["is_in_optimal_window"]
    assert set(np.unique(is_in_window)).issubset({0.0, 1.0})
