"""
tests/test_preprocessing_properties.py
======================================
Property-based tests for PhilanthroPy transformers using Hypothesis.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from philanthropy.preprocessing import (
    FiscalYearTransformer,
    WealthScreeningImputer,
    SolicitationWindowTransformer,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

fy_start_months = st.integers(min_value=1, max_value=12)

gift_dates = st.datetimes(
    min_value=pd.Timestamp("1950-01-01"),
    max_value=pd.Timestamp("2099-12-31"),
)

gift_amounts = st.one_of(
    st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False),
    st.just(0.0),
    st.just(1e9),
)


@st.composite
def fiscal_year_dataframes(draw, min_rows=1, max_rows=200):
    """Returns pd.DataFrame with gift_date (datetime) and gift_amount (float)."""
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))
    dates = [draw(gift_dates) for _ in range(n)]
    amounts = [draw(gift_amounts) for _ in range(n)]
    return pd.DataFrame({
        "gift_date": pd.to_datetime(dates),
        "gift_amount": amounts,
    })


@st.composite
def wealth_dataframes(draw, min_rows=2, max_rows=100):
    """Returns pd.DataFrame with estimated_net_worth (30% NaN), real_estate_value (30% NaN), gift_amount."""
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))
    enw = []
    rev = []
    amounts = []
    for _ in range(n):
        is_nan_enw = draw(st.booleans())
        is_nan_rev = draw(st.booleans())
        # roughly 50% NaN via booleans
        enw.append(
            np.nan if is_nan_enw
            else draw(st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False))
        )
        rev.append(
            np.nan if is_nan_rev
            else draw(st.floats(min_value=0.0, max_value=5e8, allow_nan=False, allow_infinity=False))
        )
        amounts.append(draw(st.floats(min_value=0.0, max_value=1e7, allow_nan=False, allow_infinity=False)))
    return pd.DataFrame({
        "estimated_net_worth": enw,
        "real_estate_value": rev,
        "gift_amount": amounts,
    })


# ---------------------------------------------------------------------------
# FiscalYearTransformer properties
# ---------------------------------------------------------------------------

@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(df=fiscal_year_dataframes(), fy_start=fy_start_months)
def test_fiscal_year_transformer_properties(df, fy_start):
    """FiscalYearTransformer should never change the number of rows and fiscal_years should be reasonable."""
    t = FiscalYearTransformer(fiscal_year_start=fy_start).set_output(transform="pandas")
    out = t.fit_transform(df)
    assert len(out) == len(df)
    assert "fiscal_year" in out.columns
    assert "fiscal_quarter" in out.columns
    # Basic sanity: years should be reasonable for 1950-2099 input
    valid_years = out["fiscal_year"].dropna()
    if not valid_years.empty:
        assert (valid_years >= 1950).all()
        assert (valid_years <= 2100).all()
    # Quarters should be in [1, 4]
    valid_quarters = out["fiscal_quarter"].dropna()
    if not valid_quarters.empty:
        assert (valid_quarters >= 1).all()
        assert (valid_quarters <= 4).all()


# ---------------------------------------------------------------------------
# WealthScreeningImputer properties
# ---------------------------------------------------------------------------

@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(df=wealth_dataframes())
def test_wealth_imputer_properties(df):
    """WealthScreeningImputer should eliminate all NaNs in specified columns."""
    wealth_cols = ["estimated_net_worth", "real_estate_value"]
    imputer = WealthScreeningImputer(
        wealth_cols=wealth_cols, strategy="median", add_indicator=False
    ).set_output(transform="pandas")
    out = imputer.fit_transform(df)
    assert len(out) == len(df)
    # Check if NaNs are gone in the imputed columns
    for col in wealth_cols:
        if col in out.columns:
            assert not out[col].isna().any(), f"NaN found in {col} after imputation"


# ---------------------------------------------------------------------------
# SolicitationWindowTransformer properties
# ---------------------------------------------------------------------------

@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(
    days=st.lists(
        st.one_of(
            st.integers(min_value=0, max_value=5000),
            st.just(None),
        ),
        min_size=1,
        max_size=50,
    )
)
def test_solicitation_window_properties(days):
    """SolicitationWindowTransformer should return binary in_window and score in [0, 1]."""
    # Build a simple numeric DataFrame with days_since_discharge
    parsed = [float(d) if d is not None else np.nan for d in days]
    df = pd.DataFrame({"days_since_last_discharge": parsed})
    t = SolicitationWindowTransformer(
        days_since_discharge_col="days_since_last_discharge",
        min_days_post_discharge=100,
        max_days_post_discharge=200,
    )
    t.fit(df)
    out = t.transform(df)
    assert isinstance(out, np.ndarray), "transform() must return np.ndarray"
    assert out.shape == (len(days), 2)
    assert len(out) == len(days)
    # in_window column (col 0) must be binary
    assert set(np.unique(out[~np.isnan(out[:, 0]), 0])).issubset(
        {0.0, 1.0}
    ), "in_window flag must be 0 or 1"
    # window_score column (col 1) must be in [0, 1]
    scores = out[:, 1]
    non_nan_scores = scores[~np.isnan(scores)]
    if len(non_nan_scores) > 0:
        assert (non_nan_scores >= 0.0).all() and (non_nan_scores <= 1.0).all()
