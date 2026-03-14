"""
tests/test_datasets.py
=======================
Unit tests for philanthropy.datasets.generate_synthetic_donor_data.
"""

import numpy as np
import pandas as pd
import pytest

from philanthropy.datasets import generate_synthetic_donor_data


# ---------------------------------------------------------------------------
# Schema & shape
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "total_gift_amount",
    "years_active",
    "last_gift_date",
    "event_attendance_count",
    "is_major_donor",
}


def test_returns_dataframe():
    df = generate_synthetic_donor_data(n_samples=50, random_state=0)
    assert isinstance(df, pd.DataFrame)


def test_correct_shape():
    df = generate_synthetic_donor_data(n_samples=200, random_state=1)
    assert df.shape == (200, 5)


def test_required_columns_present():
    df = generate_synthetic_donor_data(n_samples=10, random_state=2)
    assert REQUIRED_COLUMNS.issubset(df.columns)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

def test_total_gift_amount_is_float():
    df = generate_synthetic_donor_data(n_samples=50, random_state=3)
    assert pd.api.types.is_float_dtype(df["total_gift_amount"])


def test_years_active_is_integer():
    df = generate_synthetic_donor_data(n_samples=50, random_state=4)
    assert pd.api.types.is_integer_dtype(df["years_active"])


def test_last_gift_date_is_datetime():
    df = generate_synthetic_donor_data(n_samples=50, random_state=5)
    assert pd.api.types.is_datetime64_any_dtype(df["last_gift_date"])


def test_event_attendance_count_is_integer():
    df = generate_synthetic_donor_data(n_samples=50, random_state=6)
    assert pd.api.types.is_integer_dtype(df["event_attendance_count"])


def test_is_major_donor_is_binary():
    df = generate_synthetic_donor_data(n_samples=200, random_state=7)
    assert df["is_major_donor"].isin([0, 1]).all()


# ---------------------------------------------------------------------------
# Value ranges
# ---------------------------------------------------------------------------

def test_total_gift_amount_positive():
    df = generate_synthetic_donor_data(n_samples=100, random_state=8)
    assert (df["total_gift_amount"] > 0).all()


def test_years_active_range():
    df = generate_synthetic_donor_data(n_samples=500, random_state=9)
    assert df["years_active"].between(1, 30).all()


def test_event_attendance_range():
    df = generate_synthetic_donor_data(n_samples=500, random_state=10)
    assert df["event_attendance_count"].between(0, 20).all()


def test_last_gift_date_within_five_years():
    df = generate_synthetic_donor_data(n_samples=200, random_state=11)
    reference = pd.Timestamp("2026-02-21")
    five_years_ago = reference - pd.DateOffset(years=5)
    assert (df["last_gift_date"] >= five_years_ago).all()
    assert (df["last_gift_date"] <= reference).all()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_reproducible_with_seed():
    df1 = generate_synthetic_donor_data(n_samples=100, random_state=42)
    df2 = generate_synthetic_donor_data(n_samples=100, random_state=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_differ():
    df1 = generate_synthetic_donor_data(n_samples=100, random_state=0)
    df2 = generate_synthetic_donor_data(n_samples=100, random_state=999)
    assert not df1["total_gift_amount"].equals(df2["total_gift_amount"])


# ---------------------------------------------------------------------------
# Label correlation (domain plausibility)
# ---------------------------------------------------------------------------

def test_major_donors_have_higher_gift_amounts():
    """Major donors should have materially higher average total_gift_amount."""
    df = generate_synthetic_donor_data(n_samples=2000, random_state=13)
    mean_major = df.loc[df["is_major_donor"] == 1, "total_gift_amount"].mean()
    mean_standard = df.loc[df["is_major_donor"] == 0, "total_gift_amount"].mean()
    assert mean_major > mean_standard


def test_major_donors_have_higher_event_attendance():
    """Major donors should attend more events on average."""
    df = generate_synthetic_donor_data(n_samples=2000, random_state=14)
    mean_major = df.loc[df["is_major_donor"] == 1, "event_attendance_count"].mean()
    mean_standard = df.loc[df["is_major_donor"] == 0, "event_attendance_count"].mean()
    assert mean_major > mean_standard


def test_both_classes_present():
    """Dataset must contain at least one positive and one negative example."""
    df = generate_synthetic_donor_data(n_samples=500, random_state=15)
    assert df["is_major_donor"].sum() > 0
    assert (df["is_major_donor"] == 0).sum() > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_zero_samples_returns_empty_df():
    df = generate_synthetic_donor_data(n_samples=0, random_state=0)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 0
    assert REQUIRED_COLUMNS.issubset(df.columns)


def test_no_random_state_does_not_raise():
    df = generate_synthetic_donor_data(n_samples=20)
    assert df.shape == (20, 5)
