"""
tests/test_planned_giving.py
==============================
Unit tests for PlannedGivingSignalTransformer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from philanthropy.preprocessing import PlannedGivingSignalTransformer


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _make_df(**kwargs) -> pd.DataFrame:
    """Create a DataFrame with specified columns."""
    return pd.DataFrame(kwargs)


class TestPlannedGivingSignalTransformer:

    def test_output_shape_is_n_by_4(self):
        """Output must be (n_samples, 4)."""
        t = PlannedGivingSignalTransformer()
        X = _make_df(
            donor_age=[70.0, 60.0, 80.0],
            years_active=[15.0, 5.0, 12.0],
            planned_gift_inclination=[0.8, 0.3, 0.5],
        )
        t.fit(X)
        result = t.transform(X)
        assert result.shape == (3, 4)

    def test_legacy_age_flag_threshold_boundary(self):
        """age=64 → 0, age=65 → 1 (default threshold=65)."""
        t = PlannedGivingSignalTransformer(age_threshold=65)
        X = _make_df(
            donor_age=[64.0, 65.0, 66.0],
            years_active=[5.0, 5.0, 5.0],
            planned_gift_inclination=[0.5, 0.5, 0.5],
        )
        t.fit(X)
        result = t.transform(X)
        assert result[0, 0] == 0.0, "age=64 → is_legacy_age=0"
        assert result[1, 0] == 1.0, "age=65 → is_legacy_age=1"
        assert result[2, 0] == 1.0, "age=66 → is_legacy_age=1"

    def test_loyal_donor_flag_threshold_boundary(self):
        """tenure=9 → 0, tenure=10 → 1 (default threshold=10)."""
        t = PlannedGivingSignalTransformer(tenure_threshold_years=10)
        X = _make_df(
            donor_age=[70.0, 70.0, 70.0],
            years_active=[9.0, 10.0, 11.0],
            planned_gift_inclination=[0.5, 0.5, 0.5],
        )
        t.fit(X)
        result = t.transform(X)
        assert result[0, 1] == 0.0, "tenure=9 → is_loyal_donor=0"
        assert result[1, 1] == 1.0, "tenure=10 → is_loyal_donor=1"
        assert result[2, 1] == 1.0, "tenure=11 → is_loyal_donor=1"

    def test_missing_inclination_produces_negative_one_sentinel(self):
        """Missing planned_gift_inclination → inclination_score = -1.0."""
        t = PlannedGivingSignalTransformer()
        X = _make_df(
            donor_age=[70.0],
            years_active=[15.0],
            planned_gift_inclination=[np.nan],
        )
        t.fit(X)
        result = t.transform(X)
        assert result[0, 2] == -1.0, "NaN inclination → sentinel -1.0"

    def test_composite_score_max_is_three(self):
        """is_legacy_age=1 + is_loyal_donor=1 + inclination_score=1.0 = 3.0."""
        t = PlannedGivingSignalTransformer()
        X = _make_df(
            donor_age=[80.0],
            years_active=[20.0],
            planned_gift_inclination=[1.0],
        )
        t.fit(X)
        result = t.transform(X)
        assert result[0, 3] == pytest.approx(3.0), (
            f"Max composite score should be 3.0, got {result[0, 3]}"
        )

    def test_composite_score_min_is_zero(self):
        """Young, short-tenure donor with low inclination → composite = 0."""
        t = PlannedGivingSignalTransformer()
        X = _make_df(
            donor_age=[30.0],
            years_active=[2.0],
            planned_gift_inclination=[0.0],
        )
        t.fit(X)
        result = t.transform(X)
        assert result[0, 3] == pytest.approx(0.0)

    def test_missing_age_col_produces_zero_flag_not_error(self):
        """If age column is absent, is_legacy_age must be 0 (no exception)."""
        t = PlannedGivingSignalTransformer(age_col="donor_age")
        X = _make_df(
            years_active=[15.0],  # No donor_age column
            planned_gift_inclination=[0.5],
        )
        t.fit(X)
        result = t.transform(X)
        assert result[0, 0] == 0.0, "Missing age_col → is_legacy_age = 0"

    def test_missing_tenure_col_produces_zero_flag_not_error(self):
        """If tenure column is absent, is_loyal_donor must be 0 (no exception)."""
        t = PlannedGivingSignalTransformer(tenure_col="years_active")
        X = _make_df(
            donor_age=[70.0],  # No years_active column
            planned_gift_inclination=[0.5],
        )
        t.fit(X)
        result = t.transform(X)
        assert result[0, 1] == 0.0, "Missing tenure_col → is_loyal_donor = 0"

    def test_missing_inclination_col_produces_sentinel_not_error(self):
        """If inclination column absent → -1.0 sentinel for all rows."""
        t = PlannedGivingSignalTransformer()
        X = _make_df(
            donor_age=[70.0, 60.0],
            years_active=[15.0, 5.0],
            # No planned_gift_inclination column
        )
        t.fit(X)
        result = t.transform(X)
        np.testing.assert_array_equal(result[:, 2], [-1.0, -1.0])

    def test_returns_ndarray(self):
        """transform() must return np.ndarray."""
        t = PlannedGivingSignalTransformer()
        X = _make_df(
            donor_age=[70.0],
            years_active=[15.0],
            planned_gift_inclination=[0.8],
        )
        t.fit(X)
        result = t.transform(X)
        assert isinstance(result, np.ndarray)

    def test_clone_and_set_params_round_trip(self):
        """get_params / clone round-trip must preserve values."""
        t = PlannedGivingSignalTransformer(
            age_threshold=70,
            tenure_threshold_years=15,
            age_col="age",
        )
        T2 = clone(t)
        assert T2.age_threshold == 70
        assert T2.tenure_threshold_years == 15
        assert T2.age_col == "age"
        assert not hasattr(T2, "n_features_in_")

    def test_not_fitted_raises(self):
        t = PlannedGivingSignalTransformer()
        X = _make_df(donor_age=[70.0], years_active=[15.0])
        with pytest.raises(NotFittedError):
            t.transform(X)

    def test_nan_age_produces_zero_flag(self):
        """NaN donor_age must yield is_legacy_age = 0."""
        t = PlannedGivingSignalTransformer(age_threshold=65)
        X = _make_df(
            donor_age=[np.nan, 70.0],
            years_active=[15.0, 15.0],
            planned_gift_inclination=[0.5, 0.5],
        )
        t.fit(X)
        result = t.transform(X)
        assert result[0, 0] == 0.0, "NaN age → is_legacy_age = 0"

    def test_inclination_clipped_to_zero_one(self):
        """Values outside [0,1] must be clipped."""
        t = PlannedGivingSignalTransformer()
        X = _make_df(
            donor_age=[70.0, 70.0],
            years_active=[15.0, 15.0],
            planned_gift_inclination=[-0.5, 1.5],
        )
        t.fit(X)
        result = t.transform(X)
        # Negative clipped to 0
        assert result[0, 2] == pytest.approx(0.0)
        # Above 1 clipped to 1
        assert result[1, 2] == pytest.approx(1.0)

    def test_composite_score_ignores_negative_sentinel(self):
        """composite_score = max(inclination, 0): sentinel -1.0 → contributes 0."""
        t = PlannedGivingSignalTransformer()
        X = _make_df(
            donor_age=[70.0],
            years_active=[15.0],
            planned_gift_inclination=[np.nan],  # sentinel -1.0 in col 2
        )
        t.fit(X)
        result = t.transform(X)
        # is_legacy_age=1 + is_loyal_donor=1 + max(-1, 0)=0 = 2.0
        assert result[0, 3] == pytest.approx(2.0), (
            "composite_score must ignore sentinel -1.0 (treat as 0 in max)"
        )
