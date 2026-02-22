"""
tests/test_solicitation_window.py
===================================
Unit tests for SolicitationWindowTransformer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from philanthropy.preprocessing import SolicitationWindowTransformer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_X_df(days_list):
    """Return DataFrame with days_since_last_discharge column."""
    return pd.DataFrame({"days_since_last_discharge": days_list})


def _make_X_arr(days_list):
    """Return 2D numpy array with days as first column."""
    return np.array(days_list, dtype=float).reshape(-1, 1)


class TestSolicitationWindowTransformer:

    def test_in_window_correct_for_boundary_dates(self):
        """Exactly at min, max, and midpoint should be in-window."""
        t = SolicitationWindowTransformer(
            min_days_post_discharge=180,
            max_days_post_discharge=730,
        )
        t.fit(_make_X_df([180.0]))
        days = [180.0, 730.0, 455.0]  # min, max, midpoint
        result = t.transform(_make_X_df(days))
        assert result[0, 0] == 1.0, "min_days boundary must be in-window"
        assert result[1, 0] == 1.0, "max_days boundary must be in-window"
        assert result[2, 0] == 1.0, "midpoint must be in-window"

    def test_out_of_window_days_produce_zero_score(self):
        """Days outside [min, max] must have window_score = 0."""
        t = SolicitationWindowTransformer(
            min_days_post_discharge=180,
            max_days_post_discharge=730,
        )
        t.fit(_make_X_arr([[455.0]]))
        # Outside window
        result = t.transform(_make_X_arr([[179.0], [731.0], [0.0], [1000.0]]))
        np.testing.assert_array_equal(result[:, 0], [0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result[:, 1], [0.0, 0.0, 0.0, 0.0])

    def test_midpoint_produces_score_of_one(self):
        """Days at the window midpoint must produce window_score == 1.0."""
        t = SolicitationWindowTransformer(
            min_days_post_discharge=100,
            max_days_post_discharge=200,
        )
        midpoint = 150.0
        t.fit(_make_X_df([midpoint]))
        result = t.transform(_make_X_df([midpoint]))
        assert result[0, 1] == pytest.approx(1.0)

    def test_nan_days_produce_zero_in_both_columns(self):
        """NaN days → both in_window and window_score are 0.0."""
        t = SolicitationWindowTransformer(
            min_days_post_discharge=100, max_days_post_discharge=300
        )
        t.fit(_make_X_df([150.0]))
        result = t.transform(_make_X_df([np.nan, 150.0, np.nan]))
        assert result[0, 0] == 0.0
        assert result[0, 1] == 0.0
        assert result[2, 0] == 0.0
        assert result[2, 1] == 0.0

    def test_window_score_range_is_zero_to_one(self):
        """All window_score values must be in [0.0, 1.0]."""
        t = SolicitationWindowTransformer(
            min_days_post_discharge=90,
            max_days_post_discharge=540,
        )
        days = np.arange(0, 1000, 10, dtype=float)
        X = _make_X_df(days)
        t.fit(X)
        result = t.transform(X)
        assert (result[:, 1] >= 0.0).all()
        assert (result[:, 1] <= 1.0).all()

    def test_invalid_window_raises_value_error(self):
        """min_days >= max_days must raise ValueError at fit time."""
        with pytest.raises(ValueError, match="min_days_post_discharge"):
            t = SolicitationWindowTransformer(
                min_days_post_discharge=300,
                max_days_post_discharge=300,
            )
            t.fit(_make_X_df([100.0]))

        with pytest.raises(ValueError, match="min_days_post_discharge"):
            t = SolicitationWindowTransformer(
                min_days_post_discharge=400,
                max_days_post_discharge=300,
            )
            t.fit(_make_X_df([100.0]))

    def test_output_shape_is_n_by_2(self):
        """Output must always be (n_samples, 2)."""
        t = SolicitationWindowTransformer()
        n = 17
        X = _make_X_df([float(d) for d in range(n)])
        t.fit(X)
        result = t.transform(X)
        assert result.shape == (n, 2)

    def test_returns_ndarray(self):
        """transform() must return np.ndarray, not pd.DataFrame."""
        t = SolicitationWindowTransformer()
        X = _make_X_df([200.0, 500.0])
        t.fit(X)
        result = t.transform(X)
        assert isinstance(result, np.ndarray)

    def test_sklearn_pipeline_compatible(self):
        """Must work inside a Pipeline with StandardScaler downstream."""
        t = SolicitationWindowTransformer(
            min_days_post_discharge=100,
            max_days_post_discharge=500,
        )
        pipe = Pipeline([
            ("window", t),
            ("scaler", StandardScaler()),
        ])
        X = _make_X_df([150.0, 300.0, 50.0, 600.0])
        pipe.fit(X)
        result = pipe.transform(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 2)

    def test_not_fitted_raises(self):
        t = SolicitationWindowTransformer()
        with pytest.raises(NotFittedError):
            t.transform(_make_X_df([100.0]))

    def test_dataframe_input_uses_named_col(self):
        """DataFrame input should use days_since_discharge_col parameter."""
        t = SolicitationWindowTransformer(
            min_days_post_discharge=100,
            max_days_post_discharge=300,
            days_since_discharge_col="days_since_last_discharge",
        )
        X = pd.DataFrame({
            "days_since_last_discharge": [200.0, 50.0],
            "some_other_col": [1.0, 2.0],
        })
        t.fit(X)
        result = t.transform(X)
        assert result[0, 0] == 1.0  # 200 is in [100, 300]
        assert result[1, 0] == 0.0  # 50 is outside

    def test_ndarray_input_uses_first_column(self):
        """ndarray input should use first column as days."""
        t = SolicitationWindowTransformer(
            min_days_post_discharge=100,
            max_days_post_discharge=300,
        )
        X = np.array([[200.0, 999.0], [50.0, 888.0]])
        t.fit(X)
        result = t.transform(X)
        assert result[0, 0] == 1.0  # 200 in window
        assert result[1, 0] == 0.0  # 50 out of window

    def test_window_score_decreases_from_midpoint(self):
        """Window score should be highest at midpoint and decrease toward edges."""
        t = SolicitationWindowTransformer(
            min_days_post_discharge=100,
            max_days_post_discharge=300,
        )
        midpoint = 200.0
        t.fit(_make_X_df([midpoint]))
        days = [100.0, 150.0, 200.0, 250.0, 300.0]
        result = t.transform(_make_X_df(days))
        scores = result[:, 1]
        # Midpoint (index 2) should have highest score
        assert scores[2] == pytest.approx(1.0)
        # Symmetric: score at 100 == score at 300 (both at edges)
        assert scores[0] == pytest.approx(scores[4], abs=1e-10)
        # Monotonically increasing from edge to midpoint
        assert scores[0] < scores[1] < scores[2]
        assert scores[4] < scores[3] < scores[2]
