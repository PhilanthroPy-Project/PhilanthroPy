"""
philanthropy.preprocessing._discharge_window
============================================
Post-discharge solicitation window featurization for grateful patient programs.

Given a clinical discharge date and a solicitation window (in days), determines
whether each gift falls within N days after discharge. Produces in_window,
window_position_score, and discharge_recency_tier columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class DischargeToSolicitationWindowTransformer(TransformerMixin, BaseEstimator):
    """Flag donors in the clinical fundraising post-discharge solicitation window.

    This transformer outputs three features:
    - ``in_solicitation_window`` (col 0): 1 if within window, 0 otherwise.
    - ``window_position_score`` (col 1): proximity to midpoint [0.0, 1.0].
    - ``discharge_recency_tier`` (col 2): recency tier 0–4.

    Parameters
    ----------
    min_days_post_discharge : int, default=90
        Start of the solicitation window, in days post-discharge (inclusive).
    max_days_post_discharge : int, default=365
        End of the solicitation window, in days post-discharge (inclusive).
    days_since_discharge_col : str, default="days_since_last_discharge"
        Column name containing days since last discharge.
    """

    def __init__(
        self,
        min_days_post_discharge: int = 90,
        max_days_post_discharge: int = 365,
        days_since_discharge_col: str = "days_since_last_discharge",
    ) -> None:
        self.min_days_post_discharge = min_days_post_discharge
        self.max_days_post_discharge = max_days_post_discharge
        self.days_since_discharge_col = days_since_discharge_col

    def fit(self, X, y=None) -> "DischargeToSolicitationWindowTransformer":
        """Fit the transformer (no-op, validates parameters).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : DischargeToSolicitationWindowTransformer
        """
        if self.min_days_post_discharge >= self.max_days_post_discharge:
            raise ValueError(
                f"min_days_post_discharge ({self.min_days_post_discharge}) must be "
                f"strictly less than max_days_post_discharge ({self.max_days_post_discharge})."
            )
        validate_data(self, X, dtype=None, ensure_all_finite="allow-nan", reset=True)
        return self

    def transform(self, X, y=None) -> np.ndarray:
        """Transform X to three columns: in_window, score, recency_tier.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Data with days_since_discharge column or first column as days.

        Returns
        -------
        out : ndarray of shape (n_samples, 3)
            Columns: in_window (0/1), window_position_score [0,1], discharge_recency_tier [0,4].
        """
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame) and self.days_since_discharge_col in X.columns:
            days_raw = X[self.days_since_discharge_col].to_numpy(dtype=float)
        elif isinstance(X, pd.DataFrame):
            days_raw = X.iloc[:, 0].to_numpy(dtype=float)
        else:
            arr = np.asarray(X, dtype=float)
            if arr.ndim == 1:
                days_raw = arr
            else:
                days_raw = arr[:, 0]

        validate_data(self, X, dtype=None, ensure_all_finite="allow-nan", reset=False)

        min_d = float(self.min_days_post_discharge)
        max_d = float(self.max_days_post_discharge)
        midpoint = (min_d + max_d) / 2.0
        half_range = (max_d - min_d) / 2.0

        n = len(days_raw)
        in_window = np.zeros(n, dtype=np.float64)
        window_score = np.zeros(n, dtype=np.float64)
        recency_tier = np.zeros(n, dtype=np.float64)

        for i in range(n):
            d = days_raw[i]
            if np.isnan(d):
                continue

            if min_d <= d <= max_d:
                in_window[i] = 1.0
                window_score[i] = 1.0 - abs(d - midpoint) / half_range

            if d <= 30:
                recency_tier[i] = 4.0
            elif d <= 90:
                recency_tier[i] = 3.0
            elif d <= 180:
                recency_tier[i] = 2.0
            elif d <= 365:
                recency_tier[i] = 1.0
            else:
                recency_tier[i] = 0.0

        return np.column_stack([in_window, window_score, recency_tier])

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        check_is_fitted(self)
        return np.array(
            ["in_solicitation_window", "window_position_score", "discharge_recency_tier"],
            dtype=object,
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        return tags
