"""
philanthropy.preprocessing._solicitation_window
================================================
Post-discharge solicitation window featurization for grateful patient programs.

Post-discharge solicitation windows are the operational backbone of grateful
patient programs. Clinical fundraising research shows propensity peaks between
6–24 months post-discharge. Too early = patient still processing; too late =
emotional connection fades. This transformer flags in-window donors AND emits
a continuous proximity score (1.0 at the window midpoint, 0.0 at edges).
"""

from __future__ import annotations

import numbers
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils._param_validation import validate_params, Interval


class DischargeToSolicitationWindowTransformer(TransformerMixin, BaseEstimator):
    """Flag donors in the clinical fundraising post-discharge solicitation window.

    This transformer outputs three features:
    - ``in_solicitation_window`` (uint8, 0 or 1): 1 if the donor is within the
      window, 0 otherwise.
    - ``window_position_score`` (float64, [0.0, 1.0]): continuous proximity to the
      midpoint of the window. 1.0 at the midpoint, 0.0 at the edges and
      outside.
    - ``discharge_recency_tier`` (int, 0 to 4): Categorizes recency.

    Parameters
    ----------
    min_days_post_discharge : int, default=90
        Start of the optimal solicitation window, in days post-discharge
        (inclusive). Defaults to 90 days.
    max_days_post_discharge : int, default=365
        End of the optimal solicitation window, in days post-discharge
        (inclusive). Defaults to 365 days.
    days_since_discharge_col : str, default="days_since_last_discharge"
        Column name containing days since last discharge.
    """

    @validate_params(
        {
            "min_days_post_discharge": [Interval(numbers.Integral, 0, None, closed="left")],
            "max_days_post_discharge": [Interval(numbers.Integral, 1, None, closed="left")],
            "days_since_discharge_col": [str],
        },
        prefer_skip_nested_validation=True,
    )
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
        if self.min_days_post_discharge >= self.max_days_post_discharge:
            raise ValueError(
                f"min_days_post_discharge ({self.min_days_post_discharge}) must be "
                f"strictly less than max_days_post_discharge ({self.max_days_post_discharge})."
            )
        validate_data(self, X, dtype=None, ensure_all_finite="allow-nan", reset=True)
        return self

    def transform(self, X, y=None) -> np.ndarray:
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
        in_window = np.zeros(n, dtype=np.uint8)
        window_score = np.zeros(n, dtype=np.float64)
        recency_tier = np.zeros(n, dtype=int)

        for i in range(n):
            d = days_raw[i]
            if np.isnan(d):
                continue

            if min_d <= d <= max_d:
                in_window[i] = 1
                window_score[i] = 1.0 - abs(d - midpoint) / half_range

            if d <= 30:
                recency_tier[i] = 4
            elif d <= 90:
                recency_tier[i] = 3
            elif d <= 180:
                recency_tier[i] = 2
            elif d <= 365:
                recency_tier[i] = 1
            else:
                recency_tier[i] = 0

        # convert to float64 so that sklearn check_estimator handles outputs uniformly
        return np.column_stack([
            in_window.astype(np.float64), 
            window_score, 
            recency_tier.astype(np.float64)
        ])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.array(["in_solicitation_window", "window_position_score", "discharge_recency_tier"], dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        return tags
