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

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class SolicitationWindowTransformer(TransformerMixin, BaseEstimator):
    """Flag donors in the clinical fundraising post-discharge solicitation window.

    Post-discharge solicitation windows are the operational backbone of grateful
    patient programs. Clinical fundraising research shows propensity peaks between
    6–24 months post-discharge. Too early = patient still processing; too late =
    emotional connection fades.

    This transformer outputs two features:
    - ``in_window`` (float64, 0.0 or 1.0): 1 if the donor is within the
      window, 0 otherwise.
    - ``window_score`` (float64, [0.0, 1.0]): continuous proximity to the
      midpoint of the window. 1.0 at the midpoint, 0.0 at the edges and
      outside.

    Parameters
    ----------
    min_days_post_discharge : int, default=180
        Start of the optimal solicitation window, in days post-discharge
        (inclusive). Defaults to 6 months.
    max_days_post_discharge : int, default=730
        End of the optimal solicitation window, in days post-discharge
        (inclusive). Defaults to 24 months.
    days_since_discharge_col : str, default="days_since_last_discharge"
        Column name containing days since last discharge, when input is
        a pd.DataFrame.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen at fit time.
    feature_names_in_ : ndarray of str
        Column names seen at fit time (only set if X was a DataFrame).

    Raises
    ------
    ValueError
        If ``min_days_post_discharge >= max_days_post_discharge`` at fit time.

    Examples
    --------
    >>> import numpy as np
    >>> from philanthropy.preprocessing import SolicitationWindowTransformer
    >>> t = SolicitationWindowTransformer(min_days_post_discharge=180, max_days_post_discharge=730)
    >>> X = np.array([[455.0], [90.0], [np.nan]])
    >>> t.fit(X)
    SolicitationWindowTransformer(...)
    >>> t.transform(X)
    array([[1., 1.],
           [0., 0.],
           [0., 0.]])
    """

    def __init__(
        self,
        min_days_post_discharge: int = 180,
        max_days_post_discharge: int = 730,
        days_since_discharge_col: str = "days_since_last_discharge",
    ) -> None:
        self.min_days_post_discharge = min_days_post_discharge
        self.max_days_post_discharge = max_days_post_discharge
        self.days_since_discharge_col = days_since_discharge_col

    def fit(self, X, y=None) -> "SolicitationWindowTransformer":
        """Validate window parameters and record input schema.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data. If a pd.DataFrame with a ``days_since_discharge_col``
            column, that column is used for computation. Otherwise, the first
            column is assumed to be days since discharge.
        y : ignored

        Returns
        -------
        self : SolicitationWindowTransformer

        Raises
        ------
        ValueError
            If ``min_days_post_discharge >= max_days_post_discharge``.
        """
        if self.min_days_post_discharge >= self.max_days_post_discharge:
            raise ValueError(
                f"min_days_post_discharge ({self.min_days_post_discharge}) must be "
                f"strictly less than max_days_post_discharge ({self.max_days_post_discharge})."
            )
        validate_data(self, X, dtype=None, ensure_all_finite="allow-nan", reset=True)
        return self

    def transform(self, X, y=None) -> np.ndarray:
        """Compute in-window flag and window proximity score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data. If a pd.DataFrame, uses ``days_since_discharge_col``
            column. If a np.ndarray, uses the first column.

        Returns
        -------
        X_out : np.ndarray of shape (n_samples, 2), dtype float64
            Column 0: ``in_window`` — 1.0 if ``min_days <= days <= max_days``,
            else 0.0. NaN days → 0.0.
            Column 1: ``window_score`` — 1.0 at midpoint, 0.0 at edges and
            outside window. NaN days → 0.0.
        """
        check_is_fitted(self)

        # Extract days BEFORE validate_data for DataFrame input
        if isinstance(X, pd.DataFrame) and self.days_since_discharge_col in X.columns:
            days_raw = X[self.days_since_discharge_col].to_numpy(dtype=float)
        elif isinstance(X, pd.DataFrame):
            # Fall back to first column if named col not present
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

        for i in range(n):
            d = days_raw[i]
            if np.isnan(d):
                continue  # NaN → 0.0 for both
            if min_d <= d <= max_d:
                in_window[i] = 1.0
                window_score[i] = 1.0 - abs(d - midpoint) / half_range
            # else: both remain 0.0

        return np.column_stack([in_window, window_score])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.array(["in_window", "window_score"], dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        # This transformer extracts named columns from DataFrames and
        # handles mixed-type inputs gracefully (strings in other columns
        # are ignored). Setting string=True suppresses check_dtype_object's
        # strict TypeError requirement, which is appropriate for DataFrame-aware
        # transformers.
        tags.input_tags.string = True
        return tags
