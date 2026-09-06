"""
philanthropy.preprocessing._discharge_window
============================================
Post-discharge solicitation window featurization for grateful patient programs.

Given a clinical discharge date and a solicitation window (in days), determines
whether each gift falls within N days after discharge. Produces in_window
and window_position_score columns.

The window's lower bound is an ethical cooling-off floor, so the timing score
decays from the floor outwards rather than peaking at the window midpoint. See
``DischargeToSolicitationWindowTransformer.window_shape``.
"""

from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags
from sklearn.utils.validation import check_is_fitted, validate_data

_Self = TypeVar("_Self", bound="DischargeToSolicitationWindowTransformer")


class DischargeToSolicitationWindowTransformer(TransformerMixin, BaseEstimator):
    """Flag donors in the clinical fundraising post-discharge solicitation window.

    This transformer outputs two features:
    - ``in_solicitation_window`` (col 0): 1 if within window, 0 otherwise.
    - ``window_position_score`` (col 1): strength of the timing signal, in
      [0.0, 1.0], or ``NaN`` when the days-since-discharge input is missing.

    ``min_days_post_discharge`` is an **ethical cooling-off floor**, not the
    start of a ramp: soliciting a patient the week after discharge is the thing
    the floor exists to prevent. So the score is highest immediately after the
    floor is cleared and decays with elapsed time, which is also how
    grateful-patient propensity is understood to behave. That is
    ``window_shape="decay"``, the default.

    Parameters
    ----------
    min_days_post_discharge : int, default=90
        Start of the solicitation window, in days post-discharge (inclusive).
    max_days_post_discharge : int, default=365
        End of the solicitation window, in days post-discharge (inclusive).
    days_since_discharge_col : str, default="days_since_last_discharge"
        Column name containing days since last discharge.
    window_shape : {"decay", "triangle"}, default="decay"
        Shape of ``window_position_score`` inside the window.

        ``"decay"``
            Linear decay from 1.0 at ``min_days_post_discharge`` to 0.0 at
            ``max_days_post_discharge``. Monotone non-increasing.
        ``"triangle"``
            The legacy symmetric triangle, peaking at the window midpoint. It
            treats the ethical floor as a propensity minimum: with the default
            window, day 91 and day 364 both score about 0.007 while day 227
            scores 1.0. Kept only to reproduce results computed before the
            default changed.

    Notes
    -----
    A missing days-since-discharge value yields ``in_solicitation_window=0``
    and ``window_position_score=NaN``, so "no discharge on record" is
    distinguishable downstream from "discharged, but outside the window", which
    scores a hard 0.0. Both estimators that consume this column handle NaN
    natively.
    """

    _WINDOW_SHAPES = ("decay", "triangle")

    def __init__(
        self,
        min_days_post_discharge: int = 90,
        max_days_post_discharge: int = 365,
        days_since_discharge_col: str = "days_since_last_discharge",
        window_shape: str = "decay",
    ) -> None:
        self.min_days_post_discharge = min_days_post_discharge
        self.max_days_post_discharge = max_days_post_discharge
        self.days_since_discharge_col = days_since_discharge_col
        self.window_shape = window_shape

    def fit(self: _Self, X: Any, y: Any = None) -> _Self:
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
        if self.window_shape not in self._WINDOW_SHAPES:
            raise ValueError(
                f"window_shape must be one of {self._WINDOW_SHAPES}, got "
                f"{self.window_shape!r}."
            )
        validate_data(self, X, dtype=None, ensure_all_finite="allow-nan", reset=True)
        return self

    def transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Transform X to two columns: in_window, window_position_score.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            A DataFrame must carry ``days_since_discharge_col``; a bare ndarray
            is read positionally (first column, or the array itself if 1-D).

        Returns
        -------
        out : ndarray of shape (n_samples, 2)
            Columns: in_window (0/1), window_position_score [0,1].

        Raises
        ------
        ValueError
            If X is a DataFrame without ``days_since_discharge_col``.
        """
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame) and self.days_since_discharge_col in X.columns:
            days_raw = X[self.days_since_discharge_col].to_numpy(dtype=float)
        elif isinstance(X, pd.DataFrame):
            raise ValueError(
                f"{self.days_since_discharge_col!r} not found in X; columns are "
                f"{list(X.columns)}. Route this transformer with a ColumnTransformer "
                f"so it receives the days-since-discharge column, or set "
                f"days_since_discharge_col to the correct name."
            )
        else:
            arr = np.asarray(X, dtype=float)
            if arr.ndim == 1:
                days_raw = arr
            else:
                days_raw = arr[:, 0]

        validate_data(self, X, dtype=None, ensure_all_finite="allow-nan", reset=False)

        min_d = float(self.min_days_post_discharge)
        max_d = float(self.max_days_post_discharge)
        span = max_d - min_d

        days = np.asarray(days_raw, dtype=np.float64)
        missing = np.isnan(days)
        inside = ~missing & (days >= min_d) & (days <= max_d)

        in_window = inside.astype(np.float64)

        if self.window_shape == "triangle":
            midpoint = (min_d + max_d) / 2.0
            raw = 1.0 - np.abs(np.where(missing, min_d, days) - midpoint) / (span / 2.0)
        else:
            raw = 1.0 - (np.where(missing, min_d, days) - min_d) / span

        # Out of window scores a hard 0; a missing input stays NaN so the two
        # are distinguishable downstream.
        window_score = np.where(inside, np.clip(raw, 0.0, 1.0), 0.0)
        window_score = np.where(missing, np.nan, window_score)

        return np.column_stack([in_window, window_score])

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Get output feature names."""
        check_is_fitted(self)
        return np.array(
            ["in_solicitation_window", "window_position_score"],
            dtype=object,
        )

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        return tags
