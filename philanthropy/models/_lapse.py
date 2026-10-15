"""
philanthropy.models._lapse
==========================
Predictive model for donor lapse using RandomForestClassifier.
"""

from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)
from sklearn.utils import Tags
from sklearn.utils.multiclass import unique_labels

from philanthropy.utils._validation import resolve_columns, resolve_monotonic_cst

_Self = TypeVar("_Self", bound="LapsePredictor")

RETENTION_READ_THRESHOLD = 0.80
"""E.12d: above this training lapse rate, ranking by lapse score stops being
useful (almost everyone lapses, so the list barely beats random); the useful
list is the retention read instead, see :attr:`LapsePredictor.retention_read_`
and :meth:`LapsePredictor.predict_retention_score`."""


class LapsePredictor(ClassifierMixin, BaseEstimator):
    """
    Predicts whether a donor will lapse within a configurable window.
    Uses RandomForestClassifier backend by default.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the RandomForestClassifier. Ignored when
        ``backend="hist_gradient_boosting"``.
    max_depth : int or None, default=None
        Maximum depth of trees. None means nodes expand until pure
        (unbounded for ``RandomForestClassifier``; the backend's own default
        depth limit for ``HistGradientBoostingRegressor``).
    class_weight : dict, "balanced", "balanced_subsample" or None, default=None
        Class weights for imbalanced lapse prediction. Ignored when
        ``backend="hist_gradient_boosting"`` (that backend has no
        ``class_weight`` parameter).
    random_state : int or None, default=None
        Random seed for reproducibility.
    backend : {"random_forest", "hist_gradient_boosting"}, default="random_forest"
        ``"hist_gradient_boosting"`` handles ``NaN`` natively and is faster on
        large files (E.12d); adopt only after it is confirmed to beat "years
        since last gift" at top-10% on one real file without losing on
        another. The default has not changed.
    categorical_features : list of int or str, bool mask, or None, default=None
        Columns the backend should split on as unordered categories (a
        region, constituency or appeal code) instead of as numbers, so they
        need no one-hot step. Positions, a boolean mask, or column names
        when ``X`` is a DataFrame. The columns must hold non-negative
        integer codes below 255, as for ``HistGradientBoostingClassifier``.
        ``None`` treats every column as numeric.
        Used only when ``backend="hist_gradient_boosting"``; the random
        forest has no categorical splits.
    monotonic_cst : array-like of int or dict, default=None
        Monotonic constraint on each feature, passed to the underlying
        :class:`HistGradientBoostingClassifier`: ``1`` means the lapse score never falls as
        that feature rises, ``-1`` that it never rises, ``0`` no
        constraint. A dict sets only the features it names, keyed by
        position or, when ``X`` is a DataFrame, column name
        (``{"total_giving": 1}``). ``None`` applies no constraint
        (current behaviour).
        Used only when ``backend="hist_gradient_boosting"``; the random
        forest ignores it.

    Attributes
    ----------
    training_lapse_rate_ : float
        Fraction of training rows labelled "lapsed", set at fit time.
    retention_read_ : bool
        ``True`` when ``training_lapse_rate_`` exceeds
        :data:`RETENTION_READ_THRESHOLD`: on files where almost everyone
        lapses, the retention read (the 10% *least* likely to lapse, see
        :meth:`predict_retention_score`) is the useful list, not the lapse
        read (E.12d, E.13d "Lapse" fix).
    """

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.classifier_tags.poor_score = True
        return tags

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        class_weight: Any = None,
        random_state: int | None = None,
        backend: str = "random_forest",
        categorical_features: Any = None,
        monotonic_cst: Any = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state
        self.backend = backend
        self.categorical_features = categorical_features
        self.monotonic_cst = monotonic_cst

    def fit(self: _Self, X: Any, y: Any, sample_weight: Any = None) -> _Self:
        """Fit the LapsePredictor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Binary target: 1 = lapse, 0 = no lapse.
        sample_weight : array-like of shape (n_samples,), default=None
            Per-donor weights, passed to the backend (and multiplied there
            with ``class_weight`` on the random forest).

        Returns
        -------
        self : LapsePredictor
        """
        X, y = validate_data(self, X, y, ensure_all_finite="allow-nan", reset=True)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        self.training_lapse_rate_ = float(np.mean(y == 1))
        self.retention_read_ = self.training_lapse_rate_ > RETENTION_READ_THRESHOLD

        if self.backend == "random_forest":
            self.estimator_ = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
        elif self.backend == "hist_gradient_boosting":
            self.estimator_ = HistGradientBoostingClassifier(
                max_depth=self.max_depth,
                categorical_features=resolve_columns(self, self.categorical_features),
                monotonic_cst=resolve_monotonic_cst(self, self.monotonic_cst),
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
        self.estimator_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict binary lapse labels."""
        check_is_fitted(self)
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
        return self.estimator_.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return class probabilities of shape (n_samples, 2)."""
        check_is_fitted(self)
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
        return self.estimator_.predict_proba(X)

    def predict_lapse_score(self, X: Any) -> np.ndarray:
        """Return P(lapse) × 100 rounded to 2 decimal places (0–100 scale)."""
        check_is_fitted(self)
        proba = self.predict_proba(X)
        if proba.shape[1] < 2:
            # Single-class training fold (e.g. no donor lapsed in the window):
            # P(lapse) is 1.0 iff the sole class is the positive one, else 0.0.
            proba_lapse = np.full(proba.shape[0], 1.0 if 1 in self.classes_ else 0.0)
        else:
            # Column 1 is P(class=1), i.e. P(lapse) when classes_ is [0, 1].
            proba_lapse = proba[:, 1]
        return np.round(proba_lapse * 100.0, 2)

    def predict_retention_score(self, X: Any) -> np.ndarray:
        """Return P(retained) x 100, the complement of
        :meth:`predict_lapse_score`.

        Ranking donors by this score, highest first, is the useful list on a
        file where :attr:`retention_read_` is ``True``: ranking by lapse
        score barely beats random when almost everyone lapses, but the tail
        least likely to lapse still beats the rule (E.12d)."""
        return np.round(100.0 - self.predict_lapse_score(X), 2)
