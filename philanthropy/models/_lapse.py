"""
philanthropy.models._lapse
==========================
Predictive model for donor lapse using RandomForestClassifier.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils import Tags


class LapsePredictor(ClassifierMixin, BaseEstimator):
    """
    Predicts whether a donor will lapse within a configurable window.
    Uses RandomForestClassifier backend.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the RandomForestClassifier.
    max_depth : int or None, default=None
        Maximum depth of trees. None means nodes expand until pure.
    class_weight : dict, "balanced", "balanced_subsample" or None, default=None
        Class weights for imbalanced lapse prediction.
    random_state : int or None, default=None
        Random seed for reproducibility.
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
        class_weight=None,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y) -> "LapsePredictor":
        """Fit the LapsePredictor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Binary target: 1 = lapse, 0 = no lapse.

        Returns
        -------
        self : LapsePredictor
        """
        X, y = validate_data(self, X, y, ensure_all_finite="allow-nan", reset=True)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        self.estimator_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.estimator_.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        """Predict binary lapse labels."""
        check_is_fitted(self)
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
        return self.estimator_.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities of shape (n_samples, 2)."""
        check_is_fitted(self)
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
        return self.estimator_.predict_proba(X)

    def predict_lapse_score(self, X) -> np.ndarray:
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
