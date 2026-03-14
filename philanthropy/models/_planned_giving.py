"""
philanthropy.models._planned_giving
===================================
Models for predicting planned giving (bequest) intent.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.validation import check_is_fitted, validate_data


class PlannedGivingIntentScorer(ClassifierMixin, BaseEstimator):
    """
    Predicts bequest/planned giving intent. Wraps GradientBoostingClassifier
    with CalibratedClassifierCV.

    Exposes `.predict_bequest_intent_score(X)` returning a 0-100 float array.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of boosting stages to perform.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y) -> "PlannedGivingIntentScorer":
        X, y = validate_data(self, X, y, reset=True)
        
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        base_estimator = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self.estimator_ = CalibratedClassifierCV(
            estimator=base_estimator,
            method="sigmoid",
            cv=2,
        )
        self.estimator_.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self.estimator_.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self.estimator_.predict_proba(X)

    def predict_bequest_intent_score(self, X) -> np.ndarray:
        """
        Return the 0-100 float score of bequest intent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        if proba.shape[1] < 2:
            scores = np.zeros(X.shape[0], dtype=float)
        else:
            scores = proba[:, 1] * 100.0
        return scores

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
