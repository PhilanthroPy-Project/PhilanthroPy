"""
philanthropy.models._lapse
==========================
Predictive model for donor lapse using HistGradientBoostingClassifier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils import Tags
from sklearn.utils.validation import check_is_fitted, validate_data


class LapsePredictor(ClassifierMixin, BaseEstimator):
    """
    Predicts whether a donor will lapse in the upcoming fiscal year.
    Uses HistGradientBoostingClassifier with balanced class weights.

    Parameters
    ----------
    n_estimators : int, default=100
        Maximum number of iterations for HistGradientBoostingClassifier.
    max_depth : int or None, default=5
        Maximum depth of trees.
    lapse_window_years : int, default=2
        The time window over which to predict lapsing, in years.
    random_state : int or None, default=None
        Controls the randomness of the estimator.
    """

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 5,
        lapse_window_years: int = 2,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lapse_window_years = lapse_window_years
        self.random_state = random_state

    def fit(self, X, gift_dates, reference_date=None) -> "LapsePredictor":
        """
        Fit the LapsePredictor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        gift_dates : array-like of datetime, shape (n_samples,)
            Date of each donor's most recent gift. Used to compute lapse label.
        reference_date : datetime-like or None
            Scoring cutoff. Defaults to pd.Timestamp.today().

        Returns
        -------
        self : LapsePredictor
            Fitted estimator.
        """
        reference_date = (
            pd.Timestamp(reference_date)
            if reference_date is not None
            else pd.Timestamp.today().normalize()
        )
        cutoff = reference_date - pd.DateOffset(years=self.lapse_window_years)
        y = np.asarray(pd.to_datetime(gift_dates) < cutoff).astype(int)

        X = validate_data(self, X, reset=True)
        
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        self.estimator_ = HistGradientBoostingClassifier(
            max_iter=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            loss="log_loss",
            class_weight="balanced",
        )
        self.estimator_.fit(X, y)
        
        self.reference_date_ = reference_date
        self.cutoff_date_ = cutoff
        
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self.estimator_.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self.estimator_.predict_proba(X)

    def decision_function(self, X) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self.estimator_.decision_function(X)
