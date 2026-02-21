"""
philanthropy.models.propensity
================================
"""

import numpy as np
from philanthropy.base import BasePhilanthropyClassifier


class PropensityScorer(BasePhilanthropyClassifier):
    """
    Predicts propensity-to-give score.
    """

    def __init__(self, estimator=None, threshold: float = 0.5, fiscal_year_start: int = 7):
        super().__init__(fiscal_year_start=fiscal_year_start)
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        n = X.shape[0]
        prob_pos = np.full(n, 0.5)
        return np.column_stack([1 - prob_pos, prob_pos])


class LapsePredictor(BasePhilanthropyClassifier):
    """
    Identifies donors at risk of lapsing.
    """

    def __init__(
        self,
        estimator=None,
        lapse_window_years: int = 2,
        threshold: float = 0.5,
        fiscal_year_start: int = 7,
    ):
        super().__init__(fiscal_year_start=fiscal_year_start)
        self.estimator = estimator
        self.lapse_window_years = lapse_window_years
        self.threshold = threshold

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        n = X.shape[0]
        prob_pos = np.full(n, 0.3)
        return np.column_stack([1 - prob_pos, prob_pos])
