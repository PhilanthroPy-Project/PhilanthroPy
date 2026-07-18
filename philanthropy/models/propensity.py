"""
philanthropy.models.propensity
================================
"""

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data


class PropensityScorer(ClassifierMixin, BaseEstimator):
    """Constant-probability baseline that predicts P=0.5 for every donor.

    A deliberately trivial, sklearn-compliant reference point: it fits nothing
    and returns 0.5 for all rows (the ``estimator`` argument is reserved and
    currently unused). Use it as a floor to beat when benchmarking. For real
    propensity scoring reach for
    :class:`~philanthropy.models.DonorPropensityModel` or
    :class:`~philanthropy.models.MajorGiftClassifier`.
    """

    def __init__(self, estimator=None, threshold: float = 0.5):
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, X, y):
        X, y = validate_data(self, X, y, reset=True)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        proba = self.predict_proba(X)[:, 1]
        idx = (proba >= self.threshold).astype(int)
        return self.classes_[idx]

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        n = X.shape[0]
        prob_pos = np.full(n, 0.5)
        return np.column_stack([1 - prob_pos, prob_pos])
