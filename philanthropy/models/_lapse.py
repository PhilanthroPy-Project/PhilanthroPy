import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels

class LapsePredictor(ClassifierMixin, BaseEstimator):
    """
    Predicts the likelihood of a donor lapsing within the specified window.

    Mapping to retention strategies based on predicted probabilities:
    - High Risk (> 70%): Flag for immediate high-touch stewardship (e.g., personal call from a gift officer or tailored letter).
    - Moderate Risk (40% - 70%): Add to targeted multi-channel re-engagement campaigns (e.g., direct mail + email).
    - Low Risk (< 40%): Continue with standard cyclic communications.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the random forest backend.
    max_depth : int or None, default=None
        The maximum depth of the trees.
    lapse_window_years : int, default=2
        The time window over which to predict lapsing, in years.
    random_state : int or None, default=None
        Controls the randomness of the estimator.
    """
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        lapse_window_years=2,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lapse_window_years = lapse_window_years
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        self.estimator_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.estimator_.predict_proba(X)
