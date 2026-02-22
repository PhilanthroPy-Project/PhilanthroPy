"""
philanthropy.preprocessing._solicitation_window
================================================
Optimal solicitation window featurization for annual fund cycles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class SolicitationWindowTransformer(TransformerMixin, BaseEstimator):
    """
    Identifies if a donor is currently in their optimal solicitation window.

    Predictive studies in the annual fund show that the likelihood of a 
    second/repeat gift is highest within a specific window (e.g., 90 to 270 days 
    after the first gift).

    Parameters
    ----------
    days_since_last_gift_col : str, default="days_since_last_gift"
        Column containing the number of days since the donor's last gift.
    min_days : int, default=90
        Start of the optimal solicitation window (inclusive).
    max_days : int, default=270
        End of the optimal solicitation window (inclusive).
    """

    def __init__(
        self,
        days_since_last_gift_col: str = "days_since_last_gift",
        min_days: int = 90,
        max_days: int = 270,
    ) -> None:
        self.days_since_last_gift_col = days_since_last_gift_col
        self.min_days = min_days
        self.max_days = max_days

    def fit(self, X, y=None) -> "SolicitationWindowTransformer":
        """
        Validate data and record input schema.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Donor data.
        y : ignored

        Returns
        -------
        self : SolicitationWindowTransformer
        """
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns.tolist(), dtype=object)

        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=True)
        
        if not hasattr(self, "feature_names_in_"):
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
            
        return self

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def transform(self, X) -> np.ndarray:
        """
        Append 'is_in_optimal_window' feature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Donor data.

        Returns
        -------
        X_out : np.ndarray (float64)
            Original features plus binary 'is_in_optimal_window'.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
        
        if not hasattr(self, "feature_names_in_"):
             raise ValueError("SolicitationWindowTransformer requires named columns in X.")
             
        X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        
        if self.days_since_last_gift_col not in X_df.columns:
            raise ValueError(f"Column {self.days_since_last_gift_col!r} not found in X.")
            
        days = X_df[self.days_since_last_gift_col].astype(float)
        in_window = (days >= self.min_days) & (days <= self.max_days)
        X_df["is_in_optimal_window"] = in_window.astype(float)
        
        # Rule 5: transform() MUST return np.ndarray (float64)
        # Drop non-numeric columns to stay compliant
        X_final = X_df.select_dtypes(include=[np.number])
        return X_final.to_numpy(dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        # We only keep numeric columns in the final output
        # For simplicity in this transformer, we assume all input features were numeric
        # except maybe some PII that we should have dropped.
        # But scikit-learn compliance tests expect us to return names that match the output columns.
        base_features = [f for f in self.feature_names_in_]
        return np.array([*base_features, "is_in_optimal_window"], dtype=object)
