"""
philanthropy.preprocessing._planned_giving
==========================================
Legacy and planned-giving signal featurization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class PlannedGivingSignalTransformer(TransformerMixin, BaseEstimator):
    """
    Computes a legacy/planned-giving likelihood signal.

    Traditional planned-giving models (BeQuest propensity) rely on two core
    signals: loyalty (years of active giving) and identified capacity (total
    gift count or amount).

    Parameters
    ----------
    age_col : str, default="age"
        Column containing donor age.
    years_active_col : str, default="years_active"
        Column containing number of years donor has been active.
    total_gifts_col : str, default="total_gifts"
        Column containing total number of gifts made.
    age_weight : float, default=0.7
        Weight applied to the loyalty signal (years_active).
    capacity_multiplier : float, default=0.3
        Multiplier applied to the capacity signal (total_gifts).
    """

    def __init__(
        self,
        age_col: str = "age",
        years_active_col: str = "years_active",
        total_gifts_col: str = "total_gifts",
        age_weight: float = 0.7,
        capacity_multiplier: float = 0.3,
    ) -> None:
        self.age_col = age_col
        self.years_active_col = years_active_col
        self.total_gifts_col = total_gifts_col
        self.age_weight = age_weight
        self.capacity_multiplier = capacity_multiplier

    def fit(self, X, y=None) -> "PlannedGivingSignalTransformer":
        """
        Validate data and record input schema.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Donor data.
        y : ignored

        Returns
        -------
        self : PlannedGivingSignalTransformer
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
        Append 'planned_giving_score'.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Donor data.

        Returns
        -------
        X_out : np.ndarray (float64)
            Original features plus 'planned_giving_score'.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
        
        if not hasattr(self, "feature_names_in_"):
             raise ValueError("PlannedGivingSignalTransformer requires named columns in X.")
             
        X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        
        for col in [self.years_active_col, self.total_gifts_col]:
            if col not in X_df.columns:
                raise ValueError(f"Column {col!r} not found in X.")
                
        # Logic from prompt: (X[years_active] * age_weight) + (X[total_gifts] * capacity_multiplier)
        score = (
            X_df[self.years_active_col].astype(float) * self.age_weight + 
            X_df[self.total_gifts_col].astype(float) * self.capacity_multiplier
        )
        
        X_df["planned_giving_score"] = score
        
        # Rule 5: transform() MUST return np.ndarray (float64)
        X_final = X_df.select_dtypes(include=[np.number])
        return X_final.to_numpy(dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        base_features = [f for f in self.feature_names_in_]
        return np.array([*base_features, "planned_giving_score"], dtype=object)
