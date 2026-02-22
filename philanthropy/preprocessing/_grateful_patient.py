"""
philanthropy.preprocessing._grateful_patient
============================================
Clinical intensity and service-line featurization for grateful-patient programs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class GratefulPatientFeaturizer(TransformerMixin, BaseEstimator):
    """
    Featurizer for clinical signals in Grateful Patient programs.

    Hospital advancement shops use clinical data to identify prospects who had 
    high-intensity encounters or were treated in specific high-affinity service 
    lines (e.g., Oncology, Cardiology).

    This transformer maps clinical service lines to numerical capacity weights
    and scales clinical intensity scores.

    Parameters
    ----------
    service_line_col : str, default="service_line"
        Column containing clinical department or service line names.
    intensity_score_col : str, default="intensity_score"
        Column containing raw clinical intensity scores (e.g., length of stay 
        or acuity index).
    capacity_weights : dict of {str: float} or None, default=None
        Mapping of service lines to multipliers. If None, defaults to:
        {"Cardiology": 1.5, "Oncology": 2.0, "Neuroscience": 1.8}.
    """

    def __init__(
        self,
        service_line_col: str = "service_line",
        intensity_score_col: str = "intensity_score",
        capacity_weights: dict[str, float] | None = None,
    ) -> None:
        self.service_line_col = service_line_col
        self.intensity_score_col = intensity_score_col
        self.capacity_weights = capacity_weights

    def fit(self, X, y=None) -> "GratefulPatientFeaturizer":
        """
        Validate data and record input schema.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Clinical encounter data.
        y : ignored

        Returns
        -------
        self : GratefulPatientFeaturizer
        """
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns.tolist(), dtype=object)
        
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=True)
        
        if X.shape[0] == 0:
             raise ValueError("X has zero samples.")

        if not hasattr(self, "feature_names_in_"):
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        
        self.capacity_weights_ = (
            self.capacity_weights 
            if self.capacity_weights is not None 
            else {"Cardiology": 1.5, "Oncology": 2.0, "Neuroscience": 1.8}
        )
        
        return self

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        return tags

    def transform(self, X) -> np.ndarray:
        """
        Transform clinical signals into weighted propensity features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Clinical encounter data.

        Returns
        -------
        X_out : np.ndarray (float64)
            Original features plus 'weighted_clinical_intensity'.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
        
        # We need Column names to find our columns
        if not hasattr(self, "feature_names_in_"):
            # Fallback if fit was called on ndarray without columns
            raise ValueError("GratefulPatientFeaturizer requires named columns in X.")

        X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        
        if self.service_line_col not in X_df.columns:
            raise ValueError(f"Column {self.service_line_col!r} not found in X.")
        if self.intensity_score_col not in X_df.columns:
            raise ValueError(f"Column {self.intensity_score_col!r} not found in X.")

        X_df[self.intensity_score_col] = pd.to_numeric(X_df[self.intensity_score_col], errors="coerce")
        weights = X_df[self.service_line_col].map(self.capacity_weights_).fillna(1.0)
        weighted_intensity = X_df[self.intensity_score_col] * weights
        
        X_df["weighted_clinical_intensity"] = weighted_intensity
        
        # Rule 5: transform() MUST return np.ndarray (float64)
        X_final = X_df.select_dtypes(include=[np.number])
        return X_final.to_numpy(dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        base_features = [f for f in self.feature_names_in_]
        return np.array([*base_features, "weighted_clinical_intensity"], dtype=object)
