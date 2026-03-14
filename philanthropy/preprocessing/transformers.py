"""
philanthropy.preprocessing.transformers
========================================
CRM data cleaning and Fiscal Year–aware feature engineering transformers.

``CRMCleaner`` is the recommended first stage of any PhilanthroPy preprocessing
pipeline.  It standardises raw CRM exports. If you need to impute third-party 
wealth-screening data, use a ``Pipeline`` to chain ``CRMCleaner`` with
:class:`~philanthropy.preprocessing.WealthScreeningImputer`.

``FiscalYearTransformer`` enriches a gift-level DataFrame with numeric
``fiscal_year`` and ``fiscal_quarter`` columns computed from a configurable
fiscal-calendar start month.
"""

from __future__ import annotations

import numbers
import warnings
from typing import Optional, Any

import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils._param_validation import validate_params, Interval
from philanthropy.utils._validation import validate_fiscal_year_start


def _get_pandas_output(estimator: Any) -> bool:
    """Defensively check if scikit-learn is configured to output DataFrames."""
    try:
        # Check for modern sklearn config
        from sklearn.utils._set_output import _get_output_config
        config = _get_output_config("transform", estimator)
        if config and config.get("dense") == "pandas":
            return True
    except (ImportError, AttributeError):
        pass
    
    # Fallback to direct attribute check (robust to different sklearn versions)
    config = getattr(estimator, "_sklearn_output_config", {})
    if isinstance(config, dict):
        trans = config.get("transform", {})
        if trans == "pandas":
            return True
        if isinstance(trans, dict) and trans.get("dense") == "pandas":
            return True
    return False


class CRMCleaner(TransformerMixin, BaseEstimator):
    """Standardise raw CRM exports.

    ``CRMCleaner`` performs lightweight, defensive cleaning of CRM datasets
    exported from systems such as Salesforce NPSP, Raiser's Edge NXT, or
    Ellucian Advance. It is designed to be chained in a `sklearn.pipeline.Pipeline`
    along with `WealthScreeningImputer` to handle missing wealth values.

    Parameters
    ----------
    date_col : str, default="gift_date"
        Column containing ISO-8601 gift dates.  Parsed to ``datetime64``
        during :meth:`transform`.
    amount_col : str, default="gift_amount"
        Column containing raw gift amounts.  Forced to ``float64`` during
        :meth:`transform`; non-numeric values become ``NaN``.
    fiscal_year_start : int, default=7
        Month (1–12) that begins the organisation's fiscal year.

    Attributes
    ----------
    feature_names_in_ : list of str
        Column names of ``X`` seen at :meth:`fit` time.
    n_features_in_ : int
        Number of columns in ``X`` at :meth:`fit` time.
    """

    @validate_params(
        {
            "date_col": [str],
            "amount_col": [str],
            "fiscal_year_start": [Interval(numbers.Integral, 1, 12, closed="both")],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        date_col: str = "gift_date",
        amount_col: str = "gift_amount",
        fiscal_year_start: int = 7,
    ) -> None:
        self.date_col = date_col
        self.amount_col = amount_col
        self.fiscal_year_start = fiscal_year_start

    def fit(self, X, y=None) -> "CRMCleaner":
        validate_fiscal_year_start(self.fiscal_year_start)
        
        # Try standard validation, fallback to object for mixed-type DataFrames or promotion errors
        try:
            X_validated = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan", reset=True)
        except Exception as e:
            if "Complex data not supported" in str(e):
                raise
            X_val = X.astype(object) if hasattr(X, "astype") else X
            X_validated = validate_data(self, X_val, dtype=None, ensure_all_finite="allow-nan", reset=True)
            
        if np.iscomplexobj(X_validated):
            raise ValueError("Complex data not supported")
        
        return self

    def transform(self, X) -> np.ndarray | pd.DataFrame:
        check_is_fitted(self)
        try:
            X_arr = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan", reset=False)
        except Exception as e:
            if "Complex data not supported" in str(e):
                raise
            X_val = X.astype(object) if hasattr(X, "astype") else X
            X_arr = validate_data(self, X_val, dtype=None, ensure_all_finite="allow-nan", reset=False)

        if np.iscomplexobj(X_arr):
            raise ValueError("Complex data not supported")

        X_df = pd.DataFrame(X_arr, columns=getattr(self, "feature_names_in_", None)).copy()
        
        if self.date_col in X_df.columns:
            X_df[self.date_col] = pd.to_datetime(X_df[self.date_col], errors="coerce")
        if self.amount_col in X_df.columns:
            X_df[self.amount_col] = pd.to_numeric(X_df[self.amount_col], errors="coerce")

        if _get_pandas_output(self):
            return X_df
        return X_df.to_numpy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        names = list(self.feature_names_in_)
        return np.array(names, dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        return tags


class FiscalYearTransformer(TransformerMixin, BaseEstimator):
    """Append Organisation-specific Fiscal Year and Quarter to dates."""

    @validate_params(
        {
            "date_col": [str],
            "fiscal_year_start": [Interval(numbers.Integral, 1, 12, closed="both")],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(self, date_col: str = "gift_date", fiscal_year_start: int = 7):
        self.date_col = date_col
        self.fiscal_year_start = fiscal_year_start

    def fit(self, X, y=None) -> "FiscalYearTransformer":
        validate_fiscal_year_start(self.fiscal_year_start)
        try:
            X_validated = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan", reset=True)
        except Exception as e:
            if "Complex data not supported" in str(e):
                raise
            X_val = X.astype(object) if hasattr(X, "astype") else X
            X_validated = validate_data(self, X_val, dtype=None, ensure_all_finite="allow-nan", reset=True)
            
        if np.iscomplexobj(X_validated):
            raise ValueError("Complex data not supported")
        return self

    def transform(self, X) -> np.ndarray | pd.DataFrame:
        check_is_fitted(self)
        try:
            X_arr = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan", reset=False)
        except Exception as e:
            if "Complex data not supported" in str(e):
                raise
            X_val = X.astype(object) if hasattr(X, "astype") else X
            X_arr = validate_data(self, X_val, dtype=None, ensure_all_finite="allow-nan", reset=False)
        
        if np.iscomplexobj(X_arr):
            raise ValueError("Complex data not supported")
        
        X_df = pd.DataFrame(X_arr, columns=getattr(self, "feature_names_in_", None)).copy()
        
        if self.date_col not in X_df.columns:
            X_df["fiscal_year"] = np.nan
            X_df["fiscal_quarter"] = np.nan
        else:
            dates = pd.to_datetime(X_df[self.date_col], errors="coerce")
            X_df["fiscal_year"] = dates.apply(
                lambda d: np.nan if pd.isna(d) else float(d.year + 1 if d.month >= self.fiscal_year_start else d.year)
            )
            X_df["fiscal_quarter"] = dates.apply(
                lambda d: np.nan if pd.isna(d) else float(((d.month - self.fiscal_year_start) % 12) // 3 + 1)
            )
        
        out_df = pd.DataFrame({
            "fiscal_year": pd.to_numeric(X_df["fiscal_year"], errors="coerce").astype(float),
            "fiscal_quarter": pd.to_numeric(X_df["fiscal_quarter"], errors="coerce").astype(float)
        })
        
        if _get_pandas_output(self):
            return out_df
        return out_df.to_numpy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.array(["fiscal_year", "fiscal_quarter"], dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        return tags
