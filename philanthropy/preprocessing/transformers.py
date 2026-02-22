"""
philanthropy.preprocessing.transformers
========================================
CRM data cleaning and Fiscal Year–aware feature engineering transformers.

``CRMCleaner`` is the recommended first stage of any PhilanthroPy preprocessing
pipeline.  It standardises raw CRM exports and, optionally, delegates leakage-safe
imputation of missing third-party wealth-screening values to an embedded
:class:`~philanthropy.preprocessing.WealthScreeningImputer`.

``FiscalYearTransformer`` enriches a gift-level DataFrame with numeric
``fiscal_year`` and ``fiscal_quarter`` columns computed from a configurable
fiscal-calendar start month.
"""

from __future__ import annotations

import warnings
from typing import Optional, Any

import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data
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
    """Standardise raw CRM exports and optionally impute wealth-screening data.

    ``CRMCleaner`` performs lightweight, defensive cleaning of CRM datasets
    exported from systems such as Salesforce NPSP, Raiser's Edge NXT, or
    Ellucian Advance.  When a ``WealthScreeningImputer`` instance is provided
    via the ``wealth_imputer`` parameter, it is fitted on the training data
    during :meth:`fit` and applied during :meth:`transform`, ensuring that no
    imputation statistics from held-out rows contaminate training distributions.

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
    wealth_imputer : WealthScreeningImputer or None, default=None
        An **unfitted** :class:`~philanthropy.preprocessing.WealthScreeningImputer`
        instance.  When provided, :meth:`fit` calls ``wealth_imputer.fit(X)``
        and :meth:`transform` calls ``wealth_imputer.transform(X)`` so that
        fill statistics are learned exclusively from training data.

    Attributes
    ----------
    feature_names_in_ : list of str
        Column names of ``X`` seen at :meth:`fit` time.
    n_features_in_ : int
        Number of columns in ``X`` at :meth:`fit` time.
    """

    def __init__(
        self,
        date_col: str = "gift_date",
        amount_col: str = "gift_amount",
        fiscal_year_start: int = 7,
        wealth_imputer=None,
    ) -> None:
        self.date_col = date_col
        self.amount_col = amount_col
        self.fiscal_year_start = fiscal_year_start
        self.wealth_imputer = wealth_imputer

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
        
        if self.wealth_imputer is not None:
            # Wealth imputer usually expects numeric data
            X_df = pd.DataFrame(X, columns=getattr(self, "feature_names_in_", None))
            # Coerce columns needed for wealth imputer to numeric/datetime
            if self.date_col in X_df.columns:
                X_df[self.date_col] = pd.to_datetime(X_df[self.date_col], errors="coerce")
            if self.amount_col in X_df.columns:
                X_df[self.amount_col] = pd.to_numeric(X_df[self.amount_col], errors="coerce")
            if hasattr(self.wealth_imputer, "wealth_cols") and self.wealth_imputer.wealth_cols:
                for col in self.wealth_imputer.wealth_cols:
                    if col in X_df.columns:
                        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
            X_num = X_df.select_dtypes(include=[np.number])
            self.wealth_imputer.fit(X_num, y)
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

        if self.wealth_imputer is not None:
            if hasattr(self.wealth_imputer, "wealth_cols") and self.wealth_imputer.wealth_cols:
                for col in self.wealth_imputer.wealth_cols:
                    if col in X_df.columns:
                        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
            X_num = X_df.select_dtypes(include=[np.number])
            X_imp = self.wealth_imputer.transform(X_num)
            imp_cols = self.wealth_imputer.get_feature_names_out()
            if hasattr(X_imp, "columns"):
                for col in imp_cols:
                    X_df[col] = X_imp[col]
            else:
                for i, col in enumerate(imp_cols):
                    X_df[col] = X_imp[:, i]

        if _get_pandas_output(self):
            return X_df
        return X_df.to_numpy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        names = list(self.feature_names_in_)
        if self.wealth_imputer is not None:
            imp_names = self.wealth_imputer.get_feature_names_out()
            for name in imp_names:
                if name not in names:
                    names.append(name)
        return np.array(names, dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        return tags


class FiscalYearTransformer(TransformerMixin, BaseEstimator):
    """Append Organisation-specific Fiscal Year and Quarter to dates."""

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
        
        X_df["fiscal_year"] = pd.to_numeric(X_df["fiscal_year"], errors="coerce").astype(float)
        X_df["fiscal_quarter"] = pd.to_numeric(X_df["fiscal_quarter"], errors="coerce").astype(float)
        
        if _get_pandas_output(self):
            return X_df
        return X_df.to_numpy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        base = getattr(self, "feature_names_in_", None)
        if base is None:
            base = [f"x{i}" for i in range(self.n_features_in_)]
        return np.array(list(base) + ["fiscal_year", "fiscal_quarter"], dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        return tags
