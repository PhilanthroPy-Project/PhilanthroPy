"""
philanthropy.preprocessing._transformers
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

from typing import Any

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


def _coerce_currency_to_float(col: pd.Series) -> pd.Series:
    """Parse a gift-amount column to float64, tolerating currency formatting.

    Raiser's Edge NXT and Salesforce NPSP both export amounts as
    ``"$1,000.00"`` by default; a bare ``pd.to_numeric`` treats every such
    value as unparseable and NaNs the whole column. This strips currency
    symbols, thousands separators and parenthesised negatives
    (``"($500.00)"`` -> ``-500.0``) before parsing.
    """
    if pd.api.types.is_numeric_dtype(col):
        return pd.to_numeric(col, errors="coerce").astype("float64")

    had_value = col.notna()
    cleaned = col.astype(str).str.strip()
    cleaned = cleaned.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    cleaned = cleaned.str.replace(r"[^0-9eE.\-]", "", regex=True)
    parsed = pd.to_numeric(cleaned, errors="coerce").where(had_value)

    if had_value.any() and parsed.isna().all():
        raise ValueError(
            f"CRMCleaner: could not parse any value in amount column "
            f"{col.name!r} as a number."
        )
    return parsed.astype("float64")


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
        :meth:`transform`; currency symbols, thousands separators and
        parenthesised negatives (``"$1,000.00"``, ``"($500.00)"``) are
        stripped before parsing.  Values that still don't parse become
        ``NaN``; a column where *nothing* parses raises instead.
    fiscal_year_start : int, default=7
        Month (1–12) that begins the organisation's fiscal year.  Validated in
        :meth:`fit` but **not** used by :meth:`transform`, which only coerces
        ``date_col`` and ``amount_col``.  It is carried here so a cleaner and
        the :class:`FiscalYearTransformer` downstream of it can share one
        fiscal-calendar setting via ``set_params``.

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
    ) -> None:
        self.date_col = date_col
        self.amount_col = amount_col
        self.fiscal_year_start = fiscal_year_start

    def fit(self, X, y=None) -> "CRMCleaner":
        """Validate configuration and input without learning state.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training-set feature matrix.
        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        self : CRMCleaner
            Fitted transformer. This transformer is stateless.

        Raises
        ------
        ValueError
            If ``fiscal_year_start`` is invalid or ``X`` contains complex data.
        """
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
        """Clean CRM dates and amounts using the fitted column configuration.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix (training or held-out).

        Returns
        -------
        X_out : np.ndarray or pd.DataFrame
            Cleaned feature matrix. Returns a DataFrame when the transformer is
            configured with ``set_output(transform="pandas")``, otherwise an
            ndarray.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        ValueError
            If ``X`` contains complex data.
        """
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
            X_df[self.amount_col] = _coerce_currency_to_float(X_df[self.amount_col])

        if _get_pandas_output(self):
            return X_df
        return X_df.to_numpy()

    def get_feature_names_out(self, input_features=None):
        """Return the CRM columns learned during fitting.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Ignored. The output names are the input column names recorded by
            :meth:`fit`.

        Returns
        -------
        feature_names_out : ndarray of str
            The original CRM column names, in input order.

        Raises
        ------
        NotFittedError
            If the transformer has not been fitted.
        """
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

    def __init__(self, date_col: str = "gift_date", fiscal_year_start: int = 7):
        self.date_col = date_col
        self.fiscal_year_start = fiscal_year_start

    def fit(self, X, y=None) -> "FiscalYearTransformer":
        """Validate configuration and input without learning state.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training-set feature matrix.
        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        self : FiscalYearTransformer
            Fitted transformer. This transformer is stateless.

        Raises
        ------
        ValueError
            If ``fiscal_year_start`` is invalid or ``X`` contains complex data.
        """
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
        """Append fiscal year and quarter columns.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix (training or held-out).

        Returns
        -------
        X_out : np.ndarray or pd.DataFrame
            Feature matrix with ``fiscal_year`` and ``fiscal_quarter`` columns
            appended. Returns a DataFrame when the transformer is configured
            with ``set_output(transform="pandas")``, otherwise an ndarray.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        ValueError
            If ``X`` contains complex data.
        """
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
        """Return the two generated fiscal-period feature names.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Ignored because the transformer always emits the same two features.

        Returns
        -------
        feature_names_out : ndarray of str
            ``["fiscal_year", "fiscal_quarter"]``.

        Raises
        ------
        NotFittedError
            If the transformer has not been fitted.
        """
        check_is_fitted(self)
        return np.array(["fiscal_year", "fiscal_quarter"], dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        return tags
