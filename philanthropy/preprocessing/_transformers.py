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

from typing import Any, TypeVar

import warnings

import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import Tags
from sklearn.utils.validation import check_is_fitted, validate_data
from philanthropy.utils._validation import validate_fiscal_year_start

_SelfC = TypeVar("_SelfC", bound="CRMCleaner")
_SelfF = TypeVar("_SelfF", bound="FiscalYearTransformer")


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


def _validate_X(estimator: BaseEstimator, X: Any, reset: bool) -> np.ndarray:
    """Validate data while allowing mixed types by falling back to object dtype."""
    try:
        return validate_data(estimator, X, dtype=None, ensure_all_finite="allow-nan", reset=reset)
    except Exception as e:
        if "Complex data not supported" in str(e):
            raise
        X_val = X.astype(object) if hasattr(X, "astype") else X
        return validate_data(estimator, X_val, dtype=None, ensure_all_finite="allow-nan", reset=reset)


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

    # Complex values (e.g. np.complex128 cells that survive an object-cast
    # retry, as when a formula column round-trips through Excel/openpyxl)
    # must never reach the string path: str(3+4j) is "(3+4j)", which matches
    # the parenthesised-negative rule and corrupts the cell into -34.0
    # (#129). NaN + warning, consistent with "values that still don't parse
    # become NaN".
    had_value = col.notna()

    if col.dtype == object:
        complex_mask = col.map(lambda v: isinstance(v, complex)).fillna(False)
        if complex_mask.any():
            warnings.warn(
                f"CRMCleaner: {int(complex_mask.sum())} complex value(s) in "
                f"amount column {col.name!r} cannot be parsed as currency; "
                f"they became NaN.",
                stacklevel=2,
            )
            col = col.mask(complex_mask)
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

    def fit(self: _SelfC, X: Any, y: Any = None) -> _SelfC:
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
        _validate_X(self, X, reset=True)
        return self

    def transform(self, X: Any) -> np.ndarray | pd.DataFrame:
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
        X_arr = _validate_X(self, X, reset=False)

        X_df = pd.DataFrame(X_arr, columns=getattr(self, "feature_names_in_", None)).copy()
        
        if self.date_col in X_df.columns:
            X_df[self.date_col] = pd.to_datetime(X_df[self.date_col], errors="coerce")
        if self.amount_col in X_df.columns:
            X_df[self.amount_col] = _coerce_currency_to_float(X_df[self.amount_col])

        if _get_pandas_output(self):
            return X_df
        return X_df.to_numpy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
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

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        return tags


class FiscalYearTransformer(TransformerMixin, BaseEstimator):
    """Derive organisation-specific fiscal year and quarter from a date column.

    ``transform`` **replaces** the input with exactly two columns,
    ``fiscal_year`` and ``fiscal_quarter``; it does not append them to the
    input. This is what ``get_feature_names_out`` has always reported. To keep
    the original columns alongside the fiscal ones, wrap this transformer in a
    :class:`~sklearn.compose.ColumnTransformer` with ``remainder="passthrough"``
    or a :class:`~sklearn.pipeline.FeatureUnion`.
    """

    def __init__(self, date_col: str = "gift_date", fiscal_year_start: int = 7) -> None:
        self.date_col = date_col
        self.fiscal_year_start = fiscal_year_start

    def fit(self: _SelfF, X: Any, y: Any = None) -> _SelfF:
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
        _validate_X(self, X, reset=True)
        return self

    def transform(self, X: Any) -> np.ndarray | pd.DataFrame:
        """Return the fiscal year and quarter derived from ``date_col``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix (training or held-out).

        Returns
        -------
        X_out : np.ndarray or pd.DataFrame of shape (n_samples, 2)
            Exactly two columns, ``fiscal_year`` and ``fiscal_quarter``. The
            input columns are **not** carried through; see the class docstring
            for how to keep them. Both columns are ``NaN`` for rows whose date
            does not parse, and for every row when ``date_col`` is absent from
            ``X``. Returns a DataFrame when the transformer is configured with
            ``set_output(transform="pandas")``, otherwise an ndarray.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        ValueError
            If ``X`` contains complex data.
        """
        check_is_fitted(self)
        X_arr = _validate_X(self, X, reset=False)
        
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

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
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

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        return tags
