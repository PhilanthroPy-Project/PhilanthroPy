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

from typing import Optional

import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data


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

        **Important — leakage guarantee:** pass an *unfitted* imputer here;
        the ``CRMCleaner.fit()`` call will fit it.  Never pre-fit the imputer
        on the full dataset before passing it to ``CRMCleaner``.

    Attributes
    ----------
    feature_names_in_ : list of str
        Column names of ``X`` seen at :meth:`fit` time.
    n_features_in_ : int
        Number of columns in ``X`` at :meth:`fit` time.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from philanthropy.preprocessing import CRMCleaner, WealthScreeningImputer
    >>> X = pd.DataFrame({
    ...     "gift_date":           ["2023-07-01", "2023-11-15"],
    ...     "gift_amount":         [5000.0, np.nan],
    ...     "estimated_net_worth": [np.nan, 1_500_000.0],
    ... })
    >>> imputer = WealthScreeningImputer(
    ...     wealth_cols=["estimated_net_worth"], strategy="median"
    ... )
    >>> cleaner = CRMCleaner(wealth_imputer=imputer)
    >>> out = cleaner.fit_transform(X)
    >>> out["estimated_net_worth"].isna().any()
    False
    """

    def __init__(
        self,
        date_col: str = "gift_date",
        amount_col: str = "gift_amount",
        wealth_imputer=None,
    ) -> None:
        self.date_col = date_col
        self.amount_col = amount_col
        self.wealth_imputer = wealth_imputer

    def fit(self, X: pd.DataFrame, y=None) -> "CRMCleaner":
        """Fit the cleaner (and embedded wealth imputer if provided) to ``X``.

        Parameters
        ----------
        X : pd.DataFrame
            Raw CRM export.
        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        self : CRMCleaner
        """
        X = validate_data(self, X, reset=True)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns.tolist(), dtype=object)
        self.n_features_in_ = X.shape[1]

        # Fit the optional wealth-screening imputer on training data only
        if self.wealth_imputer is not None:
            self.wealth_imputer.fit(X, y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply standardisation and optional wealth imputation.

        Parameters
        ----------
        X : pd.DataFrame
            Raw CRM export (training or held-out).

        Returns
        -------
        X_out : pd.DataFrame
            Cleaned DataFrame with:

            * ``date_col`` coerced to ``datetime64[ns]``.
            * ``amount_col`` coerced to ``float64`` (non-numeric → ``NaN``).
            * Wealth columns imputed if ``wealth_imputer`` was provided.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X_out = X.copy() if hasattr(X, "columns") else pd.DataFrame(X)

        # Coerce date column
        if self.date_col in X_out.columns:
            X_out[self.date_col] = pd.to_datetime(X_out[self.date_col], errors="coerce")

        # Coerce amount column
        if self.amount_col in X_out.columns:
            X_out[self.amount_col] = pd.to_numeric(X_out[self.amount_col], errors="coerce")

        # Apply wealth imputation (uses frozen fill statistics from fit)
        if self.wealth_imputer is not None:
            X_out = self.wealth_imputer.transform(X_out)

        return X_out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.array(self.feature_names_in_, dtype=object)

    def _more_tags(self):
        return {"X_types": ["2darray", "dataframe"]}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags


class FiscalYearTransformer(TransformerMixin, BaseEstimator):
    """Append ``fiscal_year`` and ``fiscal_quarter`` columns to a gift DataFrame.

    Computes fiscal-calendar features from a configurable start month.
    For example, with ``fiscal_year_start=7`` (July), a gift on 2023-07-01
    is assigned ``fiscal_year=2024`` (FY24), while a gift on 2023-06-30
    belongs to ``fiscal_year=2023`` (FY23).

    Parameters
    ----------
    date_col : str, default="gift_date"
        Column containing ISO-8601 gift dates.
    fiscal_year_start : int, default=7
        Month (1–12) that begins the organisation's fiscal year.

    Attributes
    ----------
    feature_names_in_ : list of str
        Columns of ``X`` at :meth:`fit` time.

    Examples
    --------
    >>> import pandas as pd
    >>> from philanthropy.preprocessing import FiscalYearTransformer
    >>> df = pd.DataFrame({"gift_date": ["2023-07-01", "2023-06-30"]})
    >>> t = FiscalYearTransformer(fiscal_year_start=7)
    >>> out = t.fit_transform(df)
    >>> list(out["fiscal_year"])
    [2024, 2023]
    """

    def __init__(
        self,
        date_col: str = "gift_date",
        fiscal_year_start: int = 7,
    ) -> None:
        self.date_col = date_col
        self.fiscal_year_start = fiscal_year_start

    def fit(self, X: pd.DataFrame, y=None) -> "FiscalYearTransformer":
        """Validate parameters and record input schema.

        Parameters
        ----------
        X : pd.DataFrame
            Gift-level DataFrame.
        y : ignored

        Returns
        -------
        self : FiscalYearTransformer
        """
        if not (1 <= self.fiscal_year_start <= 12):
            raise ValueError(
                f"`fiscal_year_start` must be between 1 and 12, "
                f"got {self.fiscal_year_start!r}."
            )
        X = validate_data(self, X, reset=True)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns.tolist(), dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append ``fiscal_year`` and ``fiscal_quarter`` columns.

        Parameters
        ----------
        X : pd.DataFrame
            Gift-level DataFrame containing ``date_col``.

        Returns
        -------
        X_out : pd.DataFrame
            Copy of ``X`` with two new integer columns:

            * ``fiscal_year`` — Calendar year in which the fiscal year *ends*.
            * ``fiscal_quarter`` — Fiscal quarter (1–4) of the gift date.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X_out = X.copy() if hasattr(X, "columns") else pd.DataFrame(X)
        dates = pd.to_datetime(X_out[self.date_col], errors="coerce")

        X_out["fiscal_year"] = dates.apply(
            lambda d: pd.NA if pd.isna(d) else (d.year + 1 if d.month >= self.fiscal_year_start else d.year)
        ).astype("Int64")

        # Compute fiscal quarter
        X_out["fiscal_quarter"] = dates.apply(
            lambda d: pd.NA if pd.isna(d) else (((d.month - self.fiscal_year_start) % 12) // 3 + 1)
        ).astype("Int64")
        return X_out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.array([*self.feature_names_in_, "fiscal_year", "fiscal_quarter"], dtype=object)

    def _more_tags(self):
        return {"X_types": ["2darray", "dataframe"]}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
