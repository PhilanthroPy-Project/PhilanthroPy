"""
philanthropy.preprocessing._wealth
====================================
Leakage-safe imputation for third-party wealth-screening vendor data.

Academic medical centre (AMC) advancement shops routinely purchase wealth
screenings from vendors such as DonorSearch, iWave, or Wealth Engine.  These
datasets contain high-value features (Estimated Net Worth, Real Estate Equity,
Stock Holdings, etc.) but are *notoriously* incomplete — coverage typically
ranges from 30 % to 70 % of a prospect pool.  Naively imputing with
population-level statistics *after* a train/test split can introduce target
leakage when the imputer observes test-set wealth patterns.

``WealthScreeningImputer`` adheres strictly to the scikit-learn estimator
contract: all fill statistics are computed exclusively from ``X_train`` during
:meth:`fit` and are frozen before :meth:`transform` is called on held-out or
future data.

Typical usage
-------------
>>> import pandas as pd
>>> import numpy as np
>>> from philanthropy.preprocessing import WealthScreeningImputer
>>> wealth_cols = ["estimated_net_worth", "real_estate_value"]
>>> X_train = pd.DataFrame({
...     "donor_id":            [1, 2, 3],
...     "estimated_net_worth": [500_000.0, np.nan, 1_200_000.0],
...     "real_estate_value":   [np.nan,    350_000.0, np.nan],
... })
>>> imputer = WealthScreeningImputer(wealth_cols=wealth_cols)
>>> imputer.fit(X_train)
WealthScreeningImputer(...)
>>> imputer.fill_values_
{'estimated_net_worth': 850000.0, 'real_estate_value': 350000.0}
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data

# Default column names used by common wealth-screening vendors
_DEFAULT_WEALTH_COLS: List[str] = [
    "estimated_net_worth",
    "real_estate_value",
    "stock_holdings",
    "charitable_capacity",
    "planned_gift_inclination",
]


class WealthScreeningImputer(TransformerMixin, BaseEstimator):
    """Leakage-safe median/constant imputation for wealth-screening columns.

    This transformer learns fill statistics **only** from the training fold
    during :meth:`fit` and applies them in :meth:`transform`.  It is designed
    to slot cleanly into a :class:`sklearn.pipeline.Pipeline` immediately after
    :class:`~philanthropy.preprocessing.CRMCleaner` and before any model that
    cannot natively handle ``NaN`` values.

    Parameters
    ----------
    wealth_cols : list of str or None, default=None
        Column names containing third-party wealth-screening numeric values.
        If ``None``, defaults to a canonical set (``estimated_net_worth``,
        ``real_estate_value``, ``stock_holdings``, ``charitable_capacity``,
        ``planned_gift_inclination``).  Only columns that *actually exist* in
        ``X`` are imputed; missing columns are skipped with a warning.
    strategy : {"median", "mean", "zero"}, default="median"
        Imputation strategy applied to each wealth column:

        * ``"median"`` — Robust to the extreme right-skew and outliers common
          in wealth data.  Strongly recommended for raw vendor exports.
        * ``"mean"``   — Computationally equivalent to OLS; use only after
          outlier treatment.
        * ``"zero"``   — Sets missing values to 0.0, which is semantically
          meaningful when absence of a record implies zero capacity (e.g., no
          real-estate holdings found).
    add_indicator : bool, default=True
        If ``True``, appends a binary indicator column
        ``<column_name>__was_missing`` (dtype ``uint8``) for each imputed
        wealth column.  Retaining missingness signals allows downstream models
        to learn that the absence of a vendor record itself carries information
        (e.g., very high-net-worth individuals are often *not* found in
        commercial databases because they actively shield their assets).
    fiscal_year_start : int, default=7
        Month (1–12) starting the organisation's fiscal year.  Inherited for
        pipeline compatibility.

    Attributes
    ----------
    fill_values_ : dict of {str: float}
        Mapping from column name to the computed fill value, frozen at
        :meth:`fit` time.
    imputed_cols_ : list of str
        Wealth columns that were actually present in ``X`` at :meth:`fit`
        time and will be imputed.
    n_features_in_ : int
        Number of columns in ``X`` at :meth:`fit` time.
    feature_names_in_ : ndarray of str
        Column names of ``X`` at :meth:`fit` time.

    Raises
    ------
    ValueError
        If ``strategy`` is not one of ``{"median", "mean", "zero"}``.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from philanthropy.preprocessing import WealthScreeningImputer
    >>> X = pd.DataFrame({
    ...     "estimated_net_worth": [1e6, np.nan, 5e5, np.nan, 2e6],
    ...     "real_estate_value":   [np.nan, 3e5, np.nan, 4e5, np.nan],
    ...     "gift_amount":         [5000, 250, 1000, 750, 10000],
    ... })
    >>> imp = WealthScreeningImputer(
    ...     wealth_cols=["estimated_net_worth", "real_estate_value"],
    ...     strategy="median",
    ...     add_indicator=True,
    ... )
    >>> X_out = imp.fit_transform(X)
    >>> X_out["estimated_net_worth"].isna().any()
    False
    >>> "estimated_net_worth__was_missing" in X_out.columns
    True

    See Also
    --------
    philanthropy.preprocessing.CRMCleaner :
        Upstream cleaner that standardises column dtypes before this imputer.
    philanthropy.models.ShareOfWalletRegressor :
        Downstream model that uses wealth-screening features to estimate
        philanthropic capacity.
    """

    _VALID_STRATEGIES = frozenset({"median", "mean", "zero"})

    def __init__(
        self,
        wealth_cols: Optional[List[str]] = None,
        strategy: Literal["median", "mean", "zero"] = "median",
        add_indicator: bool = True,
    ) -> None:
        self.wealth_cols = wealth_cols
        self.strategy = strategy
        self.add_indicator = add_indicator

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_cols(self, input_cols: List[str]) -> List[str]:
        """Return the wealth columns that actually exist in ``X``."""
        candidates = (
            self.wealth_cols
            if self.wealth_cols is not None
            else _DEFAULT_WEALTH_COLS
        )
        return [c for c in candidates if c in input_cols]

    def _compute_fill(self, array: np.ndarray) -> float:
        """Return the fill value for a single wealth column."""
        if self.strategy == "median":
            val = np.nanmedian(array)
        elif self.strategy == "mean":
            val = np.nanmean(array)
        else:  # "zero"
            val = 0.0
        # If the column is entirely NaN, fall back to 0.0
        return float(val) if not np.isnan(val) else 0.0

    # ------------------------------------------------------------------
    # fit / transform
    # ------------------------------------------------------------------

    def fit(self, X, y=None) -> "WealthScreeningImputer":
        """Learn fill statistics from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training-set feature matrix.  Missing wealth columns are silently
            skipped (a ``UserWarning`` is issued for each absent column).
        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        self : WealthScreeningImputer
            Fitted imputer.

        Raises
        ------
        ValueError
            If ``strategy`` is not ``"median"``, ``"mean"``, or ``"zero"``.
        """
        import warnings
        X = validate_data(self, X, dtype="numeric",
                          ensure_all_finite="allow-nan",
                          reset=True)

        if self.strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"`strategy` must be one of {sorted(self._VALID_STRATEGIES)}, "
                f"got {self.strategy!r}."
            )

        if hasattr(X, "columns"):
            input_cols = list(X.columns)
        else:
            n_cols = X.shape[1]
            input_cols = [f"x{i}" for i in range(n_cols)]
        self.imputed_cols_ = self._resolve_cols(input_cols)

        # Warn about requested columns not found in X
        if self.wealth_cols is not None:
            missing = [c for c in self.wealth_cols if c not in input_cols]
            for col in missing:
                warnings.warn(
                    f"WealthScreeningImputer: column {col!r} was specified in "
                    f"`wealth_cols` but was not found in X.  It will be skipped.",
                    UserWarning,
                )

        computed_fills = {}
        for col in self.imputed_cols_:
            idx = input_cols.index(col)
            computed_fills[col] = self._compute_fill(X[:, idx])
        self.fill_values_ = computed_fills

        return self

    def transform(self, X) -> np.ndarray:
        """Apply imputation with frozen fill values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix (training or held-out).

        Returns
        -------
        X_out : np.ndarray
            Copy of ``X`` with missing wealth columns filled and, if
            ``add_indicator=True``, binary missingness indicator columns
            appended.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        """
        check_is_fitted(self, ["fill_values_", "imputed_cols_"])
        
        X = validate_data(self, X, dtype="numeric",
                          ensure_all_finite="allow-nan",
                          reset=False)

        if hasattr(X, "columns"):
            input_cols = list(X.columns)
        else:
            n_cols = X.shape[1]
            input_cols = [f"x{i}" for i in range(n_cols)]

        X_out = X.copy()
        
        indicators = []

        for col in self.imputed_cols_:
            if col not in input_cols:
                continue
            idx = input_cols.index(col)
            
            mask = np.isnan(X_out[:, idx])
            
            if self.add_indicator:
                indicators.append(mask.astype(np.float64).reshape(-1, 1))

            X_out[mask, idx] = self.fill_values_[col]

        if indicators:
            return np.hstack([X_out] + indicators)
        return X_out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        if input_features is not None:
            out = list(input_features)
        elif hasattr(self, "feature_names_in_"):
            out = list(self.feature_names_in_)
        else:
            out = [f"x{i}" for i in range(self.n_features_in_)]

        if self.add_indicator:
            for col in self.imputed_cols_:
                if col in out:
                    out.append(f"{col}__was_missing")
        return np.array(out, dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

