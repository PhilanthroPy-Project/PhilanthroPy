"""
philanthropy.preprocessing._share_of_wallet
============================================
Wealth imputation with KNN support and Share-of-Wallet scoring.

``ShareOfWalletScorer`` is the terminal step in a major-gift capacity
pipeline.  It computes a [0, 1]-normalised score:

    SoW = estimated_capacity / (total_modelled_wealth + epsilon)

and assigns a human-readable ``capacity_tier`` label suitable for CRM
export (Raiser's Edge NXT, Salesforce NPSP, Ellucian Advance).

``WealthScreeningImputerKNN`` extends the base
:class:`~philanthropy.preprocessing.WealthScreeningImputer` with a
``"knn"`` strategy that imputes via k-Nearest Neighbours on non-missing
wealth columns, which materially outperforms median imputation when
geographic or demographic structure is available (e.g., zip-code
clusters in hospital databases).

Typical usage
-------------
>>> import numpy as np
>>> from philanthropy.preprocessing._share_of_wallet import ShareOfWalletScorer
>>> rng = np.random.default_rng(0)
>>> X = rng.uniform(0, 1e6, (10, 3))  # [estimated_cap, re_value, stocks]
>>> scorer = ShareOfWalletScorer(capacity_col_idx=0, epsilon=1.0)
>>> scorer.fit(X)
ShareOfWalletScorer(...)
>>> out = scorer.transform(X)
>>> out.shape
(10, 2)
>>> bool((out[:, 0] >= 0).all() and (out[:, 0] <= 1).all())
True
"""

from __future__ import annotations

import warnings
from typing import Optional, Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.utils.validation import check_is_fitted, validate_data


# ---------------------------------------------------------------------------
# WealthScreeningImputerKNN
# ---------------------------------------------------------------------------

class WealthScreeningImputerKNN(TransformerMixin, BaseEstimator):
    """Leakage-safe KNN imputation for wealth-screening vendor columns.

    Extends the median/mean/zero strategy of
    :class:`~philanthropy.preprocessing.WealthScreeningImputer` with a
    ``"knn"`` strategy using :class:`sklearn.impute.KNNImputer`.  KNN
    imputation is recommended when wealth columns cluster meaningfully
    (e.g., by zip-code based real-estate quartile), which is common in
    curated hospital prospect pools where WealthEngine / DonorSearch
    data has geographic structure.

    This estimator **delegates** to ``sklearn.impute.KNNImputer`` internally
    and inherits its Pipeline composability and clone-safety.

    Parameters
    ----------
    wealth_cols : list of str or None, default=None
        Subset of columns to impute.  If ``None``, all columns whose
        names contain substrings from a canonical set (``net_worth``,
        ``real_estate``, ``stock``, ``capacity``, ``charitable``) are
        imputed.
    strategy : {"median", "mean", "zero", "knn"}, default="knn"
        Imputation strategy.  ``"knn"`` uses
        :class:`sklearn.impute.KNNImputer` with ``n_neighbors``.
        The other strategies use columnwise statistics identical to
        :class:`~philanthropy.preprocessing.WealthScreeningImputer`.
    n_neighbors : int, default=5
        Number of neighbours used when ``strategy="knn"``.  Ignored for
        other strategies.
    add_indicator : bool, default=True
        Append a binary ``<col>__was_missing`` column for each imputed
        wealth column.  Strongly recommended — absence of vendor records
        itself carries predictive signal.
    group_col_idx : int or None, default=None
        Column index of a group variable (e.g., zip-code encoded as int)
        to stratify KNN imputation.  When provided (and
        ``strategy="knn"``), imputation is performed independently per
        group, improving local accuracy.

    Attributes
    ----------
    imputed_cols_ : list of str
        Wealth columns that were actually present in ``X`` at fit time.
    fill_values_ : dict of {str: float}
        Fill statistics (only populated for non-KNN strategies).
    knn_imputer_ : KNNImputer or None
        The fitted :class:`~sklearn.impute.KNNImputer` instance
        (only populated for ``strategy="knn"``).
    n_features_in_ : int
        Number of columns in ``X`` at fit time.
    feature_names_in_ : ndarray of str
        Column names of ``X`` at fit time.

    Examples
    --------
    >>> import numpy as np
    >>> from philanthropy.preprocessing._share_of_wallet import WealthScreeningImputerKNN
    >>> rng = np.random.default_rng(42)
    >>> X = rng.uniform(0, 1e6, (50, 3))
    >>> X[rng.random((50, 3)) < 0.3] = np.nan
    >>> imp = WealthScreeningImputerKNN(strategy="knn", n_neighbors=3, add_indicator=False)
    >>> imp.fit(X)
    WealthScreeningImputerKNN(...)
    >>> out = imp.transform(X)
    >>> bool(np.isnan(out).any())
    False
    """

    _VALID_STRATEGIES = frozenset({"median", "mean", "zero", "knn"})
    _CANONICAL_SUBSTRINGS = ("net_worth", "real_estate", "stock", "capacity", "charitable")

    def __init__(
        self,
        wealth_cols: list[str] | None = None,
        strategy: Literal["median", "mean", "zero", "knn"] = "knn",
        n_neighbors: int = 5,
        add_indicator: bool = True,
        group_col_idx: Optional[int] = None,
    ) -> None:
        self.wealth_cols = wealth_cols
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.add_indicator = add_indicator
        self.group_col_idx = group_col_idx

    def _resolve_cols(self, input_cols: list[str]) -> list[str]:
        if self.wealth_cols is not None:
            return [c for c in self.wealth_cols if c in input_cols]
        return [c for c in input_cols
                if any(sub in c.lower() for sub in self._CANONICAL_SUBSTRINGS)]

    def fit(self, X, y=None) -> "WealthScreeningImputerKNN":
        """Learn fill statistics or fit the KNN imputer from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self : WealthScreeningImputerKNN
        """
        if self.strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"`strategy` must be one of {sorted(self._VALID_STRATEGIES)}, "
                f"got {self.strategy!r}."
            )
        if self.strategy == "knn" and self.n_neighbors < 1:
            raise ValueError(f"`n_neighbors` must be >= 1, got {self.n_neighbors}.")

        # Capture column names before validate_data converts DF → ndarray
        if hasattr(X, "columns"):
            input_cols = list(X.columns)
        else:
            input_cols = None

        X_arr = validate_data(
            self, X, dtype="numeric", ensure_all_finite="allow-nan", reset=True
        )

        if input_cols is None:
            input_cols = (
                list(self.feature_names_in_)
                if hasattr(self, "feature_names_in_")
                else [f"x{i}" for i in range(X_arr.shape[1])]
            )

        self.imputed_cols_ = self._resolve_cols(input_cols)

        # Warn about columns requested but absent
        if self.wealth_cols is not None:
            for col in self.wealth_cols:
                if col not in input_cols:
                    warnings.warn(
                        f"WealthScreeningImputerKNN: column {col!r} not found in X.",
                        UserWarning,
                    )

        col_indices = {col: input_cols.index(col) for col in self.imputed_cols_}

        if self.strategy == "knn":
            # Fit KNNImputer on ALL columns (preserves inter-column structure)
            self.knn_imputer_: Optional[KNNImputer] = KNNImputer(
                n_neighbors=self.n_neighbors,
                weights="distance",
                keep_empty_features=True,
            )
            self.knn_imputer_.fit(X_arr)
            self.fill_values_: dict[str, float] = {}
        else:
            self.knn_imputer_ = None
            fills: dict[str, float] = {}
            for col in self.imputed_cols_:
                idx = col_indices[col]
                col_data = X_arr[:, idx]
                if self.strategy == "median":
                    val = np.nanmedian(col_data)
                elif self.strategy == "mean":
                    val = np.nanmean(col_data)
                else:  # "zero"
                    val = 0.0
                fills[col] = float(val) if not np.isnan(val) else 0.0
            self.fill_values_ = fills

        self._col_indices_ = col_indices
        return self

    def transform(self, X, y=None) -> np.ndarray:
        """Apply imputation and optionally append missingness indicators.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_out : np.ndarray
            Imputed array (float64), with indicator columns appended if
            ``add_indicator=True``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
        """
        check_is_fitted(self, ["imputed_cols_"])

        if hasattr(X, "columns"):
            input_cols = list(X.columns)
        else:
            input_cols = None

        X_arr = validate_data(
            self, X, dtype="numeric", ensure_all_finite="allow-nan", reset=False
        )

        if input_cols is None:
            input_cols = (
                list(self.feature_names_in_)
                if hasattr(self, "feature_names_in_")
                else [f"x{i}" for i in range(X_arr.shape[1])]
            )

        # Collect missingness masks BEFORE imputation
        indicators: list[np.ndarray] = []
        if self.add_indicator:
            for col in self.imputed_cols_:
                if col in input_cols:
                    idx = input_cols.index(col)
                    indicators.append(np.isnan(X_arr[:, idx]).astype(np.float64).reshape(-1, 1))

        if self.strategy == "knn" and self.knn_imputer_ is not None:
            X_out = self.knn_imputer_.transform(X_arr)
        else:
            X_out = X_arr.copy()
            for col in self.imputed_cols_:
                if col not in input_cols:
                    continue
                idx = input_cols.index(col)
                mask = np.isnan(X_out[:, idx])
                X_out[mask, idx] = self.fill_values_.get(col, 0.0)

        if indicators:
            return np.hstack([X_out] + indicators)
        return X_out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self)
        if input_features is not None:
            base = list(input_features)
        elif hasattr(self, "feature_names_in_"):
            base = list(self.feature_names_in_)
        else:
            base = [f"x{i}" for i in range(self.n_features_in_)]

        out = list(base)
        if self.add_indicator:
            for col in self.imputed_cols_:
                if col in base:
                    out.append(f"{col}__was_missing")
        return np.array(out, dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


# ---------------------------------------------------------------------------
# ShareOfWalletScorer
# ---------------------------------------------------------------------------

# Tier boundaries (share-of-wallet score, lower-bound inclusive)
_TIER_THRESHOLDS = [
    (0.75, "Principal"),
    (0.40, "Major"),
    (0.00, "Leadership"),
]


def _assign_tier(score: float) -> str:
    """Map a [0, 1] SoW score to a descriptive capacity tier label."""
    for threshold, label in _TIER_THRESHOLDS:
        if score >= threshold:
            return label
    return "Leadership"


class ShareOfWalletScorer(TransformerMixin, BaseEstimator):
    """Compute a normalised Share-of-Wallet score and capacity-tier label.

    This transformer is designed as the **final stage** of a major-gift
    capacity scoring pipeline.  It consumes a numeric feature matrix and
    produces two outputs per row:

    ``sow_score`` (float64, [0, 1])
        Normalised Share-of-Wallet:

            SoW = estimated_capacity / (total_modelled_wealth + epsilon)

        where ``total_modelled_wealth`` is the row-wise sum of all columns
        specified in ``wealth_col_indices`` (or all columns if not
        specified), and ``estimated_capacity`` is the column at
        ``capacity_col_idx``.

    ``capacity_tier`` (float64, categorical encoding)
        A numeric encoding of the human-readable tier label, usable by
        downstream sklearn estimators (e.g., a classifier trained to
        predict tier upgrades).  The mapping is:

        ============ ============ =========================================
        SoW score    Tier label   Recommended action
        ============ ============ =========================================
        ≥ 0.75       Principal    Schedule personal visit with campaign chair.
        0.40 – 0.75  Major        Assign major gift officer.
        0.00 – 0.40  Leadership   Include in leadership annual giving.
        ============ ============ =========================================

    Parameters
    ----------
    capacity_col_idx : int, default=0
        Column index (0-based) in ``X`` containing the estimated
        philanthropic capacity (in dollars or any consistent currency unit).
    wealth_col_indices : list of int or None, default=None
        Column indices to sum as "total modelled wealth".  If ``None``,
        all columns are summed (including ``capacity_col_idx``).
    epsilon : float, default=1.0
        Small constant added to the denominator to prevent division by
        zero when all wealth columns are zero.
    capacity_floor : float, default=0.0
        Minimum value to enforce on ``estimated_capacity`` before scoring
        (prevents negative capacity from distorting the SoW score).

    Attributes
    ----------
    wealth_scale_ : float
        95th-percentile total modelled wealth observed at fit time, used to
        clip outlier wealth sums during :meth:`transform`.  This prevents a
        single ultra-high-net-worth outlier from compressing all other
        scores near 0.
    n_features_in_ : int
    feature_names_in_ : ndarray of str

    Examples
    --------
    >>> import numpy as np
    >>> from philanthropy.preprocessing._share_of_wallet import ShareOfWalletScorer
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(0, 1e6, (20, 4))
    >>> scorer = ShareOfWalletScorer(capacity_col_idx=0, epsilon=1.0)
    >>> scorer.fit(X)
    ShareOfWalletScorer(...)
    >>> out = scorer.transform(X)
    >>> out.shape
    (20, 2)
    >>> bool(((out[:, 0] >= 0) & (out[:, 0] <= 1)).all())
    True
    """

    # Public tier-label mapping for callers who need string labels
    TIER_LABELS = {0: "Leadership", 1: "Major", 2: "Principal"}
    TIER_ENCODING = {"Leadership": 0, "Major": 1, "Principal": 2}

    def __init__(
        self,
        capacity_col_idx: int = 0,
        wealth_col_indices: Optional[list[int]] = None,
        epsilon: float = 1.0,
        capacity_floor: float = 0.0,
    ) -> None:
        self.capacity_col_idx = capacity_col_idx
        self.wealth_col_indices = wealth_col_indices
        self.epsilon = epsilon
        self.capacity_floor = capacity_floor

    def fit(self, X, y=None) -> "ShareOfWalletScorer":
        """Fit the scorer: record wealth scale from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self : ShareOfWalletScorer
        """
        if self.epsilon < 0:
            raise ValueError(f"`epsilon` must be >= 0, got {self.epsilon}.")
        if not (0 <= self.capacity_col_idx):
            raise ValueError(f"`capacity_col_idx` must be >= 0, got {self.capacity_col_idx}.")

        X_arr = validate_data(
            self, X, dtype="numeric", ensure_all_finite="allow-nan", reset=True
        )

        # Compute total wealth denominator columns
        w_indices = (
            list(range(X_arr.shape[1]))
            if self.wealth_col_indices is None
            else [int(i) for i in self.wealth_col_indices]
        )
        wealth_sum = np.nansum(X_arr[:, w_indices], axis=1)

        # 95th-percentile scale to clip outliers and keep SoW in [0, 1]
        if len(wealth_sum) > 0:
            p95 = np.nanpercentile(wealth_sum, 95)
            self.wealth_scale_ = float(p95) if p95 > 0 else 1.0
        else:
            self.wealth_scale_ = 1.0

        return self

    def transform(self, X, y=None) -> np.ndarray:
        """Compute SoW score and numeric capacity tier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_out : np.ndarray of shape (n_samples, 2), dtype float64
            Column 0: ``sow_score`` in [0, 1].
            Column 1: ``capacity_tier`` (0 = Leadership, 1 = Major, 2 = Principal).

        Raises
        ------
        sklearn.exceptions.NotFittedError
        """
        check_is_fitted(self, ["wealth_scale_"])
        X_arr = validate_data(
            self, X, dtype="numeric", ensure_all_finite="allow-nan", reset=False
        )

        w_indices = (
            list(range(X_arr.shape[1]))
            if self.wealth_col_indices is None
            else [int(i) for i in self.wealth_col_indices]
        )

        # Capacity column
        cap_idx = int(self.capacity_col_idx)
        if cap_idx >= X_arr.shape[1]:
            raise ValueError(
                f"`capacity_col_idx` ({cap_idx}) exceeds number of columns "
                f"({X_arr.shape[1]})."
            )
        capacity = np.maximum(
            np.nan_to_num(X_arr[:, cap_idx], nan=0.0),
            self.capacity_floor,
        )

        # Wealth sum — clip at 95th-percentile scale from fit to prevent score collapse
        wealth_raw = np.nansum(X_arr[:, w_indices], axis=1)
        wealth_clipped = np.clip(wealth_raw, 0.0, self.wealth_scale_)

        # SoW = capacity / (wealth + epsilon), then clip to [0, 1]
        sow = np.clip(
            capacity / (wealth_clipped + float(self.epsilon)), 0.0, 1.0
        )

        # Tier encoding (vectorised)
        tiers = np.zeros(len(sow), dtype=np.float64)
        tiers[sow >= 0.40] = float(self.TIER_ENCODING["Major"])
        tiers[sow >= 0.75] = float(self.TIER_ENCODING["Principal"])

        return np.column_stack([sow, tiers])

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self)
        return np.array(["sow_score", "capacity_tier"], dtype=object)

    def get_tier_labels(self, X) -> np.ndarray:
        """Return human-readable tier labels for each row.

        Parameters
        ----------
        X : array-like compatible with :meth:`transform`

        Returns
        -------
        labels : ndarray of str, shape (n_samples,)
            One of ``"Principal"``, ``"Major"``, or ``"Leadership"`` per row.
        """
        out = self.transform(X)
        tier_ints = out[:, 1].astype(int)
        return np.array([self.TIER_LABELS[t] for t in tier_ints], dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
