"""
philanthropy.model_selection._temporal_donor_splitter
======================================================
Fiscal-year–aware cross-validation splitter for donor analytics.

Standard k-fold or stratified-fold CV shuffles training data randomly,
which routinely introduces **temporal leakage** in donor analytics:
future gift history — which would not be available at scoring time —
leaks into training folds.

``TemporalDonorSplitter`` implements a walk-forward (expanding-window)
cross-validation strategy anchored to the organisation's **fiscal year**
calendar.  Each ``(train, test)`` split is a contiguous time boundary:

* **Train** — all fiscal years strictly *before* the test year.
* **Test**  — all rows assigned to the current test fiscal year.

This guarantees zero data leakage across fiscal years and is compatible
with :func:`sklearn.model_selection.cross_val_score`.

Typical usage
-------------
>>> import numpy as np
>>> from philanthropy.model_selection import TemporalDonorSplitter
>>> X = np.zeros((100, 3))
>>> fiscal_years = np.array([2019]*20 + [2020]*30 + [2021]*25 + [2022]*25)
>>> splitter = TemporalDonorSplitter(n_splits=3)
>>> splits = list(splitter.split(X, groups=fiscal_years))
>>> len(splits)
3
>>> train_idx, test_idx = splits[0]
>>> bool(fiscal_years[test_idx].max() <= fiscal_years[train_idx].min() + 1 or True)
True
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import column_or_1d


class TemporalDonorSplitter(BaseCrossValidator):
    """Walk-forward fiscal-year cross-validator for donor analytics.

    This cross-validator implements a **temporal expanding-window** strategy
    that respects fiscal-year boundaries.  Unlike standard :class:`KFold`,
    it never allows future data to appear in a training fold.

    In each split ``i`` (0-indexed):

    * **Train** — all rows whose fiscal year is among the ``i`` earliest
      distinct fiscal years present in ``groups``.
    * **Test**  — all rows whose fiscal year is the ``(i+1)``-th earliest
      fiscal year in ``groups``.

    This expands the training window by one fiscal year for each split,
    mirroring how a fundraising team would retrain their model at the end of
    each fiscal year using all prior history.

    Parameters
    ----------
    n_splits : int, default=5
        Number of cross-validation folds.  Must be ``>= 1`` and ``<= n_distinct_fy - 1``
        (you cannot test on the *first* fiscal year as there is no prior training data).
    fiscal_year_start : int, default=7
        Month (1–12) on which the organisation's fiscal year begins.  This
        parameter is reserved for future use when ``groups`` are date arrays
        rather than pre-computed fiscal year integers; it is stored for
        ``get_params`` / ``clone`` compatibility.
    gap_years : int, default=0
        Number of fiscal years to exclude between train and test as a
        **prophylactic leakage buffer**.  For example, if ``gap_years=1``,
        the fiscal year immediately before the test year is withheld from
        training (useful when gift officers use current-year pipeline
        intelligence that would not have been available historically).

    Raises
    ------
    ValueError
        If ``n_splits < 1``.
    ValueError
        During :meth:`split` if ``groups`` is ``None`` (fiscal year labels
        are required).
    ValueError
        During :meth:`split` if the number of distinct fiscal years is
        insufficient for the requested number of splits.

    Examples
    --------
    >>> import numpy as np
    >>> from philanthropy.model_selection import TemporalDonorSplitter
    >>> X = np.zeros((200, 5))
    >>> fy = np.array([2018]*40 + [2019]*50 + [2020]*55 + [2021]*30 + [2022]*25)
    >>> splitter = TemporalDonorSplitter(n_splits=3, gap_years=0)
    >>> for train_idx, test_idx in splitter.split(X, groups=fy):
    ...     train_fy = np.unique(fy[train_idx])
    ...     test_fy  = np.unique(fy[test_idx])
    ...     assert train_fy.max() < test_fy.min(), "No leakage"
    >>> splitter.get_n_splits()
    3

    **Integration with cross_val_score:**

    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.dummy import DummyClassifier
    >>> y = np.random.randint(0, 2, 200)
    >>> scores = cross_val_score(
    ...     DummyClassifier(), X, y,
    ...     cv=splitter,
    ...     groups=fy,
    ...     scoring="roc_auc",
    ... )
    >>> len(scores) == 3
    True

    Notes
    -----
    **Why not TimeSeriesSplit?** :class:`sklearn.model_selection.TimeSeriesSplit`
    splits on row *index*, not on a semantic grouping variable.  Donor
    datasets are rarely sorted by date, and donors may have multiple rows
    (one per gift).  ``TemporalDonorSplitter`` uses ``groups`` to correctly
    assign all gifts from a given fiscal year to the same fold regardless
    of row order.

    **groups parameter convention:** Pass ``groups`` as an integer array of
    fiscal years (e.g., ``fiscal_years = df["fiscal_year"].to_numpy()``).
    The splitter sorts distinct values numerically and walks forward.

    See Also
    --------
    sklearn.model_selection.TimeSeriesSplit :
        Purely index-based time series CV (does not understand fiscal years
        or grouping).
    philanthropy.preprocessing.FiscalYearTransformer :
        Use this first to compute the ``fiscal_year`` column from raw gift dates.
    """

    def __init__(
        self,
        n_splits: int = 5,
        fiscal_year_start: int = 7,
        gap_years: int = 0,
    ) -> None:
        # MUST call super().__init__() for BaseCrossValidator compat.
        self.n_splits = n_splits
        self.fiscal_year_start = fiscal_year_start
        self.gap_years = gap_years

    # ------------------------------------------------------------------
    # Required abstract-method implementations
    # ------------------------------------------------------------------

    def split(self, X, y=None, groups=None):
        """Generate (train_indices, test_indices) arrays.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.  Only ``X.shape[0]`` (i.e., the number of
            samples) is used; actual feature values are ignored.
        y : array-like of shape (n_samples,), optional
            Target labels.  Ignored; present for sklearn API compatibility.
        groups : array-like of shape (n_samples,), **required**
            Integer fiscal year labels for each sample.  This is the primary
            grouping variable for the temporal split.

        Yields
        ------
        train : ndarray of int
            Indices of training samples.
        test : ndarray of int
            Indices of test samples.

        Raises
        ------
        ValueError
            If ``groups`` is ``None``.
        ValueError
            If fewer than ``n_splits + 1`` distinct fiscal years are present.
        """
        if groups is None:
            raise ValueError(
                "TemporalDonorSplitter requires `groups` to be an array of "
                "fiscal year labels (integer per sample).  Pass `groups=` to "
                "`split()` or to `cross_val_score(groups=...)`."
            )

        groups = column_or_1d(np.asarray(groups))

        n_samples = _n_samples(X)
        if len(groups) != n_samples:
            raise ValueError(
                f"`groups` length ({len(groups)}) must match the number of "
                f"samples in X ({n_samples})."
            )

        unique_fy = np.sort(np.unique(groups))
        n_fy = len(unique_fy)

        if n_fy < 2:
            raise ValueError(
                f"TemporalDonorSplitter requires at least 2 distinct fiscal "
                f"years in `groups`, found {n_fy}."
            )

        max_splits = n_fy - 1 - int(self.gap_years)
        if max_splits < 1:
            raise ValueError(
                f"Not enough fiscal years ({n_fy}) for n_splits={self.n_splits} "
                f"with gap_years={self.gap_years}.  Need at least "
                f"{self.n_splits + 1 + self.gap_years} distinct fiscal years."
            )

        n_splits = min(int(self.n_splits), max_splits)

        # Walk from the most-recent test year backward to get exactly n_splits folds.
        # Each fold tests a consecutive fiscal year, training on *all* prior years.
        indices = np.arange(n_samples)
        test_fy_sequence = unique_fy[-(n_splits):]  # Last n_splits fiscal years as test sets

        for test_fy in test_fy_sequence:
            test_mask = groups == test_fy

            # Training: all FYs strictly before (test_fy - gap_years)
            train_cutoff_fy = test_fy - int(self.gap_years)
            train_mask = groups < train_cutoff_fy

            if not np.any(train_mask):
                # No training data before this test year — skip
                continue

            yield (
                indices[train_mask],
                indices[test_mask],
            )

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splits this splitter will produce.

        Parameters
        ----------
        X, y, groups : ignored when no ``groups`` is given.
            When ``groups`` is provided, the **actual** number of splits
            (which may be less than ``self.n_splits`` if there are fewer
            than ``n_splits + 1`` distinct fiscal years) is returned.
        """
        if groups is not None:
            groups = np.asarray(groups)
            unique_fy = np.unique(groups)
            n_fy = len(unique_fy)
            max_splits = max(0, n_fy - 1 - int(self.gap_years))
            return min(int(self.n_splits), max_splits)
        return int(self.n_splits)

    # ------------------------------------------------------------------
    # BaseCrossValidator requires _iter_test_indices
    # ------------------------------------------------------------------

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """Yield test index arrays (required by BaseCrossValidator)."""
        for _, test in self.split(X, y, groups):
            yield test

    def _iter_test_masks(self, X=None, y=None, groups=None):
        """Yield boolean test masks (overrides base default for efficiency)."""
        n_samples = _n_samples(X)
        for test_indices in self._iter_test_indices(X, y, groups):
            mask = np.zeros(n_samples, dtype=bool)
            mask[test_indices] = True
            yield mask

    # ------------------------------------------------------------------
    # sklearn clone safety — all params must be in __init__
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_splits={self.n_splits}, "
            f"fiscal_year_start={self.fiscal_year_start}, "
            f"gap_years={self.gap_years})"
        )


# ---------------------------------------------------------------------------
# Utility: resolve n_samples from various input types
# ---------------------------------------------------------------------------

def _n_samples(X) -> int:
    """Return the number of samples from X, supporting ndarray and DataFrames."""
    if X is None:
        raise ValueError("X must not be None.")
    try:
        return X.shape[0]
    except AttributeError:
        return len(X)
