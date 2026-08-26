"""
philanthropy.model_selection._temporal_donor_splitter
======================================================
Fiscal-year–aware cross-validation splitter for donor analytics.

Standard k-fold or stratified-fold CV shuffles training data randomly,
which routinely introduces **temporal leakage** in donor analytics:
future gift history (which would not be available at scoring time)
leaks into training folds.

``FiscalYearGroupedSplitter`` implements a walk-forward (expanding-window)
cross-validation strategy anchored to the organisation's **fiscal year**
calendar.  Each ``(train, test)`` split is a contiguous time boundary:

* **Train**: all fiscal years strictly *before* the test year.
* **Test**: all rows assigned to the current test fiscal year.

This guarantees zero data leakage across fiscal years and is compatible
with :func:`sklearn.model_selection.cross_val_score`.

Typical usage
-------------
>>> import numpy as np
>>> from philanthropy.model_selection import FiscalYearGroupedSplitter
>>> X = np.zeros((100, 3))
>>> fiscal_years = np.array([2019]*20 + [2020]*30 + [2021]*25 + [2022]*25)
>>> splitter = FiscalYearGroupedSplitter(n_splits=3, drop_repeat_donors=False)
>>> splits = list(splitter.split(X, groups=fiscal_years))
>>> len(splits)
3
>>> train_idx, test_idx = splits[0]
>>> bool(fiscal_years[train_idx].max() < fiscal_years[test_idx].min())
True
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import column_or_1d


class FiscalYearGroupedSplitter(BaseCrossValidator):
    """Walk-forward fiscal-year cross-validator for donor analytics.

    This cross-validator implements a **temporal expanding-window** strategy
    that respects fiscal-year boundaries.  Unlike standard :class:`KFold`,
    it never allows future data to appear in a training fold.

    In each split ``i`` (0-indexed):

    * **Train**: all rows whose fiscal year is among the ``i`` earliest
      distinct fiscal years present in ``groups``.
    * **Test**: all rows whose fiscal year is the ``(i+1)``-th earliest
      fiscal year in ``groups``.

    This expands the training window by one fiscal year for each split,
    mirroring how a fundraising team would retrain their model at the end of
    each fiscal year using all prior history.

    Parameters
    ----------
    n_splits : int, default=5
        Number of cross-validation folds.  Must be ``>= 1`` and ``<= n_distinct_fy - 1``
        (you cannot test on the *first* fiscal year as there is no prior training data).
    gap_years : int, default=0
        Number of fiscal years to exclude between train and test as a
        **prophylactic leakage buffer**.  For example, if ``gap_years=1``,
        the fiscal year immediately before the test year is withheld from
        training (useful when gift officers use current-year pipeline
        intelligence that would not have been available historically).
    drop_repeat_donors : bool, default=False
        .. deprecated:: 0.7.0
            Leaving ``drop_repeat_donors`` at its default emits a
            ``DeprecationWarning``. The default changes to ``True`` in 0.8.0.
            Pass ``drop_repeat_donors=False`` explicitly to silence this warning
            and keep the current behaviour.

        Whether to remove from each test fold any donor who already appears in
        that fold's training rows.

        Leave this ``False`` for a **time-varying** target such as "did this
        donor give in FY22?". There, a donor appearing in both folds is correct:
        the training rows precede the test rows in time, which is the point of
        walk-forward evaluation.

        Set it ``True`` for a **static per-donor** label such as
        ``is_major_donor``, where the same answer is attached to every one of
        that donor's rows, so the model can memorise it from the donor's earlier
        years. That is the leakage described under "What this does not prevent"
        below.

        When ``True``, ``groups`` must be two-dimensional with shape
        ``(n_samples, 2)``: column 0 the fiscal year, column 1 the donor
        identifier. A :class:`pandas.DataFrame` with those two columns in that
        order works.

        It is not free. Donors active in both windows leave the test fold, so it
        shrinks and the donors remaining are systematically newer to the file.
        ``split`` emits a ``UserWarning`` with the number of rows removed, so the
        cost is visible rather than silent. A test fold emptied entirely raises
        rather than being skipped, because silently changing the fold count would
        put ``split`` and ``get_n_splits`` back out of step.

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
    >>> from philanthropy.model_selection import FiscalYearGroupedSplitter
    >>> X = np.zeros((200, 5))
    >>> fy = np.array([2018]*40 + [2019]*50 + [2020]*55 + [2021]*30 + [2022]*25)
    >>> splitter = FiscalYearGroupedSplitter(n_splits=3, gap_years=0, drop_repeat_donors=False)
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
    (one per gift).  ``FiscalYearGroupedSplitter`` uses ``groups`` to correctly
    assign all gifts from a given fiscal year to the same fold regardless
    of row order.

    **groups parameter convention:** Pass ``groups`` as an integer array of
    fiscal years (e.g., ``fiscal_years = df["fiscal_year"].to_numpy()``).
    The splitter sorts distinct values numerically and walks forward.

    **What this does not prevent.** The grouping unit is the fiscal year, not
    the donor. A donor with gifts in several fiscal years therefore appears in
    both the training and the test fold of the same split, in different rows.
    That is correct and intended for a *time-varying* target ("did this donor
    give in FY22?"), because the training rows precede the test rows. It is
    **leakage** for a *static per-donor* label such as ``is_major_donor``, where
    the same answer is attached to every one of that donor's rows and the model
    can memorise it from the training years.

    For that case, set ``drop_repeat_donors=True`` and pass ``groups`` as
    ``(n_samples, 2)`` with the donor identifier in column 1. Each test fold then
    excludes donors already present in its training rows. Aggregating to one row
    per donor and using a grouped holdout remains the cleaner option when the
    label has no time dimension at all.

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
        gap_years: int = 0,
        drop_repeat_donors: bool | str = "warn",
    ) -> None:
        # MUST call super().__init__() for BaseCrossValidator compat.
        self.n_splits = n_splits
        self.gap_years = gap_years
        self.drop_repeat_donors = drop_repeat_donors
        
        if self.drop_repeat_donors == "warn":
            warnings.warn(
                "The FiscalYearGroupedSplitter(drop_repeat_donors=...) default "
                "of False is deprecated and allows repeat donors across train "
                "and test folds. This default will change to True in 0.8.0. Pass "
                "drop_repeat_donors=False explicitly to silence this warning.",
                DeprecationWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Required abstract-method implementations
    # ------------------------------------------------------------------

    def _validate_params(self) -> tuple[int, int]:
        """Return ``(n_splits, gap_years)`` as validated ints.

        ``__init__`` stores raw parameters only, so both ``split`` and
        ``get_n_splits`` validate here. Without this, ``n_splits <= 0`` reached
        the ``unique_fy[-(n_splits):]`` slice, where a non-positive value flips
        the slice open-ended (``unique_fy[-(0):]`` is ``unique_fy[0:]``) and
        ``split`` silently yielded folds while ``get_n_splits`` reported zero or
        a negative count. ``cross_val_score`` sizes its result array from
        ``get_n_splits``, so the two disagreeing is a real failure, not a
        cosmetic one.
        """
        try:
            n_splits = int(self.n_splits)
            gap_years = int(self.gap_years)
        except (TypeError, ValueError):
            raise ValueError(
                "n_splits and gap_years must be integers, got "
                f"n_splits={self.n_splits!r}, gap_years={self.gap_years!r}."
            ) from None
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}.")
        if gap_years < 0:
            raise ValueError(f"gap_years must be >= 0, got {gap_years}.")
        return n_splits, gap_years

    def split(self, X, y=None, groups=None):
        """Generate (train_indices, test_indices) arrays.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.  Only ``X.shape[0]`` (i.e., the number of
            samples) is used; actual feature values are ignored.
        y : array-like of shape (n_samples,), optional
            Target labels.  Ignored; present for sklearn API compatibility.
        groups : array-like, **required**
            Integer fiscal year labels for each sample, shape
            ``(n_samples,)``. When ``drop_repeat_donors=True`` this must instead
            be ``(n_samples, 2)``: fiscal year in column 0, donor identifier in
            column 1.

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
        ValueError
            If ``drop_repeat_donors=True`` and ``groups`` is not
            ``(n_samples, 2)``, or its fiscal-year column is not numeric.
        ValueError
            If ``drop_repeat_donors=True`` empties a test fold entirely.
        """
        requested_splits, gap_years = self._validate_params()
        drop_repeat = False if self.drop_repeat_donors == "warn" else bool(self.drop_repeat_donors)

        if groups is None:
            raise ValueError(
                "FiscalYearGroupedSplitter requires `groups` to be an array of "
                "fiscal year labels (integer per sample).  Pass `groups=` to "
                "`split()` or to `cross_val_score(groups=...)`."
            )

        groups_arr = np.asarray(groups)
        donor_ids = None
        if drop_repeat:
            if groups_arr.ndim != 2 or groups_arr.shape[1] != 2:
                raise ValueError(
                    "drop_repeat_donors=True requires `groups` with shape "
                    "(n_samples, 2): column 0 the fiscal year, column 1 the "
                    f"donor identifier. Got shape {groups_arr.shape}."
                )
            donor_ids = groups_arr[:, 1]
            groups_arr = groups_arr[:, 0]
            # np.column_stack of integer years and string donor ids upcasts
            # everything to '<U21', which silently turns the fiscal years into
            # strings and fails much later with a bare numpy TypeError. Catch it
            # here, where the message can say what to do.
            try:
                groups_arr = groups_arr.astype(float)
            except (TypeError, ValueError):
                raise ValueError(
                    "drop_repeat_donors=True: column 0 of `groups` must be "
                    "numeric fiscal years, got dtype "
                    f"{np.asarray(groups)[:, 0].dtype!r}. If your donor ids are "
                    "strings, np.column_stack upcasts the whole array; pass a "
                    "pandas DataFrame, or factorise the ids to integers first."
                ) from None
        groups = column_or_1d(groups_arr)
        fiscal_years = groups

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
                f"FiscalYearGroupedSplitter requires at least 2 distinct fiscal "
                f"years in `groups`, found {n_fy}."
            )

        max_splits = n_fy - 1 - gap_years
        if max_splits < 1:
            raise ValueError(
                f"Not enough fiscal years ({n_fy}) for n_splits={requested_splits} "
                f"with gap_years={gap_years}.  Need at least "
                f"{requested_splits + 1 + gap_years} distinct fiscal years."
            )

        n_splits = min(requested_splits, max_splits)

        # Walk from the most-recent test year backward to get exactly n_splits folds.
        # Each fold tests a consecutive fiscal year, training on *all* prior years.
        indices = np.arange(n_samples)
        test_fy_sequence = unique_fy[-(n_splits):]  # Last n_splits fiscal years as test sets

        dropped_total = 0
        for test_fy in test_fy_sequence:
            test_mask = fiscal_years == test_fy

            # Training: all FYs strictly before (test_fy - gap_years)
            train_cutoff_fy = test_fy - gap_years
            train_mask = fiscal_years < train_cutoff_fy

            if not np.any(train_mask):
                # No training data before this test year, skip
                continue

            if donor_ids is not None:
                # A static per-donor label leaks through any donor present in
                # both folds. Drop those donors from the TEST side only: pulling
                # them out of training would discard history for no benefit.
                seen = np.isin(donor_ids, np.unique(donor_ids[train_mask]))
                # np.isin never matches NaN to NaN, so a row with a missing donor
                # id would be classified as unseen and kept. That is the wrong
                # default for a leakage guard: an unidentifiable donor cannot be
                # shown to be absent from training, so treat it as seen.
                unidentified = _missing_mask(donor_ids)
                seen = seen | unidentified
                n_dropped = int(np.count_nonzero(test_mask & seen))
                if n_dropped:
                    test_mask = test_mask & ~seen
                    dropped_total += n_dropped
                if n_dropped:
                    warnings.warn(
                        f"drop_repeat_donors=True removed {n_dropped} test "
                        f"row(s) from fiscal year {test_fy} whose donor already "
                        "appeared in training (or had no donor id). The test "
                        "donors that remain are systematically newer to the "
                        "file, so scores are not directly comparable to a run "
                        "without this flag.",
                        UserWarning,
                    )
                if not np.any(test_mask):
                    raise ValueError(
                        f"drop_repeat_donors=True emptied the test fold for "
                        f"fiscal year {test_fy}: every donor in it also appears "
                        "in the training years. Either the label is static per "
                        "donor and this splitter cannot help (aggregate to one "
                        "row per donor and use a grouped holdout), or the label "
                        "is time-varying and this flag is not needed."
                    )

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
        n_splits, gap_years = self._validate_params()
        if groups is not None:
            groups = np.asarray(groups)
            drop_repeat = False if self.drop_repeat_donors == "warn" else bool(self.drop_repeat_donors)
            if drop_repeat and groups.ndim == 2 and groups.shape[1] == 2:
                groups = groups[:, 0]
            unique_fy = np.unique(groups)
            n_fy = len(unique_fy)
            max_splits = max(0, n_fy - 1 - gap_years)
            return min(n_splits, max_splits)
        return n_splits

    # ------------------------------------------------------------------
    # sklearn clone safety: all params must be in __init__
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_splits={self.n_splits}, "
            f"gap_years={self.gap_years}, "
            f"drop_repeat_donors={self.drop_repeat_donors!r})"
        )


# ---------------------------------------------------------------------------
# Utility: resolve n_samples from various input types
# ---------------------------------------------------------------------------

def _missing_mask(values) -> np.ndarray:
    """True where a donor identifier is missing.

    ``np.isnan`` only works on float arrays, and donor ids are commonly object
    or string, so fall back to a pandas-free elementwise check.
    """
    arr = np.asarray(values)
    if arr.dtype.kind == "f":
        return np.isnan(arr)
    if arr.dtype.kind in "iub":
        return np.zeros(arr.shape, dtype=bool)
    return np.array([v is None or v != v for v in arr.ravel()]).reshape(arr.shape)


def _n_samples(X) -> int:
    """Return the number of samples from X, supporting ndarray and DataFrames."""
    if X is None:
        raise ValueError("X must not be None.")
    try:
        return X.shape[0]
    except AttributeError:
        return len(X)
