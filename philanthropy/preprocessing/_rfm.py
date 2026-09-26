from __future__ import annotations

import warnings
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import Tags
from sklearn.utils.validation import check_is_fitted

from ._encounters import _apply_as_of_cutoff

_Self = TypeVar("_Self", bound="RFMTransformer")


class RFMTransformer(TransformerMixin, BaseEstimator):
    """
    Transforms transaction logs into Recency, Frequency, and Monetary (RFM) features.

    This is a **pre-pipeline aggregation step, not a pipeline member.** It takes
    one row per gift and returns one row per donor, so the sample count changes
    across ``transform``. Putting it inside a
    :class:`~sklearn.pipeline.Pipeline` ahead of an estimator raises
    ``ValueError: Found input variables with inconsistent numbers of samples``,
    because ``y`` is still gift-shaped. Run it first, then build the pipeline on
    its donor-level output. It is exempt from the ``check_estimator`` battery
    for the same reason, with hand-written coverage in
    ``tests/test_sklearn_compliance.py``.

    Parameters
    ----------
    reference_date : str or datetime-like, default=None
        The date used as the reference point to calculate recency.
        If None, the maximum gift_date in the dataframe is used.
    agg_func : str or callable, default='sum'
        The aggregation function to calculate the monetary value. 
        Typical values are 'sum' (cumulative) or 'mean' (average).
    as_of : str, datetime-like or None, default=None
        Scoring-date cutoff. Gifts dated after ``as_of`` are dropped before any
        roll-up, so ``frequency`` and ``monetary`` describe only what had
        happened by that date. Left at ``None`` the whole gift table is
        aggregated, which inflates the roll-up with gifts that postdate the
        decision being modelled: the classic case is a new-donor model whose
        cumulative-giving feature is really the outcome. ``None`` warns rather
        than silently aggregating the future whenever the table runs past the
        frozen reference date.
    include_tenure : bool, default=False
        Emit a fifth column, ``tenure``: days from the donor's *first* gift to
        the frozen reference date. Recency-frequency-monetary alone cannot feed
        a buy-till-you-die model, which needs the observation window T as well
        [Fader, Hardie and Lee 2005]; ``tenure`` is that T. Defaults to False so
        the output shape does not change under existing callers, and will become
        the default in the next major release.

    Notes
    -----
    Output ``recency``, ``frequency``, and ``monetary`` are raw counts and
    sums, not scores or frozen bins: callers who want quintile-style RFM
    scores bin these columns themselves downstream.

    A gift with a NaN ``gift_amount`` is excluded from **both**
    ``frequency`` and ``monetary``, not just ``monetary``: an unknown amount
    means the gift's contribution to either column is unknown, so it is
    dropped from the count as well as the sum, with a ``UserWarning`` naming
    how many rows were dropped. This means ``frequency`` can be lower than
    the donor's raw row count in the input.

    ``X`` must be a ``pandas.DataFrame`` with named columns; a bare numpy
    array has no ``donor_id`` / ``gift_date`` / ``gift_amount`` to key off of
    and is rejected in :meth:`fit` with a ``TypeError``.
    """
    def __init__(
        self,
        reference_date: Any = None,
        agg_func: Any = 'sum',
        include_tenure: bool = False,
        as_of: Any = None,
    ) -> None:
        self.reference_date = reference_date
        self.agg_func = agg_func
        self.include_tenure = include_tenure
        self.as_of = as_of

    def fit(self: _Self, X: Any, y: Any = None) -> _Self:
        """Fit the transformer by validating input and freezing the reference date.
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Transaction log with required columns ``donor_id``, ``gift_date``, and
            ``gift_amount``. Must be a ``pandas.DataFrame`` with those columns
            named; a bare numpy array has no way to name them and is rejected.
        y : ignored
            Present for scikit-learn API compatibility.
        Returns
        -------
        self : RFMTransformer
            Fitted transformer with the following attributes frozen:

            * ``feature_names_in_`` : ndarray of str
                Column names from ``X``.
            * ``n_features_in_`` : int
                Number of features in ``X``.
            * ``reference_date_`` : datetime
                The reference date for recency calculation. If ``reference_date``
                was provided in the constructor, it is used directly. Otherwise,
                it is computed as the maximum ``gift_date`` in ``X`` (after any
                ``as_of`` cutoff is applied) and frozen to ensure consistent
                recency calculations across :meth:`transform` calls on different
                batches.
        Raises
        ------
        TypeError
            If ``X`` is not a pandas DataFrame.
        ValueError
            If ``X`` is missing any of the required columns ``donor_id``,
            ``gift_date``, or ``gift_amount``.
        """
        if not hasattr(X, "columns"):
            raise TypeError(
                "RFMTransformer requires a pandas DataFrame with named "
                "columns (donor_id, gift_date, gift_amount); a numpy array "
                "has no column names to validate against."
            )
        self.feature_names_in_ = np.array(X.columns.tolist(), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)

        self._validate_input(X)

        # Freeze the recency reference date from TRAINING data (leakage-safety
        # contract: fitted statistics are computed in fit and frozen before
        # transform). Mirrors EncounterRecencyTransformer.reference_date_.
        # Cut in fit as well as transform, so an unparseable as_of is caught
        # where every other parameter is validated, and so an unset
        # reference_date falls on the last gift the as_of window allows rather
        # than the last gift on file.
        X_cut = self._cut(X)
        if self.reference_date is not None:
            self.reference_date_ = pd.to_datetime(self.reference_date)
        else:
            self.reference_date_ = X_cut["gift_date"].max()
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        """Transform transaction logs into Recency, Frequency, and Monetary features.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Transaction log with required columns ``donor_id``, ``gift_date``, and
            ``gift_amount``. Rows are gift-level; output is donor-level. Must be
            a ``pandas.DataFrame`` with those columns named; a bare numpy array
            is rejected.
        Returns
        -------
        rfm_df : pd.DataFrame of shape (n_donors, 4 or 5)
            Donor-level RFM features, as raw values (not scores or bins):
            * ``donor_id`` : object
                Unique donor identifier.
            * ``recency`` : int64
                Days since each donor's most recent gift, relative to the frozen
                ``reference_date_``.
            * ``frequency`` : int64
                Total number of gifts per donor with a known ``gift_amount``, in
                the (possibly ``as_of``-filtered) transaction log. Gifts with a
                NaN ``gift_amount`` are excluded from both ``frequency`` and
                ``monetary``, with a warning, so the two counts describe the
                same set of gifts.
            * ``monetary`` : float64
                Aggregated gift amount per donor (sum, mean, or other function
                specified by ``agg_func``), over gifts with a known
                ``gift_amount``.
            * ``tenure`` : int64
                *Optional, present only if ``include_tenure=True``.*
                Days from each donor's first gift to the frozen ``reference_date_``.
        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called.
        TypeError
            If ``X`` is not a pandas DataFrame.
        ValueError
            If ``X`` is missing any of the required columns ``donor_id``,
            ``gift_date``, or ``gift_amount``.
        """
        check_is_fitted(self)
        if not hasattr(X, "columns"):
            raise TypeError(
                "RFMTransformer requires a pandas DataFrame with named "
                "columns (donor_id, gift_date, gift_amount); a numpy array "
                "has no column names to validate against."
            )
        # Manual validation
        self._validate_input(X)

        X_df = self._cut(X)
        self._warn_if_unbounded(X_df)

        # Use the reference date frozen in fit, never the transform batch's
        # max, which would make recency depend on which rows share the batch.
        ref_date = self.reference_date_

        grouped = X_df.groupby('donor_id')

        # Recency: Days since the last gift relative to reference_date
        last_gift = grouped['gift_date'].max()
        recency = (ref_date - last_gift).dt.days

        # Frequency and monetary must describe the same gifts: a gift with an
        # unknown amount doesn't count toward either, so it isn't silently
        # counted in frequency while being dropped from monetary.
        has_amount = X_df['gift_amount'].notna()
        n_missing_amount = int((~has_amount).sum())
        if n_missing_amount:
            warnings.warn(
                f"RFMTransformer is excluding {n_missing_amount} gift row(s) "
                "with a NaN gift_amount from both frequency and monetary, so "
                "the two features describe the same set of gifts.",
                UserWarning,
                stacklevel=2,
            )
        grouped_valid = X_df[has_amount].groupby('donor_id')

        # Frequency: number of gifts with a known amount
        frequency = grouped_valid['gift_date'].count().reindex(
            recency.index, fill_value=0
        )

        # Monetary: aggregated gift amount depending on agg_func, over gifts
        # with a known amount. A donor with no such gifts has no defined
        # aggregate, so it stays NaN rather than being coerced to 0.
        monetary = grouped_valid['gift_amount'].agg(self.agg_func).reindex(
            recency.index
        )

        rfm_df = pd.DataFrame({
            'donor_id': recency.index,
            'recency': recency.values,
            'frequency': frequency.values,
            'monetary': monetary.values
        })

        if self.include_tenure:
            # T for a buy-till-you-die model: the donor's observation window,
            # measured from the first gift to the same frozen reference date
            # recency uses, so the two are on one clock.
            first_gift = grouped['gift_date'].min()
            rfm_df['tenure'] = (ref_date - first_gift).dt.days.values

        return rfm_df
        
    def _cut(self, X: Any) -> pd.DataFrame:
        """Copy ``X``, parse ``gift_date``, and drop gifts after ``as_of``."""
        X_df = X.copy()
        X_df['gift_date'] = pd.to_datetime(X_df['gift_date'])
        return _apply_as_of_cutoff(
            X_df, 'gift_date', self.as_of, "RFMTransformer", row_noun="gift"
        )

    def _warn_if_unbounded(self, X_df: pd.DataFrame) -> None:
        """Warn when ``as_of`` is unset and gifts postdate the reference date.

        Those gifts inflate ``frequency`` and ``monetary`` with money that had
        not been given yet on the scoring date, and drive ``recency`` negative.
        No cross-validation splitter catches it, because the leak is inside a
        single donor's roll-up rather than across folds.
        """
        if self.as_of is not None or not len(X_df):
            return
        n_future = int((X_df['gift_date'] > self.reference_date_).sum())
        if n_future:
            warnings.warn(
                f"RFMTransformer(as_of=None) is aggregating {n_future} gift "
                f"row(s) dated after the frozen reference date "
                f"({self.reference_date_.date()}), so frequency and monetary "
                "include gifts that postdate the decision they describe and "
                "recency goes negative. Set as_of to the end of your training "
                "window, or restrict the gift table before fit. See "
                "docs/tutorials/avoiding_temporal_data_leakage.md.",
                UserWarning,
                stacklevel=2,
            )

    def _validate_input(self, X: Any) -> None:
        required_cols = {"donor_id", "gift_date", "gift_amount"}
        if not required_cols.issubset(X.columns):
            raise ValueError(f"X must contain columns: {required_cols}")

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return the donor identifier and generated RFM feature names.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Ignored because the output columns are fixed by
            ``include_tenure``, not by the input.

        Returns
        -------
        feature_names_out : ndarray of str
            ``["donor_id", "recency", "frequency", "monetary"]``, plus
            ``"tenure"`` when ``include_tenure=True``.

        Raises
        ------
        NotFittedError
            If the transformer has not been fitted.
        """
        check_is_fitted(self)
        names = ['donor_id', 'recency', 'frequency', 'monetary']
        if self.include_tenure:
            names.append('tenure')
        return np.array(names, dtype=object)

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        return tags
