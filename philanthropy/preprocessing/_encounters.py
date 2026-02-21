"""
philanthropy.preprocessing._encounters
=======================================
Clinical-encounter feature engineering for medical philanthropy.

This module bridges the clinical data warehouse and the advancement CRM by
safely merging hospital encounter records (admission/discharge dates) with
philanthropic gift histories.  The resulting temporal features—
``days_since_last_discharge`` and ``encounter_frequency_score``—are strong
signals in major-gift propensity models trained for academic medical centres
(AMCs) and hospital foundations.

**Privacy note:**  All patient/donor identifier columns are explicitly removed
from the output array so that no PII (MRN, donor ID, etc.) can accidentally
flow through a fitted pipeline into a model artefact.

Typical usage
-------------
>>> import pandas as pd
>>> from philanthropy.preprocessing import EncounterTransformer
>>> enc_df = pd.DataFrame({
...     "donor_id":        [1, 2],
...     "discharge_date":  ["2023-05-10", "2022-11-01"],
... })
>>> gift_df = pd.DataFrame({
...     "donor_id":        [1, 2],
...     "gift_date":       ["2023-08-15", "2023-03-20"],
...     "gift_amount":     [5000.0, 500.0],
... })
>>> transformer = EncounterTransformer(
...     encounter_df=enc_df,
...     discharge_col="discharge_date",
...     gift_date_col="gift_date",
...     merge_key="donor_id",
... )
>>> features = transformer.fit_transform(gift_df)
>>> list(features.columns)  # doctest: +NORMALIZE_WHITESPACE
['gift_amount', 'days_since_last_discharge', 'encounter_frequency_score']
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted

from philanthropy.base import BasePhilanthropyEstimator


# ---------------------------------------------------------------------------
# EncounterTransformer
# ---------------------------------------------------------------------------


class EncounterTransformer(BasePhilanthropyEstimator, TransformerMixin):
    """Merge clinical encounter history into philanthropic feature matrices.

    Given a lookup ``encounter_df`` containing at least one discharge date per
    donor, this transformer enriches a gift-level DataFrame with two continuous
    temporal features:

    ``days_since_last_discharge``
        Integer number of days between the donor's **most recent** discharge
        date (observed at :meth:`fit` time) and the ``gift_date`` in ``X``.
        Negative values indicate gifts made *before* discharge (pre-admission
        solicitations are uncommon and are flagged as ``NaN`` by default unless
        ``allow_negative_days=True``).
    ``encounter_frequency_score``
        Log-scaled count of distinct encounter records for the donor.  Because
        the distribution of encounter counts is highly right-skewed in real AMC
        data, the log transform normalises the feature for downstream linear
        models.  Donors with zero encounters receive a score of ``0.0``.

    All identifier columns (``merge_key`` and any column whose name contains
    common PII substrings: ``"id"``, ``"mrn"``, ``"ssn"``, ``"name"``,
    ``"dob"``, ``"zip"``) are silently dropped from the output DataFrame before
    it is returned, preventing accidental downstream leakage.

    Parameters
    ----------
    encounter_df : pd.DataFrame
        Reference table of clinical encounters.  Must contain ``merge_key``
        and ``discharge_col``.  Additional columns are ignored.
    discharge_col : str, default="discharge_date"
        Column in ``encounter_df`` holding ISO-8601 discharge timestamps.
    gift_date_col : str, default="gift_date"
        Column in ``X`` (the gift-level DataFrame) holding ISO-8601 gift
        dates.
    merge_key : str, default="donor_id"
        Column name present in **both** ``encounter_df`` and ``X`` used to
        join the two tables.  This column is dropped from the output.
    allow_negative_days : bool, default=False
        If ``False`` (recommended), ``days_since_last_discharge`` values
        below zero are coerced to ``NaN``, indicating that the gift predates
        the discharge.  Set to ``True`` only for retrospective analyses where
        pre-admission gifts are meaningful.
    id_cols_to_drop : list of str or None, default=None
        Additional column names to explicitly drop on output, beyond those
        detected via the PII heuristic.  Useful when non-standard identifiers
        (e.g., ``"pledge_record_key"``) are present in ``X``.
    fiscal_year_start : int, default=7
        Month (1–12) that begins the organisation's fiscal year.  Inherited
        from :class:`~philanthropy.base.BasePhilanthropyEstimator` for
        pipeline compatibility.

    Attributes
    ----------
    encounter_summary_ : pd.DataFrame
        Per-donor summary table (indexed by ``merge_key``) with columns
        ``last_discharge`` (Timestamp) and ``encounter_count`` (int), computed
        at :meth:`fit` time.
    dropped_cols_ : list of str
        Names of the columns that were removed from ``X`` during the last
        :meth:`transform` call for audit/logging purposes.
    n_features_in_ : int
        Number of columns seen in ``X`` at :meth:`fit` time.
    feature_names_in_ : ndarray of str
        Column names of ``X`` at :meth:`fit` time.

    Raises
    ------
    ValueError
        If ``merge_key`` is absent from ``encounter_df`` or from ``X``.
    ValueError
        If ``discharge_col`` is absent from ``encounter_df``.

    Examples
    --------
    >>> import pandas as pd
    >>> from philanthropy.preprocessing import EncounterTransformer
    >>> enc = pd.DataFrame({
    ...     "donor_id":       [1, 1, 2],
    ...     "discharge_date": ["2022-01-01", "2023-06-15", "2022-09-30"],
    ... })
    >>> gifts = pd.DataFrame({
    ...     "donor_id":    [1, 2, 3],
    ...     "gift_date":   ["2023-08-01", "2023-01-01", "2023-05-01"],
    ...     "gift_amount": [10000.0, 750.0, 250.0],
    ... })
    >>> t = EncounterTransformer(encounter_df=enc, merge_key="donor_id")
    >>> out = t.fit_transform(gifts)
    >>> "donor_id" not in out.columns
    True
    >>> "days_since_last_discharge" in out.columns
    True
    >>> "encounter_frequency_score" in out.columns
    True
    """

    # Heuristic substrings used to detect PII-like column names (case-insensitive)
    _PII_SUBSTRINGS: List[str] = ["_id", "mrn", "ssn", "name", "dob", "zip"]

    def __init__(
        self,
        encounter_df: pd.DataFrame,
        discharge_col: str = "discharge_date",
        gift_date_col: str = "gift_date",
        merge_key: str = "donor_id",
        allow_negative_days: bool = False,
        id_cols_to_drop: Optional[List[str]] = None,
        fiscal_year_start: int = 7,
    ) -> None:
        # scikit-learn rule: __init__ stores parameters and does NO logic.
        super().__init__(fiscal_year_start=fiscal_year_start)
        self.encounter_df = encounter_df
        self.discharge_col = discharge_col
        self.gift_date_col = gift_date_col
        self.merge_key = merge_key
        self.allow_negative_days = allow_negative_days
        self.id_cols_to_drop = id_cols_to_drop

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_encounter_df(self) -> None:
        """Raise ``ValueError`` if ``encounter_df`` is structurally invalid."""
        if not isinstance(self.encounter_df, pd.DataFrame):
            raise TypeError(
                f"`encounter_df` must be a pd.DataFrame, "
                f"got {type(self.encounter_df).__name__!r}."
            )
        for col, label in [
            (self.merge_key, "merge_key"),
            (self.discharge_col, "discharge_col"),
        ]:
            if col not in self.encounter_df.columns:
                raise ValueError(
                    f"Column {col!r} (specified as `{label}`) was not found "
                    f"in `encounter_df`. Available columns: "
                    f"{list(self.encounter_df.columns)}."
                )

    def _validate_X(self, X: pd.DataFrame) -> None:
        """Raise ``ValueError`` if gift DataFrame ``X`` lacks required columns."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"`X` must be a pd.DataFrame, got {type(X).__name__!r}."
            )
        for col, label in [
            (self.merge_key, "merge_key"),
            (self.gift_date_col, "gift_date_col"),
        ]:
            if col not in X.columns:
                raise ValueError(
                    f"Column {col!r} (specified as `{label}`) was not found "
                    f"in ``X``. Available columns: {list(X.columns)}."
                )

    # ------------------------------------------------------------------
    # Column-drop utilities
    # ------------------------------------------------------------------

    def _identify_pii_columns(self, columns: pd.Index) -> List[str]:
        """Return column names that match PII heuristics or explicit drop list."""
        explicit = list(self.id_cols_to_drop or [])
        heuristic = [
            c for c in columns
            if any(sub in c.lower() for sub in self._PII_SUBSTRINGS)
        ]
        # Always include the merge key itself
        merge_key_set = {self.merge_key}
        combined = set(explicit) | set(heuristic) | merge_key_set
        # Only drop columns that actually exist
        return [c for c in columns if c in combined]

    # ------------------------------------------------------------------
    # fit / transform
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None) -> "EncounterTransformer":
        """Compute per-donor encounter summaries from ``encounter_df``.

        The fitted artefact ``encounter_summary_`` is a lightweight per-donor
        lookup containing the most-recent discharge date and total encounter
        count.  No information from ``X`` flows into this summary, which
        prevents temporal data leakage when the transformer is placed **before**
        a time-based train/test split inside a pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Gift-level DataFrame.  Used only to infer ``feature_names_in_``
            and ``n_features_in_``; no target statistics are extracted.
        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        self : EncounterTransformer
            Fitted transformer instance.

        Raises
        ------
        ValueError
            If required columns are missing from ``encounter_df`` or ``X``.
        """
        self._validate_fiscal_year_start()
        self._validate_encounter_df()
        self._validate_X(X)

        # Record input schema (sklearn convention)
        self.feature_names_in_ = np.array(X.columns.tolist(), dtype=object)
        self.n_features_in_ = len(X.columns)

        # --- Build encounter summary (fit-time only, no leakage from X) ---
        enc = self.encounter_df[[self.merge_key, self.discharge_col]].copy()
        enc[self.discharge_col] = pd.to_datetime(
            enc[self.discharge_col], errors="coerce"
        )

        missing_discharge = enc[self.discharge_col].isna().sum()
        if missing_discharge > 0:
            warnings.warn(
                f"{missing_discharge} encounter row(s) had unparseable "
                f"`discharge_col` values and were excluded from the summary.",
                UserWarning,
            )

        enc = enc.dropna(subset=[self.discharge_col])

        self.encounter_summary_ = enc.groupby(self.merge_key).agg(
            last_discharge=(self.discharge_col, "max"),
            encounter_count=(self.discharge_col, "count"),
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append encounter features and strip identifying columns.

        Parameters
        ----------
        X : pd.DataFrame
            Gift-level DataFrame.  Must contain ``merge_key`` and
            ``gift_date_col``.

        Returns
        -------
        X_out : pd.DataFrame
            Enriched DataFrame with two new columns:

            * ``days_since_last_discharge`` — Days elapsed between the donor's
              latest discharge and the gift date.  ``NaN`` for donors absent
              from the encounter table or (when ``allow_negative_days=False``)
              for gifts dated before discharge.
            * ``encounter_frequency_score`` — ``log1p(encounter_count)``.
              ``0.0`` for donors with no recorded encounters.

            All identifier-like columns (including ``merge_key``) are removed.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        ValueError
            If ``merge_key`` or ``gift_date_col`` is absent from ``X``.
        """
        check_is_fitted(self, ["encounter_summary_"])
        self._validate_X(X)

        X_out = X.copy()
        X_out[self.gift_date_col] = pd.to_datetime(
            X_out[self.gift_date_col], errors="coerce"
        )

        # --- Merge the encounter summary ---
        X_out = X_out.merge(
            self.encounter_summary_.reset_index(),
            on=self.merge_key,
            how="left",
        )

        # --- days_since_last_discharge ---
        days_delta = (
            X_out[self.gift_date_col] - X_out["last_discharge"]
        ).dt.days.astype("float64")

        if not self.allow_negative_days:
            days_delta = days_delta.where(days_delta >= 0, other=np.nan)

        X_out["days_since_last_discharge"] = days_delta

        # --- encounter_frequency_score: log1p-scaled count ---
        X_out["encounter_frequency_score"] = np.log1p(
            X_out["encounter_count"].fillna(0).astype("float64")
        )

        # --- Drop temporary merge columns ---
        X_out = X_out.drop(columns=["last_discharge", "encounter_count"], errors="ignore")

        # --- Strip identifiers (privacy firewall) ---
        cols_to_drop = self._identify_pii_columns(X_out.columns)
        self.dropped_cols_ = cols_to_drop
        if cols_to_drop:
            X_out = X_out.drop(columns=cols_to_drop, errors="ignore")

        # --- Also drop the gift_date column (datetime, not modellable directly) ---
        if self.gift_date_col in X_out.columns:
            X_out = X_out.drop(columns=[self.gift_date_col])

        return X_out.reset_index(drop=True)
