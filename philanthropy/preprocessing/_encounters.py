"""
philanthropy.preprocessing._encounters
=======================================
Clinical-encounter feature engineering for medical philanthropy.

This module bridges the clinical data warehouse and the advancement CRM by
safely merging hospital encounter records (admission/discharge dates) with
philanthropic gift histories.  The resulting temporal features,
``days_since_last_discharge`` and ``encounter_frequency_score``, are strong
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
>>> transformer.set_output(transform="pandas")  # doctest: +ELLIPSIS
EncounterTransformer(...)
>>> features = transformer.fit_transform(gift_df)
>>> list(features.columns)  # doctest: +NORMALIZE_WHITESPACE
['gift_amount', 'days_since_last_discharge', 'encounter_frequency_score']
"""

from __future__ import annotations

import warnings
from typing import Any, List, TypeVar

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import Tags
from sklearn.utils.validation import check_is_fitted, validate_data

_Self = TypeVar("_Self", bound="EncounterTransformer")


# ---------------------------------------------------------------------------
# EncounterTransformer
# ---------------------------------------------------------------------------


def _apply_as_of_cutoff(
    enc: pd.DataFrame, discharge_col: str, as_of: Any, class_name: str
) -> pd.DataFrame:
    """Drop encounter rows discharged after ``as_of``.

    Returns ``enc`` unchanged when ``as_of`` is ``None``. Rows whose discharge
    date did not parse are kept, so each caller keeps its own rule for those and
    the cutoff cannot silently change it.
    """
    if as_of is None:
        return enc
    try:
        cutoff = pd.Timestamp(as_of)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{class_name}(as_of=...) must be a parseable date, got {as_of!r}."
        ) from exc
    if pd.isna(cutoff):
        raise ValueError(
            f"{class_name}(as_of=...) must be a parseable date, got {as_of!r}."
        )
    keep = enc[discharge_col].isna() | (enc[discharge_col] <= cutoff)
    if not keep.any() and len(enc):
        warnings.warn(
            f"{class_name}(as_of={cutoff.date()}) excluded every encounter row, "
            "so the summary is empty and every donor scores as having no "
            "encounters. Check that as_of is later than your encounter history.",
            UserWarning,
        )
    return enc[keep]


def _avoid_dtype_promotion(X: Any) -> Any:
    """Cast a DataFrame with a parsed datetime column to ``object`` dtype.

    ``validate_data`` converts ``X`` to a single-dtype array, and numpy cannot
    promote ``datetime64`` together with ``int``/``float`` into one dtype. A
    caller's own loader (``pd.read_csv(parse_dates=...)``, a SQL read,
    ``philanthropy.datasets.make_donor_panel``) hands back exactly this
    column mix, so ``gift_date_col`` must not need pre-formatting to strings.
    """
    if isinstance(X, pd.DataFrame) and any(
        pd.api.types.is_datetime64_any_dtype(dt) for dt in X.dtypes
    ):
        return X.astype(object)
    return X


def _warn_if_unbounded(
    enc: pd.DataFrame, discharge_col: str, gift_dates: Any, class_name: str
) -> None:
    """Warn when ``as_of`` is unset and the encounter table runs past the gifts.

    The default ``as_of=None`` aggregates the whole encounter table, which is
    only safe if the table was already restricted upstream. When it is not, the
    features are built from encounters that had not happened yet at the point
    the gift decision was made, and no cross-validation splitter can see that:
    the encounter table is a constructor argument, so its rows are never part of
    any split.
    """
    if gift_dates is None or not len(enc):
        return
    latest_gift = gift_dates.max()
    if pd.isna(latest_gift):
        return
    n_future = int((enc[discharge_col] > latest_gift).sum())
    if n_future:
        warnings.warn(
            f"{class_name}(as_of=None) is aggregating {n_future} encounter "
            f"row(s) discharged after the latest gift date in X "
            f"({latest_gift.date()}), so features are built from encounters "
            "that postdate the decision they describe. Set as_of to the end of "
            "your training window, or restrict the encounter table before fit. "
            "See docs/tutorials/avoiding_temporal_data_leakage.md.",
            UserWarning,
            stacklevel=2,
        )


class EncounterTransformer(TransformerMixin, BaseEstimator):
    """Merge clinical encounter history into philanthropic feature matrices.

    Given a lookup ``encounter_df`` containing at least one discharge date per
    donor, this transformer enriches a gift-level DataFrame with two continuous
    temporal features:

    ``days_since_last_discharge``
        Days between the donor's **most recent** discharge date (observed at
        :meth:`fit` time) and the ``gift_date`` in ``X``. ``float64``, not an
        integer: the column has to carry ``NaN`` for donors absent from the
        encounter table and, when ``allow_negative_days=False``, for gifts dated
        before discharge. Do not cast it to an integer type, because that
        discards the missingness, which is itself signal here. Negative values
        (gifts made before discharge) survive only when
        ``allow_negative_days=True``.
    ``encounter_frequency_score``
        ``log1p`` of the number of encounter **rows** for the donor. Two things
        this is not: it is not a count, because of the log transform, and it is
        not a count of *distinct* encounters, because repeated rows for the same
        donor each add one. A donor with three rows on two dates scores
        ``log1p(3)``, not ``log1p(2)``. The log transform is there because the
        distribution of encounter counts is strongly right-skewed in real AMC
        data. Donors with no encounters score ``0.0``.

    Identifier columns (``merge_key`` plus any column whose name contains a
    substring in :attr:`PII_PATTERNS`) are dropped from the output before it is
    returned, as a defense-in-depth guard against accidental downstream leakage.
    This is a **name-based heuristic, not de-identification**: it inspects column
    *names* only (never cell values) and can miss identifiers whose names it does
    not recognise. See ``docs/explanation/compliance_considerations.md``. Extend
    or replace the patterns via the ``pii_patterns`` parameter.

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

        The join keys on the donor's **latest** discharge, so a donor with a
        later encounter than the gift being scored is measured against that
        later encounter (bounded by ``as_of``, and coerced to ``NaN`` unless
        ``allow_negative_days``). Per-gift index-encounter keying is not
        implemented: it would require the raw encounter rows at transform time,
        which is exactly what ``__getstate__`` keeps out of saved bundles.
    allow_negative_days : bool, default=False
        If ``False`` (recommended), ``days_since_last_discharge`` values
        below zero are coerced to ``NaN``, indicating that the gift predates
        the discharge.  Set to ``True`` only for retrospective analyses where
        pre-admission gifts are meaningful.
    id_cols_to_drop : list of str or None, default=None
        Additional column names to explicitly drop on output, beyond those
        detected via the PII heuristic.  Useful when non-standard identifiers
        (e.g., ``"pledge_record_key"``) are present in ``X``.
    pii_patterns : tuple of str or None, default=None
        Case-insensitive substrings used to flag identifier-like column names
        for dropping. If ``None``, the class-level :attr:`PII_PATTERNS` default
        is used. Provide your own tuple to broaden or narrow the heuristic: it
        replaces (does not extend) the default when set.
    as_of : str, datetime-like or None, default=None
        As-of cutoff for the encounter table. Encounters discharged **after**
        this date are excluded from ``encounter_summary_`` at :meth:`fit` time.
        ``None`` (the default) uses the whole table, which is only correct when
        every row of ``encounter_df`` was already observable at the point the
        solicitation decision is being modelled. For walk-forward evaluation,
        set this to the last day of the training window: without it, a gift dated
        2020 is scored against encounters recorded in 2024, and
        ``days_since_last_discharge`` is measured from a discharge that had not
        happened yet.

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
    >>> t.set_output(transform="pandas")  # doctest: +ELLIPSIS
    EncounterTransformer(...)
    >>> out = t.fit_transform(gifts)
    >>> "donor_id" not in out.columns
    True
    >>> "days_since_last_discharge" in out.columns
    True
    >>> "encounter_frequency_score" in out.columns
    True
    """

    # Heuristic substrings used to detect PII-like column names (case-insensitive).
    # Defense-in-depth, NOT a de-identification guarantee: matches column *names*
    # only (never cell values) and can miss identifiers whose names it does not
    # recognise. See docs/explanation/compliance_considerations.md.
    PII_PATTERNS = (
        "_id", "mrn", "ssn", "name", "dob", "birth", "zip",
        "patient", "phone", "email", "address",
    )

    def __init__(
        self,
        encounter_df: pd.DataFrame | None = None,
        encounter_path: str | None = None,
        discharge_col: str = "discharge_date",
        gift_date_col: str = "gift_date",
        merge_key: str = "donor_id",
        allow_negative_days: bool = False,
        id_cols_to_drop: list[str] | None = None,
        pii_patterns: tuple[str, ...] | None = None,
        as_of: Any = None,
    ) -> None:
        self.encounter_df = encounter_df
        self.encounter_path = encounter_path
        self.discharge_col = discharge_col
        self.gift_date_col = gift_date_col
        self.merge_key = merge_key
        self.allow_negative_days = allow_negative_days
        self.id_cols_to_drop = id_cols_to_drop
        self.pii_patterns = pii_patterns
        self.as_of = as_of

    def __getstate__(self) -> dict:
        """Drop the raw encounter table from pickles and joblib bundles.

        ``transform`` reads only ``encounter_summary_``, the per-donor aggregate
        frozen at :meth:`fit` time. ``encounter_df`` is the PHI-bearing *input*,
        so persisting it would make every saved model a patient-data disclosure:
        a bundle handed to a vendor, attached to a ticket, or copied to a laptop
        would carry the raw clinical rows with it. It is therefore replaced with
        ``None`` on serialisation.

        A round-tripped instance can still ``transform``. It cannot ``fit``
        again until it is given the table back, which is the intended
        trade-off. :func:`sklearn.base.clone` is unaffected, because clone goes
        through ``get_params`` rather than pickle.

        The bundle still contains ``encounter_summary_``: per-donor aggregates
        keyed by ``merge_key``. That is the minimum ``transform`` needs, and it
        is derived rather than raw, but it is not nothing. Treat a saved bundle
        as donor data.
        """
        state = dict(super().__getstate__())
        state["encounter_df"] = None
        return state

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_encounter_df(self, raw_enc: pd.DataFrame) -> None:
        """Raise ``ValueError`` if ``encounter_df`` is structurally invalid."""
        if not isinstance(raw_enc, pd.DataFrame):
            raise TypeError(
                f"`encounter_df` must be a pd.DataFrame, "
                f"got {type(raw_enc).__name__!r}."
            )
        for col, label in [
            (self.merge_key, "merge_key"),
            (self.discharge_col, "discharge_col"),
        ]:
            if col not in raw_enc.columns:
                raise ValueError(
                    f"Column {col!r} (specified as `{label}`) was not found "
                    f"in `encounter_df`. Available columns: "
                    f"{list(raw_enc.columns)}."
                )

    def _validate_X(self, X: pd.DataFrame) -> None:
        """Raise ``ValueError`` if gift DataFrame ``X`` lacks required columns."""
        if not isinstance(X, pd.DataFrame):
            return  # validate_data will handle non-DataFrame inputs
        for col, label in [
            (self.merge_key, "merge_key"),
            (self.gift_date_col, "gift_date_col"),
        ]:
            if col not in X.columns:
                raise ValueError(
                    f"Required column {col!r} (specified as `{label}`) was not found "
                    f"in input X. Please ensure X contains this column or update "
                    f"the `{label}` parameter in EncounterTransformer."
                )


    # ------------------------------------------------------------------
    # Column-drop utilities
    # ------------------------------------------------------------------

    def _identify_pii_columns(self, columns: pd.Index) -> List[str]:
        """Return column names that match PII heuristics or explicit drop list."""
        explicit = list(self.id_cols_to_drop or [])
        patterns = (
            self.pii_patterns if self.pii_patterns is not None else self.PII_PATTERNS
        )
        heuristic = [
            c for c in columns
            if any(sub in c.lower() for sub in patterns)
        ]
        # Always include the merge key itself
        merge_key_set = {self.merge_key}
        combined = set(explicit) | set(heuristic) | merge_key_set
        # Only drop columns that actually exist
        return [c for c in columns if c in combined]

    # ------------------------------------------------------------------
    # fit / transform
    # ------------------------------------------------------------------

    def fit(self: _Self, X: pd.DataFrame, y: Any = None) -> _Self:
        """Compute per-donor encounter summaries from ``encounter_df``.

        The fitted artefact ``encounter_summary_`` is a lightweight per-donor
        lookup containing the most-recent discharge date and total encounter
        count.  No information from ``X`` flows into this summary, so the
        summary is identical whether it is fitted on a training split or the
        full frame, and ``transform`` is idempotent.

        .. warning::
           That is the only leakage guarantee here. With the default
           ``as_of=None`` the summary aggregates **every** row of
           ``encounter_df``, so a gift dated 2020 is scored against encounters
           recorded in 2024 if the table contains them. Set ``as_of`` to the end
           of your training window, or restrict ``encounter_df`` yourself before
           calling ``fit``. When ``as_of`` is ``None`` and the table does contain
           discharges later than the latest gift date in ``X``, ``fit`` emits a
           :class:`UserWarning` naming the row count rather than proceeding
           silently.

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
        if self.encounter_path is not None:
            from ..utils._validation import ensure_local_path

            ensure_local_path(self.encounter_path, "encounter_path")
            raw_enc = pd.read_parquet(self.encounter_path)
        elif self.encounter_df is not None:
            raw_enc = self.encounter_df.copy()
        else:
            raise ValueError(
                "EncounterTransformer requires either encounter_df or "
                "encounter_path to be set."
            )

        self._validate_encounter_df(raw_enc)

        self._validate_X(X)
        gift_dates = (
            pd.to_datetime(X[self.gift_date_col], errors="coerce")
            if isinstance(X, pd.DataFrame) and self.gift_date_col in X.columns
            else None
        )
        X = validate_data(
            self, _avoid_dtype_promotion(X), dtype=None, ensure_all_finite="allow-nan", reset=True
        )
        self.n_features_in_ = X.shape[1]

        # --- Build encounter summary (fit-time only, no leakage from X) ---
        enc = raw_enc[[self.merge_key, self.discharge_col]].copy()
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
        if self.as_of is None:
            _warn_if_unbounded(
                enc, self.discharge_col, gift_dates, "EncounterTransformer"
            )
        enc = _apply_as_of_cutoff(
            enc, self.discharge_col, self.as_of, "EncounterTransformer"
        )

        self.encounter_summary_ = enc.groupby(self.merge_key).agg(
            last_discharge=(self.discharge_col, "max"),
            encounter_count=(self.discharge_col, "count"),
        )

        if self.allow_negative_days:
            warnings.warn(
                "EncounterTransformer(allow_negative_days=True) retains gifts "
                "dated before discharge, which can model solicitation before or "
                "during active treatment. Review "
                "docs/explanation/compliance_considerations.md and your donor-"
                "relations policy before using this in production.",
                UserWarning,
            )

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Append encounter features and strip identifying columns.

        Parameters
        ----------
        X : pd.DataFrame
            Gift-level DataFrame.  Must contain ``merge_key`` and
            ``gift_date_col``.

        Returns
        -------
        X_out : np.ndarray
            Enriched array with two new columns:

            * ``days_since_last_discharge``: ``float64`` days elapsed between
              the donor's latest discharge and the gift date.  ``NaN`` for
              donors absent from the encounter table or (when
              ``allow_negative_days=False``) for gifts dated before discharge.
            * ``encounter_frequency_score``: ``log1p`` of the donor's encounter
              **row** count, not a count and not a distinct count.  ``0.0`` for
              donors with no recorded encounters.

            All identifier-like columns (including ``merge_key``) are removed.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        ValueError
            If ``merge_key`` or ``gift_date_col`` is absent from ``X``.
        """
        check_is_fitted(self)
        
        if hasattr(X, "columns"):
            input_cols = list(X.columns)
        else:
            n_cols = np.shape(X)[1] if len(np.shape(X)) > 1 else 1
            input_cols = [f"x{i}" for i in range(n_cols)]

        X = validate_data(
            self, _avoid_dtype_promotion(X), dtype=None, ensure_all_finite="allow-nan", reset=False
        )
        X_out = pd.DataFrame(X, columns=input_cols)
        
        self._validate_X(X_out)
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
        if cols_to_drop:
            X_out = X_out.drop(columns=cols_to_drop, errors="ignore")

        # --- Also drop the gift_date column (datetime, not modellable directly) ---
        if self.gift_date_col in X_out.columns:
            X_out = X_out.drop(columns=[self.gift_date_col])
            # dropped_cols_ is the operator's audit trail (see
            # docs/explanation/compliance_considerations.md), so it has to name
            # every column that left, not only the PII-heuristic matches.
            cols_to_drop = cols_to_drop + [self.gift_date_col]

        self.dropped_cols_ = cols_to_drop

        # Convert back to numpy array float64 as instructed
        return X_out.to_numpy(dtype=np.float64)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return privacy-filtered donor and generated encounter feature names.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Ignored. Names are derived from the columns recorded by :meth:`fit`.

        Returns
        -------
        feature_names_out : ndarray of str
            Fitted input columns excluding detected PII and ``gift_date_col``,
            followed by ``"days_since_last_discharge"`` and
            ``"encounter_frequency_score"``.

        Raises
        ------
        NotFittedError
            If the transformer has not been fitted.
        """
        check_is_fitted(self)
        features = list(self.feature_names_in_)
        dropped = set(self._identify_pii_columns(self.feature_names_in_))
        if self.gift_date_col in features:
            dropped.add(self.gift_date_col)
            
        out = [f for f in features if f not in dropped]
        out.extend(["days_since_last_discharge", "encounter_frequency_score"])
        return np.array(out, dtype=object)

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        return tags
