"""
philanthropy.preprocessing._encounter_recency
=============================================
HIPAA-safe encounter-date recency features for Grateful Patient Programs.

This transformer operates exclusively on **date columns** — no PHI fields
(names, MRNs, diagnosis codes) should be present.  The resulting features
are strong temporal signals in major-gift propensity models trained at
academic medical centres (AMCs):

* ``days_since_last_encounter`` — Continuous, naturally right-skewed.
* ``encounter_in_last_90d``     — Boolean trigger for hot-zone solicitations.
* ``fiscal_year_of_encounter``  — Integer FY for cohort-level reporting.

All three features handle ``NaT`` / ``NaN`` gracefully (``NaN`` fill for
continuous, ``False`` for the boolean, ``pd.NA`` for the FY integer).

Typical usage
-------------
>>> import pandas as pd
>>> from philanthropy.preprocessing import EncounterRecencyTransformer
>>> X = pd.DataFrame({
...     "last_encounter_date": ["2023-06-01", "2022-12-15", None],
... })
>>> t = EncounterRecencyTransformer(fiscal_year_start=7, reference_date="2023-09-01")
>>> out = t.fit_transform(X)
>>> out.shape
(3, 3)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class EncounterRecencyTransformer(TransformerMixin, BaseEstimator):
    """Transform HIPAA-safe encounter-date columns into predictive recency features.

    Given one or more date-only columns (no PHI — dates only), this
    transformer produces three downstream-model-ready features per date
    column:

    ``days_since_last_encounter``
        Integer days between ``reference_date`` and the encounter date.
        ``NaN`` for missing/unparseable dates.  Always non-negative when
        ``reference_date >= encounter_date``; negative values indicate
        future dates (rare in production) and are left as-is to allow models
        to detect data-quality anomalies.

    ``encounter_in_last_90d``
        Float64 0.0 / 1.0 flag — 1.0 if ``days_since_last_encounter <= 90``.
        Missing dates → 0.0.

    ``fiscal_year_of_encounter``
        Integer fiscal year in which the encounter ends (e.g., a July-start
        FY convention assigns a June-30 encounter to the current year, while
        a July-1 encounter starts the *next* FY).  Missing → ``np.nan``
        (returned as ``float64``).

    Parameters
    ----------
    date_col : str or list of str, default="last_encounter_date"
        Column name(s) in ``X`` containing ISO-8601 encounter/discharge
        dates.  If a list is provided, one set of three output features is
        produced per column (columns are prefixed by ``<col>__``).
    fiscal_year_start : int, default=7
        Month (1–12) on which the organisation's fiscal year begins.
        ``7`` = July fiscal-year start (common in US academic medical
        centres and universities).
    reference_date : str, datetime-like, or None, default=None
        The anchor date used to compute ``days_since_last_encounter``.
        If ``None``, it is determined at :meth:`fit` time as the **maximum
        observed date** in the training data (i.e., the most recent
        clinical encounter in the training fold).  Setting an explicit
        reference date is recommended for production scoring runs to ensure
        consistency between training and inference time.
    timezone : str or None, default=None
        Optional timezone name (e.g., ``"America/Chicago"``).  When
        provided, timezone-naive datetimes in ``X`` are localised to this
        timezone before difference computation, preventing offset errors for
        hospitals that cross daylight-saving boundaries.  If ``None``,
        all dates are kept timezone-naive (recommended for HIPAA-safe
        de-identified datasets where the exact timezone is unknown).

    Attributes
    ----------
    reference_date_ : pd.Timestamp
        The reference date frozen at :meth:`fit` time.
    n_features_in_ : int
        Number of columns in ``X`` at :meth:`fit` time (set by
        :func:`~sklearn.utils.validation.validate_data`).
    feature_names_in_ : ndarray of str
        Column names of ``X`` at :meth:`fit` time.

    Raises
    ------
    ValueError
        If ``fiscal_year_start`` is not an integer in [1, 12].
    TypeError
        If the resolved ``date_col`` columns cannot be coerced to
        ``datetime64``.

    Examples
    --------
    >>> import pandas as pd
    >>> from philanthropy.preprocessing import EncounterRecencyTransformer
    >>> X = pd.DataFrame({
    ...     "last_encounter_date": ["2023-06-01", "2022-12-15", None],
    ... })
    >>> t = EncounterRecencyTransformer(fiscal_year_start=7, reference_date="2023-09-01")
    >>> t.set_output(transform="pandas")  # doctest: +ELLIPSIS
    EncounterRecencyTransformer(...)
    >>> out = t.fit_transform(X)
    >>> out.shape
    (3, 3)
    >>> int(out.iloc[0, 0])  # days since 2023-06-01 from 2023-09-01 = 92
    92
    >>> bool((out.iloc[:, 1] >= 0).all())
    True

    Notes
    -----
    **HIPAA note:** This transformer accepts only date columns.  Ensure that
    no PHI fields (MRN, patient name, diagnosis code) are included in ``X``.

    **Fiscal year convention:** With ``fiscal_year_start=7``, the fiscal year
    is identified by the calendar year in which it *ends*.  A date of
    2023-07-01 belongs to FY **2024**; a date of 2023-06-30 belongs to FY
    **2023**.  This matches the convention used by most US research universities
    and many hospital foundations.
    """

    def __init__(
        self,
        date_col: str | list[str] = "last_encounter_date",
        fiscal_year_start: int = 7,
        reference_date=None,
        timezone: Optional[str] = None,
    ) -> None:
        # scikit-learn rule: __init__ ONLY assigns; no validation, no side-effects.
        self.date_col = date_col
        self.fiscal_year_start = fiscal_year_start
        self.reference_date = reference_date
        self.timezone = timezone

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_fiscal_year_start(self) -> None:
        """Raise ValueError if fiscal_year_start is out of range."""
        if not isinstance(self.fiscal_year_start, (int, np.integer)) or not (
            1 <= int(self.fiscal_year_start) <= 12
        ):
            raise ValueError(
                f"`fiscal_year_start` must be an integer in [1, 12], "
                f"got {self.fiscal_year_start!r}."
            )

    def _resolve_date_cols(self) -> list[str]:
        """Return the date column(s) as a list of strings."""
        if isinstance(self.date_col, str):
            return [self.date_col]
        return list(self.date_col)

    def _parse_dates(self, series: pd.Series) -> pd.Series:
        """Parse a date series to datetime64[ns], optionally localising timezone."""
        parsed = pd.to_datetime(series, errors="coerce", utc=(self.timezone is not None))
        if self.timezone is not None:
            # Convert to the target timezone; if already tz-aware, convert.
            try:
                parsed = parsed.dt.tz_convert(self.timezone)
            except Exception:
                parsed = parsed.dt.tz_localize(self.timezone)
        return parsed

    def _fiscal_year(self, dt: pd.Timestamp) -> int:
        """Return the fiscal year for a single Timestamp."""
        fys = int(self.fiscal_year_start)
        if dt.month >= fys:
            # Encounter is in the opening half of fiscal year → FY ends next calendar year
            return dt.year + 1
        return dt.year

    def _compute_recency_features(
        self, dates: pd.Series, prefix: str
    ) -> pd.DataFrame:
        """Compute (days_since, in_last_90d, fiscal_year) for a date series."""
        ref = self.reference_date_

        # days_since_last_encounter
        # Timezone strip for subtraction when tz-naive reference vs tz-aware series
        if dates.dt.tz is not None and ref.tzinfo is None:
            ref_ts = pd.Timestamp(ref).tz_localize(self.timezone or "UTC")
        elif dates.dt.tz is None and ref.tzinfo is not None:
            dates = dates.dt.tz_localize("UTC")
            ref_ts = ref
        else:
            ref_ts = ref

        delta_days = (ref_ts - dates).dt.days.astype("float64")

        # encounter_in_last_90d: 1.0 if <=90 days ago and non-NaN
        in_90d = np.where(dates.isna(), 0.0, (delta_days <= 90.0).astype(np.float64))

        # fiscal_year_of_encounter: float64 (NaN for missing)
        fy = dates.apply(
            lambda d: np.nan if pd.isna(d) else float(self._fiscal_year(d))
        ).astype("float64")

        cols = {}
        p = f"{prefix}__" if prefix else ""
        cols[f"{p}days_since_last_encounter"] = delta_days.values
        cols[f"{p}encounter_in_last_90d"] = in_90d
        cols[f"{p}fiscal_year_of_encounter"] = fy.values

        return pd.DataFrame(cols)

    # ------------------------------------------------------------------
    # fit / transform
    # ------------------------------------------------------------------

    def fit(self, X, y=None) -> "EncounterRecencyTransformer":
        """Validate parameters and freeze the reference date from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.  Must contain the column(s) specified in ``date_col``
            when passed as a pd.DataFrame.
        y : ignored

        Returns
        -------
        self : EncounterRecencyTransformer
        """
        self._validate_fiscal_year_start()

        # Register the input schema via validate_data; allow NaN and string types.
        validate_data(self, X, dtype=None, ensure_all_finite="allow-nan", reset=True)

        # Freeze the reference date:
        if self.reference_date is not None:
            self.reference_date_ = pd.Timestamp(self.reference_date)
        else:
            # Infer from training data: max observed date across all date columns.
            if isinstance(X, pd.DataFrame):
                cols = self._resolve_date_cols()
                max_dates = []
                for col in cols:
                    if col in X.columns:
                        parsed = self._parse_dates(X[col])
                        mx = parsed.max()
                        if not pd.isna(mx):
                            max_dates.append(mx)
                if max_dates:
                    self.reference_date_ = max(max_dates)
                else:
                    warnings.warn(
                        "EncounterRecencyTransformer: no parseable dates found in "
                        "training data; defaulting reference_date_ to today.",
                        UserWarning,
                    )
                    self.reference_date_ = pd.Timestamp.today().normalize()
            else:
                # Cannot infer from ndarray without column names; default to today.
                warnings.warn(
                    "EncounterRecencyTransformer: X is not a DataFrame — "
                    "defaulting reference_date_ to today. Provide reference_date "
                    "explicitly for reproducibility.",
                    UserWarning,
                )
                self.reference_date_ = pd.Timestamp.today().normalize()

        return self

    def transform(self, X, y=None) -> np.ndarray:
        """Compute encounter recency features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data containing the date column(s).

        Returns
        -------
        X_out : np.ndarray of shape (n_samples, 3 * n_date_cols), dtype float64
            Feature columns, in order, for each ``date_col``:

            * ``[<col>__]days_since_last_encounter``
            * ``[<col>__]encounter_in_last_90d``
            * ``[<col>__]fiscal_year_of_encounter``

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        """
        check_is_fitted(self, ["reference_date_"])
        validate_data(self, X, dtype=None, ensure_all_finite="allow-nan", reset=False)

        # Build a working DataFrame from X
        if isinstance(X, pd.DataFrame):
            df = X
        elif hasattr(self, "feature_names_in_"):
            df = pd.DataFrame(
                np.asarray(X, dtype=object), columns=self.feature_names_in_
            )
        else:
            # Fallback: cannot resolve column names — produce NaN output.
            n = np.asarray(X).shape[0]
            cols = self._resolve_date_cols()
            n_out = len(cols) * 3
            return np.full((n, n_out), np.nan, dtype=np.float64)

        cols = self._resolve_date_cols()
        parts: list[pd.DataFrame] = []
        prefix_needed = len(cols) > 1

        for col in cols:
            prefix = col if prefix_needed else ""
            if col in df.columns:
                parsed = self._parse_dates(df[col])
            else:
                warnings.warn(
                    f"EncounterRecencyTransformer: date column {col!r} not found "
                    f"in X; filling recency features with NaN.",
                    UserWarning,
                )
                n = len(df)
                p = f"{prefix}__" if prefix else ""
                parsed = pd.Series(pd.NaT, index=df.index)
                parsed = self._parse_dates(pd.Series([None] * n))

            parts.append(self._compute_recency_features(parsed, prefix=prefix))

        out_df = pd.concat(parts, axis=1) if parts else pd.DataFrame()
        return out_df.to_numpy(dtype=np.float64)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return output feature names.

        Returns
        -------
        feature_names : ndarray of str
        """
        check_is_fitted(self, ["reference_date_"])
        cols = self._resolve_date_cols()
        prefix_needed = len(cols) > 1
        names: list[str] = []
        for col in cols:
            p = f"{col}__" if prefix_needed else ""
            names += [
                f"{p}days_since_last_encounter",
                f"{p}encounter_in_last_90d",
                f"{p}fiscal_year_of_encounter",
            ]
        return np.array(names, dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True  # Date columns are string-like on entry
        return tags
