"""
philanthropy.preprocessing._grateful_patient
============================================
Clinical intensity and service-line featurization for grateful-patient programs.

This transformer bridges EHR service-line and treating-physician data with the
advancement CRM to produce clinical-depth features for major gift propensity.
It is the primary featurizer for AMC (Academic Medical Center) grateful patient
programs, where donor engagement is anchored to high-intensity clinical encounters.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

# Service-line capacity weights based on published AMC benchmarks.
# Cardiac, oncology, and neuroscience generate ~70% of grateful patient major
# gifts despite only ~35% of encounter volume.
_SERVICE_LINE_CAPACITY_WEIGHTS: dict[str, float] = {
    "cardiac": 3.2,
    "oncology": 2.9,
    "neuroscience": 2.7,
    "orthopedics": 1.8,
    "womens_health": 1.6,
    "general": 1.0,
}


def _normalise_service_line(val: str) -> str:
    """Lowercase and replace non-alpha characters with underscore."""
    s = str(val).lower()
    return re.sub(r"[^a-z]+", "_", s).strip("_")


class GratefulPatientFeaturizer(TransformerMixin, BaseEstimator):
    """Featurize clinical signals from grateful-patient encounter data.

    This transformer bridges EHR service-line and treating-physician data with
    the advancement CRM to produce clinical-depth features for major gift
    propensity models.

    Parameters
    ----------
    encounter_df : pd.DataFrame | None, default=None
        Reference table of clinical encounters. Must contain ``merge_key``
        and ``discharge_col`` columns. Stored verbatim for ``get_params``
        compatibility; snapshotted via ``.copy()`` at fit time to prevent
        mutation leakage.
    encounter_path : str | None, default=None
        Path to a Parquet or CSV file containing clinical encounters.
        Alternative to ``encounter_df``. If both are provided,
        ``encounter_path`` takes precedence.
    service_line_col : str, default="service_line"
        Column in the encounter table holding service line / department name.
    physician_col : str, default="attending_physician_id"
        Column in the encounter table holding the attending physician ID.
    drg_weight_col : str | None, default=None
        Optional column holding DRG (Diagnosis Related Group) relative weights.
        If present, total DRG weight per donor is computed.
    use_capacity_weights : bool, default=True
        If True, apply AMC-benchmarked service-line capacity weights to scale
        the clinical gravity score.
    merge_key : str, default="donor_id"
        Column name present in both the encounter table and ``X`` used to merge.
    discharge_col : str, default="discharge_date"
        Column in the encounter table holding discharge dates.

    Attributes
    ----------
    encounter_summary_ : pd.DataFrame
        Per-donor aggregated encounter features, indexed by ``merge_key``.
        Set at fit time.
    n_features_in_ : int
        Number of features seen at fit time (set by ``_validate_data``).
    feature_names_in_ : ndarray of str
        Column names of ``X`` at fit time (set by ``_validate_data`` when X
        is a DataFrame).

    Raises
    ------
    ValueError
        If neither ``encounter_df`` nor ``encounter_path`` is provided.

    Notes
    -----
    The four output columns are:

    ========================= ================================================
    Column                    Description
    ========================= ================================================
    ``clinical_gravity_score`` Encounter count × service-line capacity weight.
    ``distinct_service_lines`` Number of unique service lines.
    ``distinct_physicians``    Number of unique attending physicians.
    ``total_drg_weight``       Sum of DRG relative weights (NaN if unavailable).
    ========================= ================================================

    Donors absent from the encounter table receive zeros for all columns.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from philanthropy.preprocessing import GratefulPatientFeaturizer
    >>> enc = pd.DataFrame({
    ...     "donor_id": [1, 1, 2],
    ...     "discharge_date": ["2022-01-01", "2023-06-15", "2022-09-30"],
    ...     "service_line": ["cardiac", "cardiac", "oncology"],
    ...     "attending_physician_id": ["P1", "P2", "P3"],
    ... })
    >>> X = pd.DataFrame({"donor_id": [1, 2, 3]})
    >>> gpf = GratefulPatientFeaturizer(encounter_df=enc)
    >>> gpf.fit(X)
    GratefulPatientFeaturizer(...)
    >>> out = gpf.transform(X)
    >>> out.shape
    (3, 4)
    """

    def __init__(
        self,
        encounter_df: pd.DataFrame | None = None,
        encounter_path: str | None = None,
        service_line_col: str = "service_line",
        physician_col: str = "attending_physician_id",
        drg_weight_col: str | None = None,
        use_capacity_weights: bool = True,
        merge_key: str = "donor_id",
        discharge_col: str = "discharge_date",
    ) -> None:
        self.encounter_df = encounter_df
        self.encounter_path = encounter_path
        self.service_line_col = service_line_col
        self.physician_col = physician_col
        self.drg_weight_col = drg_weight_col
        self.use_capacity_weights = use_capacity_weights
        self.merge_key = merge_key
        self.discharge_col = discharge_col

    def fit(self, X, y=None) -> "GratefulPatientFeaturizer":
        """Build per-donor encounter summaries from encounter data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Donor-level feature matrix. Used only for schema registration via
            ``_validate_data``; no target leakage occurs here.
        y : ignored

        Returns
        -------
        self : GratefulPatientFeaturizer

        Raises
        ------
        ValueError
            If neither ``encounter_df`` nor ``encounter_path`` is set.
        """
        # Step 1: Load encounter data — snapshot, never store raw_enc
        if self.encounter_path is not None:
            raw_enc = pd.read_parquet(self.encounter_path)
        elif self.encounter_df is not None:
            raw_enc = self.encounter_df.copy()  # critical: snapshot here
        else:
            raise ValueError(
                "GratefulPatientFeaturizer requires either encounter_df or "
                "encounter_path to be set."
            )

        # Step 2: Coerce discharge dates
        raw_enc = raw_enc.copy()
        raw_enc[self.discharge_col] = pd.to_datetime(
            raw_enc[self.discharge_col], errors="coerce"
        )

        # Step 3: Normalise service_line values
        if self.service_line_col in raw_enc.columns:
            raw_enc[self.service_line_col] = (
                raw_enc[self.service_line_col]
                .astype(str)
                .apply(_normalise_service_line)
            )

        # Step 4: Groupby merge_key
        grouped = raw_enc.groupby(self.merge_key)

        summary_parts: dict[str, pd.Series] = {}

        if self.service_line_col in raw_enc.columns:
            # Mode (most frequent) service line per donor
            summary_parts["primary_service_line"] = grouped[
                self.service_line_col
            ].agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "general")
            summary_parts["distinct_service_lines"] = grouped[
                self.service_line_col
            ].nunique()
        else:
            summary_parts["primary_service_line"] = pd.Series(
                "general", index=grouped.groups.keys()
            )
            summary_parts["distinct_service_lines"] = pd.Series(
                0, dtype=int, index=grouped.groups.keys()
            )

        if self.physician_col in raw_enc.columns:
            summary_parts["distinct_physicians"] = grouped[
                self.physician_col
            ].nunique()
        else:
            summary_parts["distinct_physicians"] = pd.Series(
                0, dtype=int, index=grouped.groups.keys()
            )

        summary_parts["total_encounters"] = grouped[self.discharge_col].count()
        summary_parts["last_discharge"] = grouped[self.discharge_col].max()

        # Step 5: DRG weight column
        if (
            self.drg_weight_col is not None
            and self.drg_weight_col in raw_enc.columns
        ):
            summary_parts["total_drg_weight"] = grouped[
                self.drg_weight_col
            ].sum()
        else:
            # Sentinel: will become NaN for all donors
            keys = list(grouped.groups.keys())
            summary_parts["total_drg_weight"] = pd.Series(
                np.nan, index=keys, dtype=float
            )

        encounter_summary = pd.DataFrame(summary_parts)

        # Step 6: Clinical gravity score
        if self.use_capacity_weights:
            encounter_summary["clinical_gravity_score"] = (
                encounter_summary["total_encounters"].astype(float)
                * encounter_summary["primary_service_line"].map(
                    lambda s: _SERVICE_LINE_CAPACITY_WEIGHTS.get(s, 1.0)
                )
            )
        else:
            encounter_summary["clinical_gravity_score"] = (
                encounter_summary["total_encounters"].astype(float)
            )

        # Step 7: Store fitted attribute
        self.encounter_summary_ = encounter_summary

        # Step 8: Register feature schema
        validate_data(self, X, dtype=None, ensure_all_finite="allow-nan", reset=True)

        return self

    def transform(self, X, y=None) -> np.ndarray:
        """Merge clinical features into the donor feature matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Donor-level feature matrix. Must contain ``merge_key`` column
            (or that column as the first column for ndarray input).

        Returns
        -------
        X_out : np.ndarray of shape (n_samples, 4), dtype float64
            Columns in order:
            ``clinical_gravity_score``, ``distinct_service_lines``,
            ``distinct_physicians``, ``total_drg_weight``.
            Donors absent from encounter table get 0.0 for all columns.
        """
        validate_data(self, X, dtype=None, ensure_all_finite="allow-nan", reset=False)
        check_is_fitted(self)

        _FEATURE_COLS = [
            "clinical_gravity_score",
            "distinct_service_lines",
            "distinct_physicians",
            "total_drg_weight",
        ]

        # Build a DataFrame to merge on merge_key
        if isinstance(X, pd.DataFrame) and self.merge_key in X.columns:
            X_df = X[[self.merge_key]].copy()
        elif isinstance(X, pd.DataFrame) and hasattr(self, "feature_names_in_"):
            # merge_key must be in the feature names
            if self.merge_key in self.feature_names_in_:
                X_df = X[[self.merge_key]].copy()
            else:
                # No merge key available — return zeros
                n = len(X)
                return np.zeros((n, 4), dtype=np.float64)
        elif hasattr(self, "feature_names_in_") and self.merge_key in list(
            self.feature_names_in_
        ):
            arr = np.asarray(X)
            col_idx = list(self.feature_names_in_).index(self.merge_key)
            X_df = pd.DataFrame(
                {self.merge_key: arr[:, col_idx]}
            )
        else:
            # No merge key — cannot join, return zeros
            n = np.asarray(X).shape[0]
            return np.zeros((n, 4), dtype=np.float64)

        # Left-merge with encounter_summary_
        merged = X_df.merge(
            self.encounter_summary_[_FEATURE_COLS],
            left_on=self.merge_key,
            right_index=True,
            how="left",
        )

        # Step 5: fillna(0.0) — unknown donors get zeros
        result = merged[_FEATURE_COLS].fillna(0.0)

        return result.to_numpy(dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.array(
            [
                "clinical_gravity_score",
                "distinct_service_lines",
                "distinct_physicians",
                "total_drg_weight",
            ],
            dtype=object,
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
