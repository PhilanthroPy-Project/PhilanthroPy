"""
tests/test_grateful_patient_featurizer.py
==========================================
Unit tests for philanthropy.preprocessing.GratefulPatientFeaturizer.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from philanthropy.preprocessing import GratefulPatientFeaturizer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def enc_df():
    """Minimal encounter DataFrame with service lines and physicians."""
    return pd.DataFrame({
        "donor_id": [1, 1, 2, 3, 4],
        "discharge_date": [
            "2022-01-15",
            "2023-03-01",
            "2022-07-20",
            "2021-05-10",
            "2020-12-31",
        ],
        "service_line": ["cardiac", "cardiac", "oncology", "general", "neuroscience"],
        "attending_physician_id": ["P1", "P2", "P3", "P1", "P4"],
    })


@pytest.fixture
def X_donors():
    """Donor-level DataFrame to transform."""
    return pd.DataFrame({"donor_id": [1, 2, 3, 4, 99]})  # 99 = unknown donor


@pytest.fixture
def fitted_featurizer(enc_df, X_donors):
    gpf = GratefulPatientFeaturizer(encounter_df=enc_df)
    gpf.fit(X_donors)
    return gpf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGratefulPatientFeaturizer:

    def test_fit_returns_self(self, enc_df, X_donors):
        gpf = GratefulPatientFeaturizer(encounter_df=enc_df)
        result = gpf.fit(X_donors)
        assert result is gpf

    def test_transform_returns_ndarray(self, fitted_featurizer, X_donors):
        result = fitted_featurizer.transform(X_donors)
        assert isinstance(result, np.ndarray), "transform() must return np.ndarray"

    def test_transform_shape(self, fitted_featurizer, X_donors):
        result = fitted_featurizer.transform(X_donors)
        assert result.shape == (len(X_donors), 4), "Output must be (n_samples, 4)"

    def test_unknown_donor_gets_zero_features(self, fitted_featurizer, X_donors):
        """Donor 99 is absent from encounter table → all zeros."""
        result = fitted_featurizer.transform(X_donors)
        unknown_row = result[4]  # donor 99 at index 4
        np.testing.assert_array_equal(
            unknown_row, [0.0, 0.0, 0.0, 0.0],
            err_msg="Unknown donor must have all-zero features",
        )

    def test_clinical_gravity_score_respects_service_line_weights(
        self, fitted_featurizer, X_donors
    ):
        """clinical_gravity_score must be scaled by service-line capacity weights."""
        result = fitted_featurizer.transform(X_donors)
        # Donor 1: 2 cardiac encounters × 3.2 = 6.4
        assert result[0, 0] == pytest.approx(2 * 3.2, rel=1e-5), (
            f"Expected cardiac gravity 6.4, got {result[0, 0]}"
        )
        # Donor 2: 1 oncology encounter × 2.9 = 2.9
        assert result[1, 0] == pytest.approx(1 * 2.9, rel=1e-5)

    def test_cardiac_patient_has_higher_gravity_than_general_patient(
        self, fitted_featurizer, X_donors
    ):
        result = fitted_featurizer.transform(X_donors)
        cardiac_gravity = result[0, 0]  # donor 1: cardiac
        general_gravity = result[2, 0]  # donor 3: general
        assert cardiac_gravity > general_gravity, (
            "Cardiac service line must have higher gravity score than general"
        )

    def test_missing_drg_weight_col_fills_nan_then_zero_after_transform(
        self, enc_df, X_donors
    ):
        """Without drg_weight_col, total_drg_weight (col 3) should be 0.0 after fillna."""
        gpf = GratefulPatientFeaturizer(
            encounter_df=enc_df,
            drg_weight_col=None,  # no DRG col
        )
        gpf.fit(X_donors)
        result = gpf.transform(X_donors)
        # total_drg_weight is col 3 — should be 0.0 (from fillna after NaN)
        assert (result[:, 3] == 0.0).all(), (
            "Without drg_weight_col, total_drg_weight must be 0.0 after fillna"
        )

    def test_drg_weight_col_sums_correctly(self, X_donors):
        """With drg_weight_col present, total_drg_weight must sum DRG weights per donor."""
        enc = pd.DataFrame({
            "donor_id": [1, 1, 2],
            "discharge_date": ["2022-01-01", "2022-06-01", "2022-03-01"],
            "service_line": ["cardiac", "cardiac", "oncology"],
            "attending_physician_id": ["P1", "P2", "P3"],
            "drg_weight": [1.5, 2.0, 3.0],
        })
        X = pd.DataFrame({"donor_id": [1, 2]})
        gpf = GratefulPatientFeaturizer(encounter_df=enc, drg_weight_col="drg_weight")
        gpf.fit(X)
        result = gpf.transform(X)
        assert result[0, 3] == pytest.approx(3.5), "Donor 1: sum(1.5+2.0) = 3.5"
        assert result[1, 3] == pytest.approx(3.0), "Donor 2: sum(3.0) = 3.0"

    def test_clone_does_not_carry_fitted_state(self, enc_df, X_donors):
        gpf = GratefulPatientFeaturizer(encounter_df=enc_df)
        gpf.fit(X_donors)
        cloned = clone(gpf)
        assert not hasattr(cloned, "encounter_summary_"), (
            "Cloned estimator must not carry fitted state"
        )

    def test_not_fitted_raises(self, enc_df, X_donors):
        gpf = GratefulPatientFeaturizer(encounter_df=enc_df)
        with pytest.raises(NotFittedError):
            gpf.transform(X_donors)

    def test_encounter_path_loads_parquet(self, X_donors):
        """encounter_path triggers pd.read_parquet at fit time."""
        enc = pd.DataFrame({
            "donor_id": [1, 2],
            "discharge_date": ["2022-01-01", "2022-06-01"],
            "service_line": ["cardiac", "oncology"],
            "attending_physician_id": ["P1", "P2"],
        })
        gpf = GratefulPatientFeaturizer(encounter_path="/fake/path.parquet")
        with patch("pandas.read_parquet", return_value=enc) as mock_rp:
            gpf.fit(X_donors)
            mock_rp.assert_called_once_with("/fake/path.parquet")
        result = gpf.transform(X_donors)
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 4

    def test_use_capacity_weights_false(self, enc_df, X_donors):
        """use_capacity_weights=False → clinical_gravity_score == total_encounters."""
        gpf = GratefulPatientFeaturizer(
            encounter_df=enc_df, use_capacity_weights=False
        )
        gpf.fit(X_donors)
        result = gpf.transform(X_donors)
        # Donor 1: 2 encounters, no weights → clinical_gravity_score = 2.0
        assert result[0, 0] == pytest.approx(2.0), (
            "Without capacity weights, gravity_score == encounter count"
        )

    def test_no_encounter_source_raises(self, X_donors):
        gpf = GratefulPatientFeaturizer()  # no encounter_df, no encounter_path
        with pytest.raises(ValueError, match="encounter_df|encounter_path"):
            gpf.fit(X_donors)

    def test_get_params_returns_all_init_params(self, enc_df):
        gpf = GratefulPatientFeaturizer(
            encounter_df=enc_df,
            service_line_col="service_line",
            physician_col="attending_physician_id",
            use_capacity_weights=True,
            merge_key="donor_id",
        )
        params = gpf.get_params()
        assert "encounter_df" in params
        assert "encounter_path" in params
        assert "service_line_col" in params
        assert "physician_col" in params
        assert "use_capacity_weights" in params
        assert "merge_key" in params
        assert "discharge_col" in params
        assert "drg_weight_col" in params

    def test_service_line_normalisation(self, X_donors):
        """Service lines are normalised to lowercase with underscores."""
        enc = pd.DataFrame({
            "donor_id": [1],
            "discharge_date": ["2022-01-01"],
            "service_line": ["Cardiac"],  # mixed case
            "attending_physician_id": ["P1"],
        })
        gpf = GratefulPatientFeaturizer(encounter_df=enc)
        gpf.fit(X_donors)
        # After normalisation, "Cardiac" → "cardiac" → weight 3.2
        result = gpf.transform(pd.DataFrame({"donor_id": [1]}))
        assert result[0, 0] == pytest.approx(1 * 3.2)
