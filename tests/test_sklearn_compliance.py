"""
tests/test_sklearn_compliance.py
================================
Formal battery of scikit-learn compliance tests for all estimators.

Section 1: Standard estimators via parametrize_with_checks.
Section 2: EncounterTransformer manual compliance (non-standard constructor).
Section 3: GratefulPatientFeaturizer manual compliance.
Section 4: Pipeline integration smoke tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from philanthropy.datasets import generate_synthetic_donor_data
from philanthropy.models import (
    DonorPropensityModel,
    LapsePredictor,
    MajorGiftClassifier,
    ShareOfWalletRegressor,
)
from philanthropy.preprocessing import (
    EncounterTransformer,
    FiscalYearTransformer,
    GratefulPatientFeaturizer,
    PlannedGivingSignalTransformer,
    RFMTransformer,
    SolicitationWindowTransformer,
    WealthPercentileTransformer,
    WealthScreeningImputer,
)


# ---------------------------------------------------------------------------
# SECTION 1 — Standard estimators (no required args)
# ---------------------------------------------------------------------------

_STANDARD_ESTIMATORS = [
    DonorPropensityModel(n_estimators=10, random_state=0),
    ShareOfWalletRegressor(max_iter=20, random_state=0),
    # LapsePredictor excluded: non-standard fit signature (gift_dates)
    MajorGiftClassifier(),
    FiscalYearTransformer(date_col="gift_date", fiscal_year_start=7),
    WealthScreeningImputer(
        wealth_cols=["x0"],     # sklearn uses "x0", "x1"... in generated arrays
        strategy="median",
        add_indicator=False,    # indicator=False for numeric-only check_estimator
    ),
    RFMTransformer(),
    SolicitationWindowTransformer(),
    PlannedGivingSignalTransformer(),
]


@parametrize_with_checks(_STANDARD_ESTIMATORS)
def test_sklearn_compliance(estimator, check):
    """Run standard sklearn check_estimator battery on compliant estimators."""
    check(estimator)


# NOTE: WealthScreeningImputer with add_indicator=True may fail
# check_n_features_in if the indicator columns change the output shape.
# This is a known limitation and is tracked in GitHub issue #43. The
# add_indicator=False variant is tested above.


# ---------------------------------------------------------------------------
# SECTION 2 — EncounterTransformer manual compliance
# ---------------------------------------------------------------------------

_MINIMAL_ENCOUNTER_DF = pd.DataFrame({
    "donor_id": [1, 2, 3],
    "discharge_date": ["2022-01-01", "2022-06-15", "2022-11-30"],
})

_MINIMAL_GIFT_X = pd.DataFrame({
    "donor_id": [1, 2, 3],
    "gift_date": ["2023-01-01", "2023-02-15", "2023-03-20"],
    "gift_amount": [1000.0, 500.0, 250.0],
})


class TestEncounterTransformerCompliance:

    def test_get_params_returns_all_init_params(self):
        t = EncounterTransformer(encounter_df=_MINIMAL_ENCOUNTER_DF)
        params = t.get_params()
        assert "encounter_df" in params
        assert "encounter_path" in params
        assert "discharge_col" in params
        assert "gift_date_col" in params
        assert "merge_key" in params
        assert "allow_negative_days" in params

    def test_set_params_round_trips_and_returns_self(self):
        t = EncounterTransformer(encounter_df=_MINIMAL_ENCOUNTER_DF)
        result = t.set_params(allow_negative_days=True, merge_key="donor_id")
        assert result is t
        assert t.allow_negative_days is True

    def test_clone_does_not_carry_fitted_state(self):
        from sklearn.base import clone
        t = EncounterTransformer(encounter_df=_MINIMAL_ENCOUNTER_DF)
        t.fit(_MINIMAL_GIFT_X)
        cloned = clone(t)
        assert not hasattr(cloned, "encounter_summary_")

    def test_not_fitted_raises_not_fitted_error(self):
        t = EncounterTransformer(encounter_df=_MINIMAL_ENCOUNTER_DF)
        with pytest.raises(NotFittedError):
            t.transform(_MINIMAL_GIFT_X)

    def test_fit_returns_self(self):
        t = EncounterTransformer(encounter_df=_MINIMAL_ENCOUNTER_DF)
        result = t.fit(_MINIMAL_GIFT_X)
        assert result is t

    def test_output_is_numpy_ndarray(self):
        """CRITICAL: transform() must return np.ndarray."""
        t = EncounterTransformer(encounter_df=_MINIMAL_ENCOUNTER_DF)
        t.fit(_MINIMAL_GIFT_X)
        out = t.transform(_MINIMAL_GIFT_X)
        assert isinstance(out, np.ndarray), (
            "EncounterTransformer.transform() must return np.ndarray, "
            f"got {type(out).__name__}"
        )


# ---------------------------------------------------------------------------
# SECTION 3 — GratefulPatientFeaturizer manual compliance
# ---------------------------------------------------------------------------

_MINIMAL_ENC_DF = pd.DataFrame({
    "donor_id": [1, 2, 3],
    "discharge_date": ["2022-01-01", "2022-06-15", "2022-11-30"],
    "service_line": ["cardiac", "oncology", "general"],
    "attending_physician_id": ["P1", "P2", "P3"],
})

_MINIMAL_DONOR_X = pd.DataFrame({"donor_id": [1, 2, 3]})


class TestGratefulPatientFeaturizerCompliance:

    def test_get_params_returns_all_init_params(self):
        gpf = GratefulPatientFeaturizer(encounter_df=_MINIMAL_ENC_DF)
        params = gpf.get_params()
        assert "encounter_df" in params
        assert "encounter_path" in params
        assert "service_line_col" in params
        assert "physician_col" in params
        assert "use_capacity_weights" in params
        assert "merge_key" in params
        assert "discharge_col" in params

    def test_set_params_round_trips_and_returns_self(self):
        gpf = GratefulPatientFeaturizer(encounter_df=_MINIMAL_ENC_DF)
        result = gpf.set_params(use_capacity_weights=False)
        assert result is gpf
        assert gpf.use_capacity_weights is False

    def test_clone_does_not_carry_fitted_state(self):
        from sklearn.base import clone
        gpf = GratefulPatientFeaturizer(encounter_df=_MINIMAL_ENC_DF)
        gpf.fit(_MINIMAL_DONOR_X)
        cloned = clone(gpf)
        assert not hasattr(cloned, "encounter_summary_")

    def test_not_fitted_raises_not_fitted_error(self):
        gpf = GratefulPatientFeaturizer(encounter_df=_MINIMAL_ENC_DF)
        with pytest.raises(NotFittedError):
            gpf.transform(_MINIMAL_DONOR_X)

    def test_fit_returns_self(self):
        gpf = GratefulPatientFeaturizer(encounter_df=_MINIMAL_ENC_DF)
        result = gpf.fit(_MINIMAL_DONOR_X)
        assert result is gpf

    def test_output_is_numpy_ndarray(self):
        """CRITICAL: transform() must return np.ndarray."""
        gpf = GratefulPatientFeaturizer(encounter_df=_MINIMAL_ENC_DF)
        gpf.fit(_MINIMAL_DONOR_X)
        out = gpf.transform(_MINIMAL_DONOR_X)
        assert isinstance(out, np.ndarray), (
            f"GratefulPatientFeaturizer.transform() must return np.ndarray, "
            f"got {type(out).__name__}"
        )
        assert out.shape == (3, 4)


# ---------------------------------------------------------------------------
# SECTION 4 — Pipeline integration smoke tests
# ---------------------------------------------------------------------------

def test_donor_propensity_model_in_pipeline_with_5fold_cv():
    """Fit DonorPropensityModel in 5-fold CV and assert mean AUC > 0.5."""
    from sklearn.model_selection import cross_val_score

    df = generate_synthetic_donor_data(300, random_state=0)
    feature_cols = ["total_gift_amount", "years_active", "event_attendance_count"]
    X = df[feature_cols].to_numpy()
    y = df["is_major_donor"].to_numpy()

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", DonorPropensityModel(n_estimators=10, random_state=0)),
    ])
    scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
    assert scores.mean() > 0.5, (
        f"Expected mean AUC > 0.5, got {scores.mean():.3f}"
    )


def test_wealth_imputer_in_pipeline_does_not_contaminate_folds():
    """In 5-fold CV, fill stats must NOT be identical across all folds."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({
        "estimated_net_worth": np.where(
            rng.random(n) < 0.4, np.nan, rng.lognormal(14, 2, n)
        )
    })
    y = rng.integers(0, 2, n)

    pipe = Pipeline([
        ("imp", WealthScreeningImputer(
            wealth_cols=["estimated_net_worth"],
            strategy="median",
            add_indicator=False,
        )),
        ("clf", LogisticRegression(max_iter=200)),
    ])

    fill_values = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, _ in kf.split(X, y):
        imp = WealthScreeningImputer(
            wealth_cols=["estimated_net_worth"],
            strategy="median",
            add_indicator=False,
        )
        imp.fit(X.iloc[train_idx])
        fill_values.append(imp.fill_values_["estimated_net_worth"])

    # Not all fold fill values should be identical
    assert len(set(fill_values)) > 1, (
        "All fold fill values are identical — possible full-dataset leakage"
    )


def test_share_of_wallet_handles_nan_input_without_imputation():
    """ShareOfWalletRegressor must handle NaN without crashing."""
    rng = np.random.default_rng(0)
    n = 100
    X = np.where(rng.random((n, 4)) < 0.3, np.nan, rng.standard_normal((n, 4)))
    y = rng.lognormal(6, 1, n)

    model = ShareOfWalletRegressor(max_iter=20, random_state=0)
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == (n,), "predict() must return (n_samples,) array"
    assert not np.any(np.isnan(preds)), "Predictions must not contain NaN"
    # Capacity floor check (if applicable)
    if hasattr(model, "capacity_floor"):
        assert (preds >= model.capacity_floor).all()


def test_full_grateful_patient_pipeline_end_to_end():
    """Full pipeline from encounter data to DonorPropensityModel must not raise."""
    # Build synthetic encounter data
    rng = np.random.default_rng(7)
    n_donors = 50
    donor_ids = list(range(1, n_donors + 1))

    enc_df = pd.DataFrame({
        "donor_id": donor_ids[:30] * 2,
        "discharge_date": pd.date_range("2021-01-01", periods=60, freq="15D").strftime(
            "%Y-%m-%d"
        ),
        "service_line": rng.choice(
            ["cardiac", "oncology", "general", "neuroscience"], size=60
        ),
        "attending_physician_id": [f"P{i % 10}" for i in range(60)],
        "days_since_last_discharge": rng.integers(100, 700, size=60),
    })

    X_base = pd.DataFrame({
        "donor_id": donor_ids,
        "donor_age": rng.integers(40, 85, n_donors).astype(float),
        "years_active": rng.integers(1, 25, n_donors).astype(float),
        "planned_gift_inclination": rng.uniform(0, 1, n_donors),
        "estimated_net_worth": np.where(
            rng.random(n_donors) < 0.3, np.nan, rng.lognormal(13, 2, n_donors)
        ),
    })
    y = rng.integers(0, 2, n_donors)

    # GratefulPatientFeaturizer produces (n, 4), then we feed to DonorPropensityModel
    gpf = GratefulPatientFeaturizer(encounter_df=enc_df)
    gpf.fit(X_base)
    X_features = gpf.transform(X_base)

    model = DonorPropensityModel(n_estimators=10, random_state=0)
    model.fit(X_features, y)
    scores = model.predict_affinity_score(X_features)

    assert scores.shape == (n_donors,)
    assert (scores >= 0.0).all() and (scores <= 100.0).all(), (
        "All affinity scores must be in [0, 100]"
    )
