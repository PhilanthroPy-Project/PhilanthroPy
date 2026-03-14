"""
tests/test_donor_propensity_model.py
=====================================
Comprehensive unit tests for philanthropy.models.DonorPropensityModel.

Tests cover:
- Fit/predict interface compliance
- Affinity score shape, range, and monotonicity
- Fitted attributes
- Pipeline compatibility
- check_estimator compatibility
- NotFittedError behaviour
- get_params / set_params round-trip
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from philanthropy.datasets import generate_synthetic_donor_data
from philanthropy.models import DonorPropensityModel


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def donor_Xy():
    """Return (X, y) numpy arrays from synthetic donor data."""
    df = generate_synthetic_donor_data(n_samples=300, random_state=42)
    feature_cols = ["total_gift_amount", "years_active", "event_attendance_count"]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["is_major_donor"].to_numpy()
    return X, y


@pytest.fixture(scope="module")
def fitted_model(donor_Xy):
    X, y = donor_Xy
    model = DonorPropensityModel(n_estimators=10, random_state=0)
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# __init__ Golden Rule: no logic, only attribute assignment
# ---------------------------------------------------------------------------

def test_init_stores_params_verbatim():
    """All __init__ kwargs must be stored as same-named attributes."""
    model = DonorPropensityModel(
        n_estimators=50,
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight=None,
        random_state=7,
    )
    assert model.n_estimators == 50
    assert model.max_depth == 3
    assert model.min_samples_split == 4
    assert model.min_samples_leaf == 2
    assert model.class_weight is None
    assert model.random_state == 7


def test_no_fitted_attributes_before_fit():
    """Before fit(), no trailing-underscore attributes should exist."""
    model = DonorPropensityModel()
    for attr in ("estimator_", "classes_", "n_features_in_"):
        assert not hasattr(model, attr), f"Unexpected attribute {attr} before fit"


# ---------------------------------------------------------------------------
# Fitted attributes
# ---------------------------------------------------------------------------

def test_fitted_attributes_present(fitted_model):
    assert hasattr(fitted_model, "estimator_")
    assert hasattr(fitted_model, "classes_")
    assert hasattr(fitted_model, "n_features_in_")


def test_classes_contains_zero_and_one(fitted_model):
    assert set(fitted_model.classes_) == {0, 1}


def test_n_features_in_equals_three(fitted_model):
    assert fitted_model.n_features_in_ == 3


# ---------------------------------------------------------------------------
# fit returns self (method chaining)
# ---------------------------------------------------------------------------

def test_fit_returns_self(donor_Xy):
    X, y = donor_Xy
    model = DonorPropensityModel(n_estimators=5, random_state=1)
    result = model.fit(X, y)
    assert result is model


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def test_predict_shape(fitted_model, donor_Xy):
    X, _ = donor_Xy
    preds = fitted_model.predict(X)
    assert preds.shape == (len(X),)


def test_predict_values_binary(fitted_model, donor_Xy):
    X, _ = donor_Xy
    preds = fitted_model.predict(X)
    assert set(preds).issubset({0, 1})


def test_predict_returns_numpy_array(fitted_model, donor_Xy):
    X, _ = donor_Xy
    preds = fitted_model.predict(X)
    assert isinstance(preds, np.ndarray)


# ---------------------------------------------------------------------------
# predict_proba
# ---------------------------------------------------------------------------

def test_predict_proba_shape(fitted_model, donor_Xy):
    X, _ = donor_Xy
    proba = fitted_model.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_predict_proba_rows_sum_to_one(fitted_model, donor_Xy):
    X, _ = donor_Xy
    proba = fitted_model.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_predict_proba_non_negative(fitted_model, donor_Xy):
    X, _ = donor_Xy
    proba = fitted_model.predict_proba(X)
    assert (proba >= 0).all()


def test_predict_proba_returns_numpy_array(fitted_model, donor_Xy):
    X, _ = donor_Xy
    proba = fitted_model.predict_proba(X)
    assert isinstance(proba, np.ndarray)


# ---------------------------------------------------------------------------
# predict_affinity_score
# ---------------------------------------------------------------------------

def test_affinity_score_shape(fitted_model, donor_Xy):
    X, _ = donor_Xy
    scores = fitted_model.predict_affinity_score(X)
    assert scores.shape == (len(X),)


def test_affinity_score_range(fitted_model, donor_Xy):
    X, _ = donor_Xy
    scores = fitted_model.predict_affinity_score(X)
    assert (scores >= 0).all()
    assert (scores <= 100).all()


def test_affinity_score_monotone_with_proba(fitted_model, donor_Xy):
    """Affinity scores must be monotonically equivalent to predict_proba[:,1]."""
    X, _ = donor_Xy
    scores = fitted_model.predict_affinity_score(X)
    proba = fitted_model.predict_proba(X)[:, 1]
    rank_scores = scores.argsort()
    rank_proba = proba.argsort()
    assert np.array_equal(rank_scores, rank_proba)


def test_affinity_score_is_float_array(fitted_model, donor_Xy):
    X, _ = donor_Xy
    scores = fitted_model.predict_affinity_score(X)
    assert isinstance(scores, np.ndarray)
    assert np.issubdtype(scores.dtype, np.floating)


# ---------------------------------------------------------------------------
# NotFittedError
# ---------------------------------------------------------------------------

def test_predict_before_fit_raises(donor_Xy):
    X, _ = donor_Xy
    model = DonorPropensityModel()
    with pytest.raises(NotFittedError):
        model.predict(X)


def test_predict_proba_before_fit_raises(donor_Xy):
    X, _ = donor_Xy
    model = DonorPropensityModel()
    with pytest.raises(NotFittedError):
        model.predict_proba(X)


def test_affinity_score_before_fit_raises(donor_Xy):
    X, _ = donor_Xy
    model = DonorPropensityModel()
    with pytest.raises(NotFittedError):
        model.predict_affinity_score(X)


# ---------------------------------------------------------------------------
# get_params / set_params (sklearn contracts)
# ---------------------------------------------------------------------------

def test_get_params_keys():
    model = DonorPropensityModel(n_estimators=77, random_state=3)
    params = model.get_params()
    expected_keys = {
        "n_estimators", "max_depth", "min_samples_split",
        "min_samples_leaf", "min_weight_fraction_leaf",
        "class_weight", "random_state",
    }
    assert expected_keys == set(params.keys())


def test_get_params_values():
    model = DonorPropensityModel(n_estimators=77, random_state=3)
    params = model.get_params()
    assert params["n_estimators"] == 77
    assert params["random_state"] == 3


def test_set_params_round_trip():
    model = DonorPropensityModel(n_estimators=50)
    model.set_params(n_estimators=200, random_state=99)
    assert model.n_estimators == 200
    assert model.random_state == 99


# ---------------------------------------------------------------------------
# Pandas DataFrame input (I/O contract)
# ---------------------------------------------------------------------------

def test_accepts_pandas_dataframe():
    import pandas as pd
    df = generate_synthetic_donor_data(n_samples=100, random_state=20)
    feature_cols = ["total_gift_amount", "years_active", "event_attendance_count"]
    X_df = df[feature_cols]
    y = df["is_major_donor"].to_numpy()
    model = DonorPropensityModel(n_estimators=5, random_state=0)
    model.fit(X_df, y)
    preds = model.predict(X_df)
    assert preds.shape == (100,)


# ---------------------------------------------------------------------------
# Pipeline compatibility
# ---------------------------------------------------------------------------

def test_pipeline_fit_predict(donor_Xy):
    X, y = donor_Xy
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", DonorPropensityModel(n_estimators=5, random_state=0)),
    ])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds).issubset({0, 1})


def test_pipeline_predict_proba(donor_Xy):
    X, y = donor_Xy
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", DonorPropensityModel(n_estimators=5, random_state=0)),
    ])
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_reproducible_with_random_state(donor_Xy):
    X, y = donor_Xy
    m1 = DonorPropensityModel(n_estimators=10, random_state=42).fit(X, y)
    m2 = DonorPropensityModel(n_estimators=10, random_state=42).fit(X, y)
    np.testing.assert_array_equal(m1.predict(X), m2.predict(X))


# ---------------------------------------------------------------------------
# clone() compatibility (sklearn.base.clone)
# ---------------------------------------------------------------------------

def test_clone_preserves_params():
    from sklearn.base import clone
    model = DonorPropensityModel(n_estimators=55, max_depth=4, random_state=9)
    cloned = clone(model)
    assert cloned.n_estimators == 55
    assert cloned.max_depth == 4
    assert cloned.random_state == 9
    # Clone must NOT carry over fitted state
    assert not hasattr(cloned, "estimator_")


# ---------------------------------------------------------------------------
# sklearn.utils.estimator_checks.check_estimator
# ---------------------------------------------------------------------------

try:
    # scikit-learn >= 1.1: use the dedicated parametrize helper
    from sklearn.utils.estimator_checks import parametrize_with_checks

    @parametrize_with_checks([DonorPropensityModel()])
    def test_sklearn_estimator_checks(estimator, check):
        """Pass every check_estimator test generated by scikit-learn."""
        check(estimator)

except ImportError:
    # Fallback for older sklearn – generate_only flag still available
    from sklearn.utils.estimator_checks import check_estimator as _ce

    @pytest.mark.parametrize("estimator, check", _ce(
        DonorPropensityModel(), generate_only=True
    ))
    def test_sklearn_estimator_checks(estimator, check):
        """Pass every check_estimator test generated by scikit-learn."""
        check(estimator)
