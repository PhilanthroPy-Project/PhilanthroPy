"""
tests/test_propensity.py
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from philanthropy.models import PropensityScorer, LapsePredictor


@pytest.fixture
def dummy_Xy():
    rng = np.random.default_rng(1)
    X = rng.random((40, 5))
    y = rng.integers(0, 2, size=40)
    return X, y


# ---------------------------------------------------------------------------
# PropensityScorer tests (original)
# ---------------------------------------------------------------------------

def test_propensity_scorer_fit_predict(dummy_Xy):
    X, y = dummy_Xy
    clf = PropensityScorer()
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (40,)
    assert set(preds).issubset({0, 1})


def test_propensity_scorer_predict_proba_shape(dummy_Xy):
    X, y = dummy_Xy
    clf = PropensityScorer()
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (40, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# LapsePredictor tests (production implementation)
# ---------------------------------------------------------------------------

def test_lapse_predictor_fit_predict(dummy_Xy):
    X, y = dummy_Xy
    clf = LapsePredictor(lapse_window_years=2)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (40,)
    assert set(preds).issubset({0, 1})
    assert preds.dtype in (np.int32, np.int64)


def test_lapse_predictor_predict_proba_shape(dummy_Xy):
    X, y = dummy_Xy
    clf = LapsePredictor()
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (40, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert proba.dtype == np.float64


def test_lapse_predictor_predict_lapse_score_range(dummy_Xy):
    X, y = dummy_Xy
    clf = LapsePredictor()
    clf.fit(X, y)
    scores = clf.predict_lapse_score(X)
    assert scores.shape == (40,)
    assert scores.dtype == np.float64
    assert scores.min() >= 0.0
    assert scores.max() <= 100.0


def test_lapse_predictor_predict_lapse_score_two_decimal_places(dummy_Xy):
    X, y = dummy_Xy
    clf = LapsePredictor()
    clf.fit(X, y)
    scores = clf.predict_lapse_score(X)
    for s in scores:
        rounded = np.round(s, 2)
        assert s == rounded


def test_lapse_predictor_not_fitted_raises(dummy_Xy):
    X, y = dummy_Xy
    clf = LapsePredictor()
    with pytest.raises(NotFittedError):
        clf.predict(X)
    with pytest.raises(NotFittedError):
        clf.predict_proba(X)
    with pytest.raises(NotFittedError):
        clf.predict_lapse_score(X)


def test_lapse_predictor_pipeline_compatibility(dummy_Xy):
    X, y = dummy_Xy
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LapsePredictor(n_estimators=20, random_state=0)),
    ])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (40,)


def test_lapse_predictor_clone_compatibility(dummy_Xy):
    X, y = dummy_Xy
    clf = LapsePredictor()
    clf.fit(X, y)
    clf2 = clone(clf)
    assert not hasattr(clf2, "estimator_")


def test_lapse_predictor_lapse_window_years_values(dummy_Xy):
    X, y = dummy_Xy
    for years in [1, 2, 5, 10]:
        clf = LapsePredictor(lapse_window_years=years)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (40,)


def test_lapse_predictor_class_weight_balanced(dummy_Xy):
    X, y = dummy_Xy
    clf = LapsePredictor(class_weight="balanced")
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (40,)
    assert hasattr(clf, "estimator_")


def test_lapse_predictor_random_state_reproducibility(dummy_Xy):
    X, y = dummy_Xy
    clf1 = LapsePredictor(random_state=42)
    clf2 = LapsePredictor(random_state=42)
    clf1.fit(X, y)
    clf2.fit(X, y)
    np.testing.assert_array_equal(clf1.predict(X), clf2.predict(X))
    np.testing.assert_array_equal(clf1.predict_lapse_score(X), clf2.predict_lapse_score(X))


def test_lapse_predictor_fitted_attributes(dummy_Xy):
    X, y = dummy_Xy
    clf = LapsePredictor()
    clf.fit(X, y)
    assert hasattr(clf, "estimator_")
    assert hasattr(clf, "classes_")
    assert hasattr(clf, "n_features_in_")
    assert clf.n_features_in_ == 5


def test_lapse_predictor_predict_returns_ndarray(dummy_Xy):
    X, y = dummy_Xy
    clf = LapsePredictor()
    clf.fit(X, y)
    preds = clf.predict(X)
    assert isinstance(preds, np.ndarray)


def test_lapse_predictor_predict_proba_returns_ndarray(dummy_Xy):
    X, y = dummy_Xy
    clf = LapsePredictor()
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert isinstance(proba, np.ndarray)


def test_lapse_predictor_predict_lapse_score_returns_ndarray(dummy_Xy):
    X, y = dummy_Xy
    clf = LapsePredictor()
    clf.fit(X, y)
    scores = clf.predict_lapse_score(X)
    assert isinstance(scores, np.ndarray)


def test_lapse_predictor_single_sample(dummy_Xy):
    X, y = dummy_Xy
    clf = LapsePredictor()
    clf.fit(X, y)
    pred = clf.predict(X[:1])
    assert pred.shape == (1,)
    score = clf.predict_lapse_score(X[:1])
    assert score.shape == (1,)


def test_lapse_predictor_max_depth_none(dummy_Xy):
    X, y = dummy_Xy
    clf = LapsePredictor(max_depth=None)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (40,)


def test_lapse_predictor_max_depth_int(dummy_Xy):
    X, y = dummy_Xy
    clf = LapsePredictor(max_depth=5)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (40,)


def test_lapse_predictor_get_params_set_params():
    clf = LapsePredictor(n_estimators=50, lapse_window_years=3)
    params = clf.get_params()
    assert params["n_estimators"] == 50
    assert params["lapse_window_years"] == 3
    clf.set_params(n_estimators=30)
    assert clf.n_estimators == 30


@parametrize_with_checks([LapsePredictor(n_estimators=10, random_state=0)])
def test_lapse_predictor_sklearn_compliance(estimator, check):
    """LapsePredictor must pass check_estimator (fit uses standard X, y)."""
    check(estimator)
