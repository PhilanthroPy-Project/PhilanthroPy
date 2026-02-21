"""
tests/test_propensity.py
"""

import numpy as np
import pytest
from philanthropy.models import PropensityScorer, LapsePredictor


@pytest.fixture
def dummy_Xy():
    rng = np.random.default_rng(1)
    X = rng.random((40, 5))
    y = rng.integers(0, 2, size=40)
    return X, y


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


def test_lapse_predictor_fit_predict(dummy_Xy):
    X, y = dummy_Xy
    clf = LapsePredictor()
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (40,)
    assert set(preds).issubset({0, 1})
