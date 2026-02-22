"""
tests/test_propensity.py
"""

import numpy as np
import pandas as pd
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
    X, _ = dummy_Xy
    # Generate some synthetic gift dates
    rng = np.random.default_rng(2)
    days_ago = rng.integers(0, 1500, size=40)
    gift_dates = pd.Timestamp.today() - pd.to_timedelta(days_ago, unit="D")
    
    clf = LapsePredictor(lapse_window_years=2)
    clf.fit(X, gift_dates)
    preds = clf.predict(X)
    assert preds.shape == (40,)
    assert set(preds).issubset({0, 1})
