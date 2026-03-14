"""
tests/test_major_gift_classifier.py
Test suite for MajorGiftClassifier.
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from philanthropy.models import MajorGiftClassifier


@pytest.fixture
def mg_Xy():
    rng = np.random.default_rng(42)
    X = rng.random((100, 5))
    y = rng.integers(0, 2, size=100)
    return X, y


def test_fit_predict_shapes(mg_Xy):
    X, y = mg_Xy
    clf = MajorGiftClassifier(max_iter=20, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (100,)


def test_fit_predict_proba_shapes(mg_Xy):
    X, y = mg_Xy
    clf = MajorGiftClassifier(max_iter=20, random_state=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (100, 2)


def test_predict_proba_rows_sum_to_one(mg_Xy):
    X, y = mg_Xy
    clf = MajorGiftClassifier(max_iter=20, random_state=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(100))


def test_predict_affinity_score_range(mg_Xy):
    X, y = mg_Xy
    clf = MajorGiftClassifier(max_iter=20, random_state=0)
    clf.fit(X, y)
    scores = clf.predict_affinity_score(X)
    assert scores.min() >= 0.0
    assert scores.max() <= 100.0


def test_predict_affinity_score_dtype(mg_Xy):
    X, y = mg_Xy
    clf = MajorGiftClassifier(max_iter=20, random_state=0)
    clf.fit(X, y)
    scores = clf.predict_affinity_score(X)
    assert np.issubdtype(scores.dtype, np.floating)


def test_predict_affinity_score_shape(mg_Xy):
    X, y = mg_Xy
    clf = MajorGiftClassifier(max_iter=20, random_state=0)
    clf.fit(X, y)
    scores = clf.predict_affinity_score(X)
    assert scores.shape == (100,)


def test_not_fitted_raises(mg_Xy):
    X, y = mg_Xy
    clf = MajorGiftClassifier()
    with pytest.raises(NotFittedError):
        clf.predict(X)
    with pytest.raises(NotFittedError):
        clf.predict_proba(X)
    with pytest.raises(NotFittedError):
        clf.predict_affinity_score(X)


def test_pipeline_compatibility(mg_Xy):
    X, y = mg_Xy
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MajorGiftClassifier(max_iter=20, random_state=0)),
    ])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (100,)


def test_clone_compatibility(mg_Xy):
    X, y = mg_Xy
    clf = MajorGiftClassifier()
    clf.fit(X, y)
    cloned = clone(clf)
    assert not hasattr(cloned, "estimator_")


def test_get_params_set_params_round_trip():
    clf = MajorGiftClassifier(max_iter=50, learning_rate=0.05)
    params = clf.get_params()
    assert params["max_iter"] == 50
    assert params["learning_rate"] == 0.05
    clf.set_params(max_iter=30)
    assert clf.max_iter == 30


def test_works_with_nan_inputs(mg_Xy):
    X, y = mg_Xy
    X_nan = X.copy()
    X_nan[0:10, 0] = np.nan
    clf = MajorGiftClassifier(max_iter=20, random_state=0)
    clf.fit(X_nan, y)
    preds = clf.predict(X_nan)
    assert preds.shape == (100,)
    assert not np.any(np.isnan(preds))


def test_classes_after_fit(mg_Xy):
    X, y = mg_Xy
    clf = MajorGiftClassifier(max_iter=20, random_state=0)
    clf.fit(X, y)
    assert hasattr(clf, "classes_")
    assert np.array_equal(np.sort(clf.classes_), np.array([0, 1]))


def test_n_features_in_after_fit(mg_Xy):
    X, y = mg_Xy
    clf = MajorGiftClassifier(max_iter=20, random_state=0)
    clf.fit(X, y)
    assert clf.n_features_in_ == 5


def test_predict_returns_ndarray(mg_Xy):
    X, y = mg_Xy
    clf = MajorGiftClassifier(max_iter=20, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert isinstance(preds, np.ndarray)


def test_predict_proba_returns_ndarray(mg_Xy):
    X, y = mg_Xy
    clf = MajorGiftClassifier(max_iter=20, random_state=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert isinstance(proba, np.ndarray)


def test_random_state_reproducibility(mg_Xy):
    X, y = mg_Xy
    clf1 = MajorGiftClassifier(max_iter=20, random_state=42)
    clf2 = MajorGiftClassifier(max_iter=20, random_state=42)
    clf1.fit(X, y)
    clf2.fit(X, y)
    np.testing.assert_array_equal(clf1.predict(X), clf2.predict(X))


def test_single_sample(mg_Xy):
    X, y = mg_Xy
    clf = MajorGiftClassifier(max_iter=20, random_state=0)
    clf.fit(X, y)
    pred = clf.predict(X[:1])
    assert pred.shape == (1,)
    score = clf.predict_affinity_score(X[:1])
    assert score.shape == (1,)


def test_fit_returns_self(mg_Xy):
    X, y = mg_Xy
    clf = MajorGiftClassifier(max_iter=5, random_state=0)
    result = clf.fit(X, y)
    assert result is clf


def test_learning_rate_parameter(mg_Xy):
    X, y = mg_Xy
    clf = MajorGiftClassifier(max_iter=20, learning_rate=0.01, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (100,)


def test_estimator_attribute_after_fit(mg_Xy):
    X, y = mg_Xy
    clf = MajorGiftClassifier(max_iter=20, random_state=0)
    clf.fit(X, y)
    assert hasattr(clf, "estimator_")
