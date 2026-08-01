"""
tests/test_uplift.py
====================
Tests for the experimental UpliftTLearner (T-learner) estimator.
"""

import numpy as np
import pytest

from philanthropy.experimental import UpliftTLearner


def make_uplift_data(n=600, seed=0):
    """Synthetic data where the appeal causally lifts giving for a subgroup.

    A treated donor with ``feature > 0`` gives with p=0.8; everyone else
    (control, or feature<=0) gives with p=0.2.  So uplift is concentrated in
    the responsive subgroup ``feature > 0``.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 1))
    treatment = rng.integers(0, 2, size=n)
    responsive = X[:, 0] > 0
    p = np.where((treatment == 1) & responsive, 0.8, 0.2)
    y = (rng.random(n) < p).astype(int)
    return X, y, treatment


def test_predict_uplift_score_shape_and_dtype():
    X, y, treatment = make_uplift_data()
    model = UpliftTLearner(random_state=0).fit(X, y, treatment)
    uplift = model.predict_uplift_score(X)
    assert uplift.shape == (X.shape[0],)
    assert np.issubdtype(uplift.dtype, np.floating)


def test_uplift_score_in_range():
    X, y, treatment = make_uplift_data()
    model = UpliftTLearner(random_state=0).fit(X, y, treatment)
    uplift = model.predict_uplift_score(X)
    assert (uplift >= -1.0).all()
    assert (uplift <= 1.0).all()


def test_uplift_positive_on_responsive_subgroup():
    X, y, treatment = make_uplift_data()
    model = UpliftTLearner(random_state=0).fit(X, y, treatment)
    uplift = model.predict_uplift_score(X)
    responsive = X[:, 0] > 0
    assert uplift[responsive].mean() > 0


def test_predict_returns_binary():
    X, y, treatment = make_uplift_data()
    model = UpliftTLearner(random_state=0).fit(X, y, treatment)
    decision = model.predict(X)
    assert set(np.unique(decision)).issubset({0, 1})
    assert decision.shape == (X.shape[0],)


def test_fit_rejects_non_binary_treatment():
    X, y, treatment = make_uplift_data(n=50)
    treatment = treatment.copy()
    treatment[0] = 2
    with pytest.raises(ValueError):
        UpliftTLearner(random_state=0).fit(X, y, treatment)


def test_fit_rejects_length_mismatch():
    X, y, treatment = make_uplift_data(n=50)
    with pytest.raises(ValueError):
        UpliftTLearner(random_state=0).fit(X, y, treatment[:-1])


def test_fit_rejects_single_arm():
    X, y, treatment = make_uplift_data(n=50)
    all_treated = np.ones_like(treatment)
    with pytest.raises(ValueError):
        UpliftTLearner(random_state=0).fit(X, y, all_treated)


def test_random_state_reproducibility():
    X, y, treatment = make_uplift_data()
    a = UpliftTLearner(random_state=42).fit(X, y, treatment).predict_uplift_score(X)
    b = UpliftTLearner(random_state=42).fit(X, y, treatment).predict_uplift_score(X)
    np.testing.assert_array_equal(a, b)


def test_single_class_arm_no_index_error():
    # Control arm sees only y == 0 (nobody in control gave); treated arm mixed.
    rng = np.random.default_rng(1)
    n = 80
    X = rng.normal(size=(n, 1))
    treatment = rng.integers(0, 2, size=n)
    y = np.where(treatment == 1, rng.integers(0, 2, size=n), 0)
    model = UpliftTLearner(random_state=0).fit(X, y, treatment)
    uplift = model.predict_uplift_score(X)  # must not raise IndexError
    assert uplift.shape == (n,)
    assert (uplift >= -1.0).all() and (uplift <= 1.0).all()
