"""
tests/test_ask.py
Test suite for AskAmountRecommender.
"""

import numpy as np
import pytest

from philanthropy.models import AskAmountRecommender


@pytest.fixture
def ask_Xy():
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1e6, (100, 5))
    y = rng.uniform(1e3, 250_000, 100)
    return X, y


def test_predict_shape_and_floor(ask_Xy):
    X, y = ask_Xy
    model = AskAmountRecommender(max_iter=20, random_state=0)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (100,)
    assert (preds >= model.ask_floor).all()


def test_predict_respects_custom_floor(ask_Xy):
    X, y = ask_Xy
    model = AskAmountRecommender(ask_floor=5000.0, max_iter=20, random_state=0)
    model.fit(X, y)
    preds = model.predict(X)
    assert (preds >= 5000.0).all()


def test_ask_array_shape_and_monotonic(ask_Xy):
    X, y = ask_Xy
    model = AskAmountRecommender(max_iter=20, random_state=0)
    model.fit(X, y)
    ladder = model.predict_ask_array(X)
    assert ladder.shape == (100, 3)
    # Columns ascend with the (ascending) default multipliers, elementwise.
    assert (ladder[:, 1] >= ladder[:, 0]).all()
    assert (ladder[:, 2] >= ladder[:, 1]).all()


def test_ask_array_matches_base_times_multipliers(ask_Xy):
    X, y = ask_Xy
    model = AskAmountRecommender(max_iter=20, random_state=0)
    model.fit(X, y)
    multipliers = (1.0, 1.5, 2.5)
    ladder = model.predict_ask_array(X, multipliers=multipliers)
    expected = model.predict(X)[:, None] * np.asarray(multipliers)[None, :]
    np.testing.assert_array_almost_equal(ladder, expected)


def test_nan_input_handled(ask_Xy):
    X, y = ask_Xy
    X_nan = X.copy()
    X_nan[0:10, 0] = np.nan
    model = AskAmountRecommender(max_iter=20, random_state=0)
    model.fit(X_nan, y)
    preds = model.predict(X_nan)
    assert preds.shape == (100,)
    assert not np.any(np.isnan(preds))


def test_random_state_reproducibility(ask_Xy):
    X, y = ask_Xy
    m1 = AskAmountRecommender(max_iter=20, random_state=7).fit(X, y)
    m2 = AskAmountRecommender(max_iter=20, random_state=7).fit(X, y)
    np.testing.assert_array_equal(m1.predict(X), m2.predict(X))


def test_ask_array_rejects_empty_multipliers(ask_Xy):
    X, y = ask_Xy
    model = AskAmountRecommender(max_iter=20, random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        model.predict_ask_array(X, multipliers=())


def test_ask_array_rejects_negative_multipliers(ask_Xy):
    X, y = ask_Xy
    model = AskAmountRecommender(max_iter=20, random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        model.predict_ask_array(X, multipliers=(1.0, -2.0))


def test_fit_returns_self(ask_Xy):
    X, y = ask_Xy
    model = AskAmountRecommender(max_iter=5, random_state=0)
    assert model.fit(X, y) is model


def test_n_features_in_after_fit(ask_Xy):
    X, y = ask_Xy
    model = AskAmountRecommender(max_iter=5, random_state=0).fit(X, y)
    assert model.n_features_in_ == 5


def test_n_iter_property(ask_Xy):
    X, y = ask_Xy
    model = AskAmountRecommender(max_iter=20, random_state=0).fit(X, y)
    assert model.n_iter_ >= 1
