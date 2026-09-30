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
    ladder = model.ask_ladder(X)
    assert ladder.shape == (100, 3)
    # Columns ascend with the (ascending) default multipliers, elementwise.
    assert (ladder[:, 1] >= ladder[:, 0]).all()
    assert (ladder[:, 2] >= ladder[:, 1]).all()


def test_ask_array_matches_base_times_multipliers(ask_Xy):
    X, y = ask_Xy
    model = AskAmountRecommender(max_iter=20, random_state=0)
    model.fit(X, y)
    multipliers = (1.0, 1.5, 2.5)
    ladder = model.ask_ladder(X, multipliers=multipliers)
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
        model.ask_ladder(X, multipliers=())


def test_ask_array_rejects_negative_multipliers(ask_Xy):
    X, y = ask_Xy
    model = AskAmountRecommender(max_iter=20, random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        model.ask_ladder(X, multipliers=(1.0, -2.0))


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


def test_relative_target_mode_predicts_positive_and_floored():
    rng = np.random.default_rng(0)
    last_gift = rng.uniform(10, 500, 200)
    avg_gift = rng.uniform(10, 500, 200)
    other = rng.uniform(0, 1, (200, 3))
    X = np.column_stack([last_gift, avg_gift, other])
    y = np.maximum(last_gift, avg_gift) * rng.uniform(0.5, 1.5, 200)

    model = AskAmountRecommender(
        target_mode="relative", last_gift_idx=0, avg_gift_idx=1, max_iter=50, random_state=0,
    ).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (200,)
    assert (preds >= model.ask_floor).all()


def test_relative_target_mode_recovers_exact_ratio():
    # A perfect log-linear relationship: the estimator should reconstruct it
    # to a tight tolerance, proving the log/exp round-trip through the
    # reference amount is correct, not just "doesn't crash".
    rng = np.random.default_rng(1)
    last_gift = rng.uniform(50, 500, 300)
    avg_gift = rng.uniform(50, 500, 300)
    ref = np.maximum(last_gift, avg_gift)
    y = ref * 1.2  # every donor gives exactly 1.2x the rule
    X = np.column_stack([last_gift, avg_gift])

    model = AskAmountRecommender(
        target_mode="relative", last_gift_idx=0, avg_gift_idx=1, max_iter=200, random_state=0,
    ).fit(X, y)
    preds = model.predict(X)
    np.testing.assert_allclose(preds, y, rtol=0.05)


def test_relative_target_mode_requires_indices():
    X = np.random.default_rng(0).uniform(0, 1000, (20, 3))
    y = np.random.default_rng(1).uniform(0, 1000, 20)
    model = AskAmountRecommender(target_mode="relative", random_state=0)
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_unknown_target_mode_raises(ask_Xy):
    X, y = ask_Xy
    model = AskAmountRecommender(target_mode="bogus", random_state=0)
    with pytest.raises(ValueError):
        model.fit(X, y)
