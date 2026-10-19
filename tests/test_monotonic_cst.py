"""monotonic_cst on the HistGradientBoosting-backed models."""

import numpy as np
import pandas as pd
import pytest

from philanthropy.models import (
    AskAmountRecommender,
    LapsePredictor,
    MajorGiftClassifier,
    PlannedGivingIntentScorer,
    ShareOfWalletRegressor,
)


def _score(est, X):
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    return est.predict(X)


# (factory, regression?) for every model that takes monotonic_cst.
MODELS = [
    (
        lambda **kw: AskAmountRecommender(
            loss="squared_error", random_state=0, **kw
        ),
        True,
    ),
    (lambda **kw: ShareOfWalletRegressor(random_state=0, **kw), True),
    (
        lambda **kw: LapsePredictor(
            backend="hist_gradient_boosting", random_state=0, **kw
        ),
        False,
    ),
    (lambda **kw: PlannedGivingIntentScorer(random_state=0, **kw), False),
    (lambda **kw: MajorGiftClassifier(random_state=0, **kw), False),
]
IDS = ["ask", "wallet", "lapse", "planned_giving", "major_gift"]


def _data(regression):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(300, 3)), columns=["total_giving", "age", "events"]
    )
    # Non-monotone in total_giving, so an unconstrained fit is free to dip.
    signal = np.sin(3 * X["total_giving"]) + X["age"] + rng.normal(size=300)
    y = signal * 100 + 1000 if regression else (signal > 0).astype(int)
    return X, np.asarray(y)


@pytest.mark.parametrize("make,regression", MODELS, ids=IDS)
def test_positive_constraint_is_non_decreasing(make, regression):
    X, y = _data(regression)
    est = make(monotonic_cst={"total_giving": 1}).fit(X, y)
    grid = X.iloc[[0] * 50].copy()
    grid["total_giving"] = np.linspace(-3, 3, 50)
    assert np.all(np.diff(_score(est, grid)) >= -1e-12)


@pytest.mark.parametrize("make,regression", MODELS, ids=IDS)
def test_name_keyed_dict_matches_position_list(make, regression):
    X, y = _data(regression)
    by_name = make(monotonic_cst={"total_giving": 1, "events": -1}).fit(X, y)
    by_list = make(monotonic_cst=[1, 0, -1]).fit(X, y)
    by_index = make(monotonic_cst={0: 1, 2: -1}).fit(X.to_numpy(), y)
    np.testing.assert_allclose(_score(by_name, X), _score(by_list, X))
    np.testing.assert_allclose(_score(by_name, X), _score(by_index, X))


def test_unknown_key_raises():
    X, y = _data(True)
    with pytest.raises(ValueError, match="not in the data"):
        ShareOfWalletRegressor(monotonic_cst={"wealth": 1}).fit(X, y)
    with pytest.raises(ValueError, match="not a column position"):
        ShareOfWalletRegressor(monotonic_cst={5: 1}).fit(X.to_numpy(), y)


def test_ask_absolute_error_refuses_constraint():
    X, y = _data(True)
    with pytest.raises(ValueError, match="squared_error"):
        AskAmountRecommender(monotonic_cst={"total_giving": 1}).fit(X, y)


def test_random_forest_lapse_ignores_constraint():
    X, y = _data(False)
    plain = LapsePredictor(random_state=0).fit(X, y)
    constrained = LapsePredictor(
        monotonic_cst={"total_giving": 1}, random_state=0
    ).fit(X, y)
    np.testing.assert_array_equal(
        plain.predict_proba(X), constrained.predict_proba(X)
    )

