"""Model inputs as a CRM export delivers them: named columns, category codes
and per-donor weights.

Covers three pass-throughs on the HistGradientBoosting-backed models:
column names for ``AskAmountRecommender``'s gift columns, the
``categorical_features`` param, and ``sample_weight`` in ``fit``.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from philanthropy.models import (
    AskAmountRecommender,
    LapsePredictor,
    MovesManagementClassifier,
    PlannedGivingIntentScorer,
    ShareOfWalletRegressor,
)


def _export(n=300, seed=0):
    """A small CRM-style frame: two gift columns, a region code, noise."""
    rng = np.random.default_rng(seed)
    region = rng.integers(0, 8, n)
    X = pd.DataFrame({
        "noise": rng.normal(size=n),
        "last_gift": rng.uniform(10, 1000, n),
        "region": region,
        "avg_gift": rng.uniform(10, 1000, n),
    })
    # Non-monotone in the region code, so a categorical split is what fits it.
    effect = np.isin(region, [1, 4, 6])
    y_amount = X["last_gift"].to_numpy() * (1 + effect) + rng.uniform(0, 50, n)
    y_binary = (effect ^ (rng.random(n) < 0.15)).astype(int)
    y_stage = np.where(effect, "SOLICIT", np.where(region < 4, "IDENTIFY", "QUALIFY"))
    return X, y_amount, y_binary, y_stage


# (estimator factory, target key, how to reach the fitted HistGradientBoosting)
_MODELS = {
    "ask": (lambda **kw: AskAmountRecommender(max_iter=30, random_state=0, **kw),
            "amount", lambda m: m.estimator_),
    "wallet": (lambda **kw: ShareOfWalletRegressor(max_iter=30, random_state=0, **kw),
               "amount", lambda m: m.estimator_),
    "lapse": (lambda **kw: LapsePredictor(
                  backend="hist_gradient_boosting", random_state=0, **kw),
              "binary", lambda m: m.estimator_),
    "moves": (lambda **kw: MovesManagementClassifier(
                  max_iter=30, class_weight=None, random_state=0, **kw),
              "stage", lambda m: m.estimator_),
    "planned": (lambda **kw: PlannedGivingIntentScorer(
                    n_estimators=30, random_state=0, **kw),
                "binary", lambda m: m.estimator_.calibrated_classifiers_[0].estimator),
}


def _target(key, amount, binary, stage):
    return {"amount": amount, "binary": binary, "stage": stage}[key]


def _output(model, X):
    return model.predict_proba(X) if hasattr(model, "predict_proba") else model.predict(X)


# -- column names -------------------------------------------------------------

def test_ask_gift_columns_by_name_match_by_position():
    X, y, _, _ = _export()
    by_name = AskAmountRecommender(
        target_mode="relative", last_gift_idx="last_gift", avg_gift_idx="avg_gift",
        random_state=0,
    ).fit(X, y)
    by_index = AskAmountRecommender(
        target_mode="relative", last_gift_idx=1, avg_gift_idx=3, random_state=0,
    ).fit(X.to_numpy(), y)

    assert (by_name.last_gift_idx_, by_name.avg_gift_idx_) == (1, 3)
    assert by_name.last_gift_idx == "last_gift"  # the param is left as passed
    np.testing.assert_array_equal(by_name.predict(X), by_index.predict(X.to_numpy()))
    assert by_name.beats_rule_ == by_index.beats_rule_
    assert by_name.model_mae_ == by_index.model_mae_


def test_ask_gift_column_name_follows_a_reordered_export():
    X, y, _, _ = _export()
    reordered = X[["avg_gift", "region", "noise", "last_gift"]]
    model = AskAmountRecommender(
        target_mode="relative", last_gift_idx="last_gift", avg_gift_idx="avg_gift",
        random_state=0,
    ).fit(reordered, y)
    assert (model.last_gift_idx_, model.avg_gift_idx_) == (3, 0)


@pytest.mark.parametrize("X_kind", ["frame", "array"])
def test_ask_unknown_gift_column_name_raises(X_kind):
    X, y, _, _ = _export()
    X = X if X_kind == "frame" else X.to_numpy()
    with pytest.raises(ValueError, match="'last_gfit' is not in the data"):
        AskAmountRecommender(last_gift_idx="last_gfit", avg_gift_idx=3).fit(X, y)


# -- categorical_features -----------------------------------------------------

@pytest.mark.parametrize("name", list(_MODELS))
def test_categorical_code_column_by_name_or_position(name):
    make, key, backend = _MODELS[name]
    X, *targets = _export()
    y = _target(key, *targets)

    by_name = make(categorical_features=["region"]).fit(X, y)
    by_index = make(categorical_features=[2]).fit(X.to_numpy(), y)

    np.testing.assert_array_equal(
        backend(by_name).is_categorical_, [False, False, True, False]
    )
    np.testing.assert_array_equal(_output(by_name, X), _output(by_index, X.to_numpy()))


@pytest.mark.parametrize("name", list(_MODELS))
def test_default_treats_every_column_as_numeric(name):
    make, key, backend = _MODELS[name]
    X, *targets = _export()
    model = make().fit(X, _target(key, *targets))
    assert backend(model).is_categorical_ is None


def test_categorical_feature_name_needs_a_frame():
    X, y, _, _ = _export()
    with pytest.raises(ValueError, match="'region' is not in the data"):
        ShareOfWalletRegressor(categorical_features="region").fit(X.to_numpy(), y)


# -- sample_weight ------------------------------------------------------------

# Zero weight is not the same as dropping a row for sklearn's own
# HistGradientBoosting (binning and min_samples_leaf count rows), nor for a
# random forest's bootstrap or calibration's CV folds, so the contract checked
# here is that weights reach the backend unchanged.
@pytest.mark.parametrize("name", list(_MODELS) + ["lapse_forest"])
def test_sample_weight_reaches_the_backend_unchanged(name):
    if name == "lapse_forest":
        make, key = (lambda: LapsePredictor(n_estimators=20, random_state=0)), "binary"
    else:
        make, key, _ = _MODELS[name]
    X, *targets = _export()
    y = _target(key, *targets)
    w = np.random.default_rng(1).integers(0, 4, len(X)).astype(float)

    weighted = make().fit(X, y, sample_weight=w)
    backend_y = weighted.label_encoder_.transform(y) if name == "moves" else y
    backend = clone(weighted.estimator_).fit(X.to_numpy(), backend_y, sample_weight=w)

    if hasattr(weighted, "predict_proba"):
        np.testing.assert_allclose(weighted.predict_proba(X), backend.predict_proba(X.to_numpy()))
    else:
        np.testing.assert_allclose(weighted.predict(X), backend.predict(X.to_numpy()))
    assert not np.allclose(_output(weighted, X), _output(make().fit(X, y), X))


@pytest.mark.parametrize("make", [
    lambda **kw: PlannedGivingIntentScorer(n_estimators=30, random_state=0, **kw),
    lambda **kw: LapsePredictor(n_estimators=20, random_state=0, **kw),
])
def test_class_weight_equals_the_same_sample_weight(make):
    X, _, y, _ = _export()
    by_class = make(class_weight={0: 1.0, 1: 3.0}).fit(X, y)
    by_row = make().fit(X, y, sample_weight=np.where(y == 1, 3.0, 1.0))
    unweighted = make().fit(X, y)

    np.testing.assert_allclose(by_class.predict_proba(X), by_row.predict_proba(X))
    assert not np.allclose(by_row.predict_proba(X), unweighted.predict_proba(X))


def test_planned_giving_class_weight_and_sample_weight_multiply():
    X, _, y, _ = _export()
    w = np.random.default_rng(2).uniform(0.5, 2.0, len(y))
    both = PlannedGivingIntentScorer(
        n_estimators=30, class_weight={0: 1.0, 1: 3.0}, random_state=0
    ).fit(X, y, sample_weight=w)
    product = PlannedGivingIntentScorer(n_estimators=30, random_state=0).fit(
        X, y, sample_weight=w * np.where(y == 1, 3.0, 1.0)
    )
    np.testing.assert_allclose(both.predict_proba(X), product.predict_proba(X))
