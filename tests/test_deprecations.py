"""tests/test_deprecations.py

Every deprecation the 0.6.0 CHANGELOG promises, asserted here.

Two shapes:

* **Renamed methods** — the old name still works, warns `DeprecationWarning`,
  names its replacement, and returns exactly what the new name returns.
* **Dead constructor params** — a non-default value warns; the default is
  silent, so existing users who never touched the knob see nothing.

`test_every_alias_is_registered_here` makes adding a shim without a test fail.
"""

import warnings

import numpy as np
import pytest

from philanthropy.model_selection import FiscalYearGroupedSplitter
from philanthropy.models import (
    AskAmountRecommender,
    LapsePredictor,
    MovesManagementClassifier,
    PlannedGivingIntentScorer,
    PropensityScorer,
    ShareOfWalletRegressor,
)
from philanthropy.utils._deprecation import deprecated_alias

REMOVED_IN = "0.7.0"


@pytest.fixture(scope="module")
def X():
    return np.random.default_rng(0).random((40, 4))


# ---------------------------------------------------------------------------
# The helper itself
# ---------------------------------------------------------------------------

def test_deprecated_alias_warns_and_delegates():
    class Model:
        def new_name(self, a, b=0):
            return a + b

        @deprecated_alias("new_name", removed_in="9.9.9")
        def old_name(self, a, b=0):
            """Original docstring."""

    m = Model()
    with pytest.warns(DeprecationWarning, match=r"Model\.old_name is deprecated"):
        assert m.old_name(2, b=3) == 5

    assert "9.9.9" in Model.old_name.__doc__
    assert "new_name" in Model.old_name.__doc__


def test_deprecated_alias_keeps_the_method_name():
    class Model:
        def new_name(self):
            return 1

        @deprecated_alias("new_name", removed_in="9.9.9")
        def old_name(self): ...

    assert Model.old_name.__name__ == "old_name"


# ---------------------------------------------------------------------------
# Renamed methods (W3.5, W3.6)
# ---------------------------------------------------------------------------

def test_predict_ask_array_aliases_ask_ladder(X):
    y = np.random.default_rng(1).uniform(100, 1000, len(X))
    m = AskAmountRecommender(max_iter=20, random_state=0).fit(X, y)

    with pytest.warns(DeprecationWarning, match=r"use \.ask_ladder instead"):
        old = m.predict_ask_array(X[:3])
    np.testing.assert_array_equal(old, m.ask_ladder(X[:3]))


def test_predict_capacity_ratio_aliases_capacity_ratio(X):
    y = np.random.default_rng(2).uniform(1e4, 1e6, len(X))
    m = ShareOfWalletRegressor(max_iter=20, random_state=0).fit(X, y)
    hist = np.array([100.0, 200.0, 300.0])

    with pytest.warns(DeprecationWarning, match=r"use \.capacity_ratio instead"):
        old = m.predict_capacity_ratio(X[:3], historical_giving=hist)
    np.testing.assert_array_equal(old, m.capacity_ratio(X[:3], historical_giving=hist))


def test_predict_action_priority_aliases_action_priority(X):
    y = np.tile(["IDENTIFY", "QUALIFY"], len(X) // 2)
    m = MovesManagementClassifier(max_iter=10, random_state=0).fit(X, y)

    with pytest.warns(DeprecationWarning, match=r"use \.action_priority instead"):
        old = m.predict_action_priority(X[:3])
    new = m.action_priority(X[:3])
    assert set(old) == set(new)
    np.testing.assert_array_equal(old["stage"], new["stage"])


def test_predict_bequest_intent_score_aliases_predict_intent_score(X):
    y = (X[:, 0] > 0.5).astype(int)
    m = PlannedGivingIntentScorer(n_estimators=10, random_state=0).fit(X, y)

    with pytest.warns(DeprecationWarning, match=r"use \.predict_intent_score instead"):
        old = m.predict_bequest_intent_score(X[:3])
    np.testing.assert_array_equal(old, m.predict_intent_score(X[:3]))


@pytest.mark.parametrize("cls, old_name, new_name", [
    (AskAmountRecommender, "predict_ask_array", "ask_ladder"),
    (ShareOfWalletRegressor, "predict_capacity_ratio", "capacity_ratio"),
    (MovesManagementClassifier, "predict_action_priority", "action_priority"),
    (PlannedGivingIntentScorer, "predict_bequest_intent_score", "predict_intent_score"),
])
def test_every_alias_names_its_replacement_and_removal_version(cls, old_name, new_name):
    doc = getattr(cls, old_name).__doc__
    assert new_name in doc
    assert REMOVED_IN in doc


def test_every_alias_is_registered_here():
    """A shim added without a test in this file fails the build."""
    import inspect

    import philanthropy.experimental as experimental
    import philanthropy.models as models

    registered = {
        "predict_ask_array",
        "predict_capacity_ratio",
        "predict_action_priority",
        "predict_bequest_intent_score",
    }
    # deprecated_alias stamps every wrapper's docstring with this prefix, so
    # the scan finds alias shims without also matching the inline dead-param
    # warnings in fit()/split().
    found = {
        attr
        for module in (models, experimental)
        for name in module.__all__
        for attr, obj in vars(getattr(module, name)).items()
        if inspect.isfunction(obj)
        and (obj.__doc__ or "").startswith("Deprecated alias of")
    }
    assert found == registered, (
        f"untested deprecation shims: {sorted(found - registered)}; "
        f"registered but gone: {sorted(registered - found)}"
    )


# ---------------------------------------------------------------------------
# Dead constructor params (W3.7)
# ---------------------------------------------------------------------------

def test_lapse_window_years_warns_only_when_set(X):
    y = (X[:, 0] > 0.5).astype(int)

    with pytest.warns(DeprecationWarning, match=r"lapse_window_years.*no effect"):
        LapsePredictor(n_estimators=5, lapse_window_years=3, random_state=0).fit(X, y)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        LapsePredictor(n_estimators=5, random_state=0).fit(X, y)


def test_propensity_scorer_estimator_warns_only_when_set(X):
    y = (X[:, 0] > 0.5).astype(int)

    with pytest.warns(DeprecationWarning, match=r"PropensityScorer\(estimator=\.\.\.\) is unused"):
        PropensityScorer(estimator=LapsePredictor()).fit(X, y)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        PropensityScorer().fit(X, y)


def test_splitter_fiscal_year_start_warns_only_when_set():
    groups = [2018, 2018, 2019, 2019, 2020, 2020]
    Xs = np.zeros((6, 2))

    with pytest.warns(DeprecationWarning, match=r"fiscal_year_start.*no\s+effect"):
        list(FiscalYearGroupedSplitter(n_splits=2, fiscal_year_start=1).split(
            Xs, groups=groups
        ))

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        list(FiscalYearGroupedSplitter(n_splits=2).split(Xs, groups=groups))


@pytest.mark.parametrize("cls, param", [
    (LapsePredictor, "lapse_window_years"),
    (PropensityScorer, "estimator"),
])
def test_dead_estimator_params_still_round_trip_through_get_params(cls, param):
    """Deprecated is not removed: clone() must keep working until 0.7.0."""
    from sklearn.base import clone

    obj = cls()
    assert param in obj.get_params()
    assert param in clone(obj).get_params()


def test_splitter_dead_param_survives_on_the_instance():
    # FiscalYearGroupedSplitter is a BaseCrossValidator, not a BaseEstimator,
    # so it has no get_params; the attribute and __repr__ are the contract.
    s = FiscalYearGroupedSplitter(fiscal_year_start=1)
    assert s.fiscal_year_start == 1
    assert "fiscal_year_start=1" in repr(s)
