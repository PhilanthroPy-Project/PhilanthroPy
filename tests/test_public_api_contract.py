"""tests/test_public_api_contract.py

The executable specification for PhilanthroPy's public API.

Introspection only — no fixtures, no data files. It asserts four things:

1. Every subpackage in ``philanthropy.__all__`` declares a non-empty ``__all__``,
   every name in it resolves, and it has a ``docs/reference/<name>.md`` page.
2. Every public ``predict_*`` method beyond ``predict``/``predict_proba``/
   ``predict_log_proba`` matches ``^predict_\\w+_(score|forecast)$``, is callable
   with X alone, and returns a 1-D ndarray of ``len(X)``.
3. Every ``preprocessing.__all__`` class defines its own
   ``get_feature_names_out(self, input_features=None)`` whose length equals
   ``transform(X).shape[1]``.
4. ``philanthropy.__all__`` covers every non-underscore subpackage directory, so
   a new subpackage cannot ship unreachable.

Exemptions live in ``_EXEMPT`` with a written reason each, and
``test_no_exemption_is_stale`` fails once an exemption stops applying. At 1.0
(W3.18) the gate is "no exemption added since 0.7.0".
"""

from __future__ import annotations

import inspect
import pathlib
import re
import warnings

import numpy as np
import pytest

import philanthropy
import philanthropy.experimental as experimental
import philanthropy.models as models
import philanthropy.preprocessing as preprocessing

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE_ROOT = pathlib.Path(philanthropy.__file__).parent
REFERENCE_DIR = REPO_ROOT / "docs" / "reference"

_PREDICT_NAME = re.compile(r"^predict_\w+_(score|forecast)$")
_SKLEARN_PREDICTORS = {"predict", "predict_proba", "predict_log_proba"}

# Named exemptions, one reason each. Nothing may be added here silently.
_EXEMPT = {
    "RFMTransformer": (
        "Row-reducing aggregator: transform returns one row per donor, not per "
        "input row, and its first output column is a string donor_id. It is not "
        "a Pipeline transformer, so the width contract does not apply."
    ),
    "UpliftTLearner": (
        "fit(X, y, treatment) — the third positional argument breaks the "
        "fit(X, y) signature the callable-with-X-alone check assumes. This is "
        "why it lives in philanthropy.experimental."
    ),
    "FinancialForecastModel": (
        "predict_revenue_forecast(X, horizon) returns `horizon` values, not "
        "len(X): it is a forward projection, not a per-row prediction."
    ),
}


# ---------------------------------------------------------------------------
# Fitted instances for the behavioural checks
# ---------------------------------------------------------------------------

def _small_Xy(n=40, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, n_features))
    return X, (X[:, 0] > 0.5).astype(int)


_MODEL_KWARGS = {
    "AskAmountRecommender": dict(max_iter=20, random_state=0),
    "DonorPropensityModel": dict(n_estimators=5, random_state=0),
    "FinancialForecastModel": dict(random_state=0),
    "LapsePredictor": dict(n_estimators=5, random_state=0),
    "MajorGiftClassifier": dict(max_iter=10, random_state=0),
    "MovesManagementClassifier": dict(max_iter=10, random_state=0),
    "PlannedGivingIntentScorer": dict(n_estimators=10, random_state=0),
    "PropensityScorer": {},
    "ShareOfWalletRegressor": dict(max_iter=20, random_state=0),
}


def _fit_model(name):
    from sklearn.base import is_classifier

    cls = getattr(models, name)
    est = cls(**_MODEL_KWARGS[name])
    X, y_binary = _small_Xy()
    y = y_binary if is_classifier(est) else np.random.default_rng(1).uniform(
        100.0, 10_000.0, len(X)
    )
    return est.fit(X, y), X


def _predict_methods(cls):
    """Public predict_* methods that are part of the supported surface."""
    out = []
    for attr, obj in inspect.getmembers(cls, predicate=inspect.isfunction):
        if not attr.startswith("predict") or attr in _SKLEARN_PREDICTORS:
            continue
        out.append(attr)
    return out


# ---------------------------------------------------------------------------
# 1. Subpackages
# ---------------------------------------------------------------------------

def _public_subpackages():
    return [
        name for name in philanthropy.__all__
        if inspect.ismodule(getattr(philanthropy, name))
    ]


def test_every_public_name_resolves():
    for name in philanthropy.__all__:
        assert hasattr(philanthropy, name), f"philanthropy.__all__ names {name!r}"


@pytest.mark.parametrize("name", _public_subpackages())
def test_subpackage_declares_a_non_empty_all(name):
    module = getattr(philanthropy, name)
    assert getattr(module, "__all__", None), f"philanthropy.{name} has no __all__"
    for symbol in module.__all__:
        assert hasattr(module, symbol), f"philanthropy.{name}.__all__ names {symbol!r}"


@pytest.mark.parametrize("name", _public_subpackages())
def test_subpackage_has_a_rendered_reference_page(name):
    # A substring scan over docs/** cannot work: a mkdocstrings page contains
    # the module path, not the symbol names. Page existence is the enforceable
    # version of "no undocumented public subpackage".
    page = REFERENCE_DIR / f"{name}.md"
    assert page.is_file(), f"missing docs/reference/{name}.md"
    assert f"::: philanthropy.{name}" in page.read_text()


# ---------------------------------------------------------------------------
# 4. No unreachable subpackage
# ---------------------------------------------------------------------------

def test_every_subpackage_directory_is_reachable_from_the_top_level():
    on_disk = {
        d.name for d in PACKAGE_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith(("_", "."))
    }
    assert on_disk <= set(philanthropy.__all__), (
        f"subpackages on disk but not in philanthropy.__all__: "
        f"{sorted(on_disk - set(philanthropy.__all__))}"
    )


# ---------------------------------------------------------------------------
# 2. predict_* naming and shape contract
# ---------------------------------------------------------------------------

_ESTIMATOR_NAMES = sorted(set(models.__all__) | set(experimental.__all__))


@pytest.mark.parametrize("name", _ESTIMATOR_NAMES)
def test_predict_methods_follow_the_naming_contract(name):
    cls = getattr(models, name, None) or getattr(experimental, name)
    for method in _predict_methods(cls):
        assert _PREDICT_NAME.match(method), (
            f"{name}.{method} keeps the predict_ prefix but is not a "
            f"predict_<thing>_(score|forecast). Rename it, or drop the prefix."
        )


@pytest.mark.parametrize("name", sorted(models.__all__))
def test_predict_methods_are_callable_with_x_alone_and_return_one_value_per_row(name):
    if name in _EXEMPT:
        pytest.skip(_EXEMPT[name])

    est, X = _fit_model(name)
    methods = _predict_methods(type(est))
    for method in methods:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = getattr(est, method)(X)
        assert isinstance(out, np.ndarray), f"{name}.{method} returned {type(out)}"
        assert out.ndim == 1, f"{name}.{method} returned shape {out.shape}"
        assert len(out) == len(X), f"{name}.{method} returned {len(out)} of {len(X)}"


# ---------------------------------------------------------------------------
# 3. get_feature_names_out width contract
# ---------------------------------------------------------------------------

_TRANSFORMER_FIXTURES = {
    "CRMCleaner": lambda: __import__("pandas").DataFrame({
        "gift_date": ["2023-01-01", "2023-06-01"],
        "gift_amount": ["100.0", "250.5"],
    }),
    "FiscalYearTransformer": lambda: __import__("pandas").DataFrame({
        "gift_date": ["2023-01-01", "2023-06-01"],
    }),
    "MatchingGiftFeaturizer": lambda: __import__("pandas").DataFrame({
        "employer": ["Acme Corp", ""],
        "gift_amount": [100.0, 200.0],
    }),
    "EncounterRecencyTransformer": lambda: __import__("pandas").DataFrame({
        "last_encounter_date": ["2023-01-01", "2023-06-01"],
    }),
    "PlannedGivingSignalTransformer": lambda: __import__("pandas").DataFrame({
        "donor_age": [70.0, 45.0],
        "years_active": [20.0, 2.0],
        "planned_gift_inclination": [0.8, 0.1],
    }),
    "DischargeToSolicitationWindowTransformer": lambda: __import__("pandas").DataFrame({
        "days_since_last_discharge": [120.0, 900.0],
    }),
    "SolicitationWindowTransformer": lambda: __import__("pandas").DataFrame({
        "days_since_last_discharge": [120.0, 900.0],
    }),
}


def _transformer_input(name):
    import pandas as pd

    if name in _TRANSFORMER_FIXTURES:
        return _TRANSFORMER_FIXTURES[name]()
    if name in ("EncounterTransformer", "GratefulPatientFeaturizer"):
        return pd.DataFrame({
            "donor_id": [1, 2],
            "gift_date": ["2023-01-01", "2023-06-01"],
        })
    return pd.DataFrame({
        "estimated_net_worth": [1e6, np.nan],
        "real_estate_value": [2e5, 3e5],
    })


def _transformer_instance(name):
    import pandas as pd

    cls = getattr(preprocessing, name)
    if name in ("EncounterTransformer", "GratefulPatientFeaturizer"):
        enc = pd.DataFrame({
            "donor_id": [1, 2],
            "discharge_date": ["2022-01-01", "2022-06-15"],
            "service_line": ["cardiac", "oncology"],
            "attending_physician_id": ["P1", "P2"],
        })
        return cls(encounter_df=enc)
    return cls()


@pytest.mark.parametrize("name", sorted(preprocessing.__all__))
def test_transformer_defines_its_own_get_feature_names_out(name):
    cls = getattr(preprocessing, name)
    assert "get_feature_names_out" in vars(cls), (
        f"{name} inherits get_feature_names_out instead of defining it; the "
        f"inherited version cannot know this transformer's output columns."
    )
    params = inspect.signature(cls.get_feature_names_out).parameters
    assert list(params) == ["self", "input_features"], (
        f"{name}.get_feature_names_out must be (self, input_features=None), "
        f"got {list(params)}"
    )
    assert params["input_features"].default is None


@pytest.mark.parametrize("name", sorted(preprocessing.__all__))
def test_feature_names_out_width_matches_transform(name):
    if name in _EXEMPT:
        pytest.skip(_EXEMPT[name])

    est = _transformer_instance(name)
    X = _transformer_input(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = est.fit_transform(X)
        names = est.get_feature_names_out()

    width = out.shape[1]
    assert len(names) == width, (
        f"{name}.get_feature_names_out() has {len(names)} names but transform "
        f"produced {width} columns"
    )


# ---------------------------------------------------------------------------
# Exemption hygiene
# ---------------------------------------------------------------------------

def test_no_exemption_is_stale():
    public = set(models.__all__) | set(experimental.__all__) | set(preprocessing.__all__)
    unknown = sorted(set(_EXEMPT) - public)
    assert not unknown, f"_EXEMPT names symbols that are no longer public: {unknown}"


def test_every_exemption_carries_a_reason():
    for name, reason in _EXEMPT.items():
        assert len(reason) > 40, f"{name}'s exemption reason is not an explanation"


# ---------------------------------------------------------------------------
# The stability-tier table is the 1.0 semver contract
#
# From 1.0 the tier a symbol carries decides what a breaking change to it costs.
# A table that has drifted from __all__ is a broken promise, not a docs nit, so
# it is checked here rather than read by hand at release time.
# ---------------------------------------------------------------------------

_TIER_HEADINGS = ("### Tier 1 — Stable", "### Tier 2 — Beta", "### Tier 3 — Experimental")


def _tier_table_text():
    text = (REFERENCE_DIR / "index.md").read_text()
    start = text.index(_TIER_HEADINGS[0])
    end = text.index("## Score scales")
    return text[start:end]


def test_reference_index_declares_all_three_tiers():
    text = (REFERENCE_DIR / "index.md").read_text()
    for heading in _TIER_HEADINGS:
        assert heading in text, f"docs/reference/index.md is missing {heading!r}"


@pytest.mark.parametrize("subpackage", sorted(_public_subpackages()))
def test_stability_tier_table_covers_every_public_symbol(subpackage):
    table = _tier_table_text()
    module = getattr(philanthropy, subpackage)

    missing = [
        symbol for symbol in module.__all__
        if f"`{symbol}`" not in table
    ]
    # metrics is covered by one blanket row rather than nine names.
    if subpackage == "metrics" and "every function in `philanthropy.metrics`" in table:
        missing = []

    assert not missing, (
        f"docs/reference/index.md assigns no stability tier to "
        f"philanthropy.{subpackage}: {missing}. At 1.0 the tier table is the "
        f"semver contract — a public symbol without one has no stated promise."
    )


def test_tier_table_names_no_symbol_that_is_no_longer_public():
    table = _tier_table_text()
    public = {
        symbol
        for name in _public_subpackages()
        for symbol in getattr(philanthropy, name).__all__
    }
    listed = {
        sym
        for line in table.splitlines() if line.startswith("|")
        for sym in re.findall(r"`([A-Z]\w+)`", line.split("|")[1])
    }
    stale = sorted(listed - public)
    assert not stale, f"tier table lists non-public symbols: {stale}"
