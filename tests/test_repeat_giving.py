"""
tests/test_repeat_giving.py
===========================
BetaGeoModel (BG/NBD), GammaGammaModel and repeat_giving_summary.

``tests/data/cdnow_summary.csv`` is the CDNOW sample summary (2,357
customers, weeks; columns x, t_x, T, zbar) from Bruce Hardie's site, the data
both source papers fit, as redistributed by the lifetimes and BTYD packages.
"""

from __future__ import annotations

import pathlib
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from philanthropy.models import BetaGeoModel, GammaGammaModel, repeat_giving_summary

CDNOW = pd.read_csv(pathlib.Path(__file__).parent / "data" / "cdnow_summary.csv")
BG_X = CDNOW[["x", "t_x", "T"]].to_numpy(float)
GG_X = CDNOW[["x", "zbar"]].to_numpy(float)


@pytest.fixture(scope="module")
def bg():
    return BetaGeoModel().fit(BG_X)


@pytest.fixture(scope="module")
def gg():
    return GammaGammaModel().fit(GG_X)


# --- published parameters -------------------------------------------------

def test_bgnbd_recovers_published_cdnow_parameters(bg):
    # Fader, Hardie and Lee (2005), Marketing Science 24(2), section 4.
    expected = {"r": 0.243, "alpha": 4.414, "a": 0.793, "b": 2.426}
    for k, v in expected.items():
        assert bg.params_[k] == pytest.approx(v, abs=2e-3), k
    assert bg.log_likelihood_ == pytest.approx(-9582.4, abs=0.1)


def test_gamma_gamma_recovers_published_cdnow_parameters(gg):
    # Fader and Hardie (2013), "The gamma-gamma model of monetary value".
    expected = {"p": 6.25, "q": 3.74, "v": 15.44}
    for k, v in expected.items():
        assert gg.params_[k] == pytest.approx(v, abs=0.02), k


def test_bgnbd_forecast_matches_published_total(bg):
    # The paper's 39-week holdout forecast for the cohort is about 1,654
    # repeat purchases (actual: 1,788).
    assert bg.predict_gifts_forecast(BG_X, horizon=39).sum() == pytest.approx(1653.4, abs=1)


def test_recovers_simulated_bgnbd_parameters():
    rng = np.random.default_rng(0)
    r, alpha, a, b, n, T = 0.5, 2.0, 0.8, 3.0, 20_000, 10.0
    lam = rng.gamma(r, 1 / alpha, n)
    p = rng.beta(a, b, n)
    X = np.zeros((n, 3))
    X[:, 2] = T
    for i in range(n):
        t, x, last = 0.0, 0, 0.0
        while True:
            t += rng.exponential(1 / lam[i])
            if t > T:
                break
            x, last = x + 1, t
            if rng.random() < p[i]:
                break
        X[i, :2] = x, last
    fitted = BetaGeoModel().fit(X).params_
    for k, v in {"r": r, "alpha": alpha, "a": a, "b": b}.items():
        assert fitted[k] == pytest.approx(v, rel=0.15), k


# --- predictions -----------------------------------------------------------

def test_alive_score_is_a_probability_and_one_without_repeat_gifts(bg):
    score = bg.predict_alive_score(BG_X)
    assert ((score > 0) & (score <= 1)).all()
    assert (score[CDNOW["x"].to_numpy() == 0] == 1).all()
    # Same frequency, longer silence: less likely to be active.
    recent, stale = bg.predict_alive_score([[3, 30, 38], [3, 5, 38]])
    assert recent > stale


def test_gifts_forecast_grows_with_horizon_and_rejects_non_positive(bg):
    short = bg.predict_gifts_forecast(BG_X[:50], horizon=10)
    long = bg.predict_gifts_forecast(BG_X[:50], horizon=50)
    assert (long > short).all()
    with pytest.raises(ValueError, match="horizon"):
        bg.predict_gifts_forecast(BG_X[:5], horizon=0)


def test_gift_size_is_shrunk_toward_population_mean(gg):
    p, q, v = (gg.params_[k] for k in "pqv")
    pop_mean = p * v / (q - 1)
    one_gift, many_gifts = gg.predict_gift_size_forecast([[1, 200.0], [20, 200.0]])
    assert pop_mean < one_gift < many_gifts < 200
    assert gg.predict_gift_size_forecast([[0, 0.0]])[0] == pytest.approx(pop_mean)


# --- input validation ------------------------------------------------------

@pytest.mark.parametrize("X, match", [
    ([[1, 2]], "3 columns"),
    ([[-1, 0, 1]], "non-negative"),
    ([[1, 5, 4]], "recency cannot exceed T"),
])
def test_bgnbd_rejects_invalid_input(X, match):
    with pytest.raises(ValueError, match=match):
        BetaGeoModel().fit(X)


def test_gamma_gamma_rejects_invalid_input():
    with pytest.raises(ValueError, match="2 columns"):
        GammaGammaModel().fit([[1, 2, 3]])
    with pytest.raises(ValueError, match="repeat gift"):
        GammaGammaModel().fit([[0, 0.0], [0, 0.0]])


# --- estimator contract (the generated battery xfails most checks; see
# tests/test_sklearn_compliance.py) -----------------------------------------

_CASES = [
    (BetaGeoModel, BG_X, "predict_alive_score"),
    (BetaGeoModel, BG_X, "predict_gifts_forecast"),
    (GammaGammaModel, GG_X, "predict_gift_size_forecast"),
]


@pytest.mark.parametrize("cls, X, method", _CASES)
def test_estimator_contract(cls, X, method):
    with pytest.raises(NotFittedError):
        getattr(cls(), method)(X)
    est = cls(max_iter=500)
    assert est.fit(X) is est
    assert est.n_features_in_ == X.shape[1]
    assert not hasattr(clone(est), "params_")
    predict = getattr(est, method)
    restored = pickle.loads(pickle.dumps(est))
    np.testing.assert_allclose(getattr(restored, method)(X), predict(X))
    # Row order and batch split do not change a donor's score.
    perm = np.random.default_rng(0).permutation(len(X))
    np.testing.assert_allclose(predict(X[perm]), predict(X)[perm])
    np.testing.assert_allclose(predict(X[:10]), predict(X)[:10])
    with pytest.raises(ValueError):
        predict(X[:, :1])


# --- leakage contract ------------------------------------------------------

@pytest.mark.parametrize("cls, X, method", _CASES)
def test_params_are_frozen_after_fit(cls, X, method):
    est = cls().fit(X[:1000])
    frozen = dict(est.params_)
    held_out = getattr(est, method)(X[1000:])
    getattr(est, method)(X[:1000])
    np.testing.assert_array_equal(getattr(est, method)(X[1000:]), held_out)
    assert est.params_ == frozen


# --- repeat_giving_summary -------------------------------------------------

GIFTS = pd.DataFrame({
    "donor_id": ["a", "a", "a", "a", "b", "c", "c"],
    "gift_date": ["2020-01-01", "2020-01-01", "2021-01-01", "2022-01-01",
                  "2021-07-01", "2020-07-01", "2024-01-01"],
    "gift_amount": [50.0, 50.0, 300.0, 500.0, 25.0, 10.0, 999.0],
})


def test_summary_counts_repeat_gift_days_and_excludes_first_gift():
    s = repeat_giving_summary(GIFTS, as_of="2022-12-31", time_unit_days=1)
    a, b = s.loc["a"], s.loc["b"]
    # Two same-day gifts merge, so a has three gift days: two repeats.
    assert a["frequency"] == 2
    assert a["recency"] == 731 and a["T"] == 1095
    assert a["monetary_value"] == 400.0  # (300 + 500) / 2, first day excluded
    assert b["frequency"] == 0 and b["monetary_value"] == 0 and b["recency"] == 0
    # c's 2024 gift is after as_of.
    assert s.loc["c", "frequency"] == 0


def test_summary_feeds_both_models():
    s = repeat_giving_summary(GIFTS)
    assert (s["recency"] <= s["T"]).all()
    BetaGeoModel().fit(s[["frequency", "recency", "T"]])
    GammaGammaModel().fit(s[["frequency", "monetary_value"]])
