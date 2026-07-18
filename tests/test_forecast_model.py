"""
tests/test_forecast_model.py
============================
Test suite for FinancialForecastModel (hybrid LSTM-ARIMA revenue forecaster).

Covers fit/predict shape, the multi-step revenue forecast, leakage-safety of
the frozen fitted statistics, NaN handling, and scikit-learn check_estimator
compliance.
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from philanthropy.models import FinancialForecastModel


@pytest.fixture
def revenue_Xy():
    """Feature matrix + revenue target with linear and nonlinear structure."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(150, 4))
    y = 5_000 + 800 * X[:, 0] + 300 * X[:, 1] ** 2 + rng.normal(0, 50, 150)
    return X, y


# ---------------------------------------------------------------------------
# sklearn compliance
# ---------------------------------------------------------------------------

@parametrize_with_checks(
    [FinancialForecastModel(hidden_layer_sizes=(8,), max_iter=50, random_state=0)]
)
def test_sklearn_compliance(estimator, check):
    """Run the standard sklearn check_estimator battery."""
    check(estimator)


# ---------------------------------------------------------------------------
# fit / predict
# ---------------------------------------------------------------------------

def test_fit_returns_self(revenue_Xy):
    X, y = revenue_Xy
    model = FinancialForecastModel(random_state=0)
    assert model.fit(X, y) is model


def test_predict_shape(revenue_Xy):
    X, y = revenue_Xy
    model = FinancialForecastModel(random_state=0).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (150,)


def test_predict_returns_finite_ndarray(revenue_Xy):
    X, y = revenue_Xy
    model = FinancialForecastModel(random_state=0).fit(X, y)
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert np.isfinite(preds).all()


def test_predict_single_sample(revenue_Xy):
    X, y = revenue_Xy
    model = FinancialForecastModel(random_state=0).fit(X, y)
    assert model.predict(X[:1]).shape == (1,)


def test_fitted_attributes_present(revenue_Xy):
    X, y = revenue_Xy
    model = FinancialForecastModel(random_state=0).fit(X, y)
    for attr in ("linear_model_", "fill_values_", "ar_coef_",
                 "ar_intercept_", "y_mean_", "n_features_in_"):
        assert hasattr(model, attr), attr
    assert model.n_features_in_ == 4


def test_hybrid_uses_nonlinear_component(revenue_Xy):
    """With nonlinear structure present, the residual network must be fitted
    and must change the prediction relative to the linear part alone."""
    X, y = revenue_Xy
    model = FinancialForecastModel(random_state=0).fit(X, y)
    assert model.nonlinear_model_ is not None
    linear_only = model.linear_model_.predict(model._impute(X))
    hybrid = model.predict(X)
    assert not np.allclose(linear_only, hybrid)


# ---------------------------------------------------------------------------
# predict_revenue_forecast
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("horizon", [1, 3, 12])
def test_forecast_shape(revenue_Xy, horizon):
    X, y = revenue_Xy
    model = FinancialForecastModel(random_state=0).fit(X, y)
    forecast = model.predict_revenue_forecast(X, horizon=horizon)
    assert forecast.shape == (horizon,)
    assert np.isfinite(forecast).all()


def test_forecast_flat_when_ar_order_zero(revenue_Xy):
    X, y = revenue_Xy
    model = FinancialForecastModel(ar_order=0, random_state=0).fit(X, y)
    forecast = model.predict_revenue_forecast(X, horizon=5)
    assert np.allclose(forecast, forecast[0])


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "3"])
def test_forecast_rejects_invalid_horizon(revenue_Xy, bad):
    X, y = revenue_Xy
    model = FinancialForecastModel(random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        model.predict_revenue_forecast(X, horizon=bad)


def test_forecast_before_fit_raises(revenue_Xy):
    X, _ = revenue_Xy
    model = FinancialForecastModel()
    with pytest.raises(NotFittedError):
        model.predict_revenue_forecast(X, horizon=3)


# ---------------------------------------------------------------------------
# Leakage safety — fitted statistics are frozen at fit() time
# ---------------------------------------------------------------------------

def test_fitted_stats_frozen_after_predict(revenue_Xy):
    """Predicting / forecasting on extreme held-out data must NOT mutate any
    fitted statistic (mirrors WealthScreeningImputer's leakage contract)."""
    X, y = revenue_Xy
    model = FinancialForecastModel(random_state=0).fit(X, y)

    fill_before = model.fill_values_.copy()
    ar_before = model.ar_coef_.copy()
    intercept_before = model.ar_intercept_

    X_extreme = X.copy() * 1e6
    X_extreme[0, 0] = np.nan
    model.predict(X_extreme)
    model.predict_revenue_forecast(X_extreme, horizon=6)

    assert np.allclose(model.fill_values_, fill_before)
    assert np.allclose(model.ar_coef_, ar_before)
    assert model.ar_intercept_ == intercept_before


def test_fill_values_are_train_median_only():
    """Fill values must equal the training-set median, never touched by the
    values seen at predict time."""
    X_train = np.array([[10.0], [np.nan], [30.0], [np.nan], [50.0]])
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = FinancialForecastModel(ar_order=1, random_state=0).fit(X_train, y_train)
    # median of [10, 30, 50] = 30
    assert model.fill_values_[0] == pytest.approx(30.0)

    # Predict on extreme test data, fill value must stay 30
    model.predict(np.array([[99_000.0], [np.nan]]))
    assert model.fill_values_[0] == pytest.approx(30.0)


def test_fill_values_fold_specific_in_cv():
    """Across CV folds the frozen fill values must differ — identical values
    would indicate the full dataset leaked into every fold."""
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(0)
    n = 200
    X = np.where(rng.random((n, 1)) < 0.4, np.nan, rng.lognormal(3, 1, (n, 1)))
    y = rng.normal(1000, 100, n)

    fills = []
    for train_idx, _ in KFold(n_splits=5, shuffle=True, random_state=1).split(X):
        m = FinancialForecastModel(random_state=0).fit(X[train_idx], y[train_idx])
        fills.append(m.fill_values_[0])
    assert len(set(fills)) > 1


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def test_fit_and_predict_with_nan(revenue_Xy):
    X, y = revenue_Xy
    X_nan = X.copy()
    X_nan[:20, 0] = np.nan
    X_nan[30:40, 2] = np.nan
    model = FinancialForecastModel(random_state=0).fit(X_nan, y)
    preds = model.predict(X_nan)
    assert preds.shape == (150,)
    assert np.isfinite(preds).all()


def test_all_nan_column_falls_back_to_zero():
    X = np.column_stack([np.full(20, np.nan), np.arange(20.0)])
    y = np.arange(20.0)
    model = FinancialForecastModel(ar_order=2, random_state=0).fit(X, y)
    assert model.fill_values_[0] == 0.0
    assert np.isfinite(model.predict(X)).all()


# ---------------------------------------------------------------------------
# sklearn API conventions
# ---------------------------------------------------------------------------

def test_get_set_params_round_trip():
    model = FinancialForecastModel(ar_order=5, alpha=0.01)
    params = model.get_params()
    assert params["ar_order"] == 5
    assert params["alpha"] == 0.01
    model.set_params(ar_order=2)
    assert model.ar_order == 2


def test_clone_drops_fitted_state(revenue_Xy):
    X, y = revenue_Xy
    model = FinancialForecastModel(random_state=0).fit(X, y)
    cloned = clone(model)
    assert not hasattr(cloned, "linear_model_")
    assert not hasattr(cloned, "fill_values_")


def test_not_fitted_predict_raises(revenue_Xy):
    X, _ = revenue_Xy
    with pytest.raises(NotFittedError):
        FinancialForecastModel().predict(X)


def test_reproducible_with_random_state(revenue_Xy):
    X, y = revenue_Xy
    a = FinancialForecastModel(random_state=7).fit(X, y).predict(X)
    b = FinancialForecastModel(random_state=7).fit(X, y).predict(X)
    np.testing.assert_array_equal(a, b)


def test_pipeline_compatibility(revenue_Xy):
    X, y = revenue_Xy
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", FinancialForecastModel(hidden_layer_sizes=(8,), max_iter=50,
                                         random_state=0)),
    ])
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (150,)
