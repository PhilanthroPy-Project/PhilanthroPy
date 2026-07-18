"""
philanthropy.models._forecast
==============================
Hybrid LSTM-ARIMA revenue/giving forecaster for nonprofit advancement teams.

Nonprofit and academic medical centre (AMC) advancement shops plan campaigns,
staffing, and cash flow around *forward* estimates of giving revenue.  The
research literature on hybrid time-series forecasting (Zhang, 2003, "Time
series forecasting using a hybrid ARIMA and neural network model", *Neuro-
computing*; and the more recent LSTM-ARIMA revenue-forecasting work) shows that
a giving series carries **two** distinct kinds of structure:

* a *linear* / autoregressive component — trend, fiscal-year momentum, and the
  short-memory autocorrelation that classical **ARIMA** captures well; and
* a *nonlinear* residual component — appeal-driven spikes, transformational
  mega-gifts, and macro-economic shocks that a linear model cannot express and
  that a neural network (an **LSTM** in the source papers) learns from the
  ARIMA residuals.

``FinancialForecastModel`` reproduces that additive decomposition
``y = linear(X) + nonlinear_residual(X)`` **without any heavy dependency** (no
TensorFlow, Keras, or statsmodels): the linear component is a
:class:`~sklearn.linear_model.LinearRegression` and the nonlinear residual
component is a :class:`~sklearn.neural_network.MLPRegressor` — scikit-learn's
feed-forward neural network, which stands in for the LSTM while keeping the
package pure-``scikit-learn``.  Multi-step forecasting is produced by rolling a
frozen autoregressive model of the target forward over the requested horizon.

Leakage safety
--------------
Every fitted statistic — the missing-value fill values, both sub-models, and
the autoregressive roll-forward coefficients — is computed **exclusively** from
the training data during :meth:`fit` and frozen before any forecast is
produced, mirroring the contract of
:class:`~philanthropy.preprocessing.WealthScreeningImputer`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.utils import Tags
from sklearn.utils.validation import check_is_fitted, validate_data


class FinancialForecastModel(RegressorMixin, BaseEstimator):
    """Hybrid LSTM-ARIMA forecaster for nonprofit revenue / giving series.

    ``FinancialForecastModel`` is a scikit-learn–compatible regressor that
    closes the loop with the LSTM-ARIMA hybrid forecasting literature.  It fits
    two complementary sub-models on the training data:

    * a **linear (ARIMA-surrogate) component** — a
      :class:`~sklearn.linear_model.LinearRegression` mapping the feature
      matrix to giving revenue, capturing the linear / trend structure; and
    * a **nonlinear (LSTM-surrogate) component** — a
      :class:`~sklearn.neural_network.MLPRegressor` fitted on the *residuals*
      of the linear component, capturing the nonlinear structure a linear model
      leaves behind.

    Point predictions (:meth:`predict`) are the additive hybrid
    ``linear(X) + nonlinear_residual(X)``.  Forward-looking, multi-period
    forecasts (:meth:`predict_revenue_forecast`) are produced by seeding a
    frozen autoregressive model with the most recent hybrid predictions and
    rolling it forward over the requested ``horizon``.

    The model handles missing values natively: at :meth:`fit` time it freezes a
    per-column median fill (falling back to ``0.0`` for all-``NaN`` columns) and
    applies it before either sub-model sees the data, so no upstream imputer is
    required and no test-set statistic can leak backwards into training.

    Parameters
    ----------
    ar_order : int, default=3
        Order ``p`` of the autoregressive roll-forward used by
        :meth:`predict_revenue_forecast`.  Each future period is predicted from
        the previous ``ar_order`` periods.  ``0`` disables the autoregressive
        dynamics and produces a flat forecast at the last observed level.
    hidden_layer_sizes : tuple of int, default=(64,)
        Hidden-layer architecture of the nonlinear residual network, passed
        straight through to :class:`~sklearn.neural_network.MLPRegressor`.
        This is the LSTM stand-in; widen or deepen it for more expressive
        nonlinear structure at the cost of training time.
    max_iter : int, default=300
        Maximum optimisation iterations for the residual network.
    alpha : float, default=1e-4
        L2 regularisation strength of the residual network.  Increase to combat
        overfitting on short giving histories.
    random_state : int or None, default=None
        Seed for the residual network's weight initialisation.  Pass an integer
        for fully reproducible forecasts suitable for board-level audit trails.

    Attributes
    ----------
    linear_model_ : LinearRegression
        The fitted linear (ARIMA-surrogate) component.
    nonlinear_model_ : MLPRegressor or None
        The fitted nonlinear (LSTM-surrogate) residual component.  ``None`` when
        the residuals are degenerate (fewer than two samples or zero variance),
        in which case predictions fall back to the linear component alone.
    fill_values_ : ndarray of shape (n_features_in_,)
        Per-column median fill values frozen at :meth:`fit` time.
    ar_coef_ : ndarray of shape (ar_order,)
        Frozen autoregressive coefficients used for the forecast roll-forward.
    ar_intercept_ : float
        Frozen autoregressive intercept.
    y_mean_ : float
        Mean of the training target, used to pad short forecast seeds.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> from philanthropy.models import FinancialForecastModel
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(120, 4))
    >>> # revenue with linear + mild nonlinear structure
    >>> y = 5_000 + 800 * X[:, 0] + 300 * X[:, 1] ** 2 + rng.normal(0, 50, 120)
    >>> model = FinancialForecastModel(random_state=0).fit(X, y)
    >>> preds = model.predict(X)
    >>> preds.shape
    (120,)
    >>> forecast = model.predict_revenue_forecast(X, horizon=4)
    >>> forecast.shape
    (4,)

    See Also
    --------
    philanthropy.models.ShareOfWalletRegressor :
        Cross-sectional capacity regressor; pair with this forecaster to move
        from per-donor capacity to portfolio-level revenue projections.
    philanthropy.preprocessing.WealthScreeningImputer :
        The leakage-safe fill contract this model mirrors internally.

    References
    ----------
    .. [1] Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA
       and neural network model. *Neurocomputing*, 50, 159-175.
    """

    def __init__(
        self,
        ar_order: int = 3,
        hidden_layer_sizes: Tuple[int, ...] = (64,),
        max_iter: int = 300,
        alpha: float = 1e-4,
        random_state: Optional[int] = None,
    ) -> None:
        # scikit-learn rule: __init__ stores parameters and does NO logic.
        self.ar_order = ar_order
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.alpha = alpha
        self.random_state = random_state

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.regressor_tags.poor_score = True
        return tags

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _impute(self, X) -> np.ndarray:
        """Fill NaNs with the frozen per-column training medians.

        Casting to ``float64`` guarantees ``np.isnan`` works on integer inputs
        and never mutates the caller's array.
        """
        X = np.asarray(X, dtype=np.float64)
        mask = np.isnan(X)
        if mask.any():
            rows, cols = np.where(mask)
            X = X.copy()
            X[rows, cols] = np.take(self.fill_values_, cols)
        return X

    def _fit_autoregressive(self, y: np.ndarray) -> None:
        """Fit and freeze an AR(``ar_order``) model on the training target."""
        p = int(self.ar_order)
        n = y.shape[0]
        self.y_mean_ = float(np.mean(y)) if n else 0.0

        if p <= 0 or n <= p:
            # Not enough history for the requested order: degenerate to a flat
            # forecast at the series mean.
            self.ar_coef_ = np.zeros(max(p, 0), dtype=float)
            self.ar_intercept_ = self.y_mean_
            return

        rows = n - p
        # Column k (1-indexed) holds y[t-k]; row t ranges over [p, n).
        lags = np.column_stack([y[p - k : n - k] for k in range(1, p + 1)])
        target = y[p:]
        design = np.column_stack([np.ones(rows), lags])
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        self.ar_intercept_ = float(coef[0])
        self.ar_coef_ = coef[1:]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y) -> "FinancialForecastModel":
        """Fit the hybrid forecaster on labelled revenue data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix describing each period (e.g. fiscal-year index,
            appeal counts, prior-period giving, macro indicators).  May contain
            ``NaN``; missing values are filled with frozen training medians.
        y : array-like of shape (n_samples,)
            Giving revenue for each period.

        Returns
        -------
        self : FinancialForecastModel
            Fitted estimator (enables method chaining).
        """
        X, y = validate_data(self, X, y, ensure_all_finite="allow-nan", reset=True)
        self.n_features_in_ = X.shape[1]

        # Freeze leakage-safe fill values (per-column training median; all-NaN
        # columns fall back to 0.0) before either sub-model sees the data.
        Xf = np.asarray(X, dtype=np.float64)
        with np.errstate(all="ignore"):
            medians = np.nanmedian(Xf, axis=0)
        self.fill_values_ = np.where(np.isnan(medians), 0.0, medians)
        X_imp = self._impute(Xf)

        # Linear (ARIMA-surrogate) component.
        self.linear_model_ = LinearRegression()
        self.linear_model_.fit(X_imp, y)
        residuals = y - self.linear_model_.predict(X_imp)

        # Nonlinear (LSTM-surrogate) residual component.  A neural network is
        # only meaningful with >1 sample and non-constant residuals; otherwise
        # fall back to the linear component alone.
        self.nonlinear_model_ = None
        if X_imp.shape[0] > 1 and float(residuals.max() - residuals.min()) > 0.0:
            nn = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                alpha=self.alpha,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            nn.fit(X_imp, residuals)
            self.nonlinear_model_ = nn

        # Freeze autoregressive roll-forward coefficients from the training
        # target only (used by predict_revenue_forecast).
        self._fit_autoregressive(np.asarray(y, dtype=float))

        # Iterations run by the residual network (1 when it was skipped).
        self.n_iter_ = (
            self.nonlinear_model_.n_iter_
            if self.nonlinear_model_ is not None
            else 1
        )
        return self

    def predict(self, X) -> np.ndarray:
        """Predict revenue for each period in ``X`` (cross-sectional hybrid).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix with the same number of columns as seen at
            :meth:`fit`.  May contain ``NaN``.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Additive hybrid predictions ``linear(X) + nonlinear_residual(X)``.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
        X_imp = self._impute(X)
        out = self.linear_model_.predict(X_imp)
        if self.nonlinear_model_ is not None:
            out = out + self.nonlinear_model_.predict(X_imp)
        return out

    def predict_revenue_forecast(self, X, horizon: int) -> np.ndarray:
        """Forecast giving revenue for the next ``horizon`` periods.

        The supplied ``X`` provides the most recent observed context: its hybrid
        predictions seed a frozen autoregressive roll-forward that projects
        ``horizon`` periods into the future.  Because the autoregressive
        coefficients are frozen at :meth:`fit` time, no information from ``X``
        can contaminate the learned dynamics.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix for the most recent periods, ordered oldest to
            newest.  May contain ``NaN``.
        horizon : int
            Number of future periods to forecast.  Must be a positive integer.

        Returns
        -------
        forecast : ndarray of shape (horizon,)
            Forecasted revenue for each of the next ``horizon`` periods.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        ValueError
            If ``horizon`` is not a positive integer.
        """
        check_is_fitted(self)
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
            raise ValueError(
                f"`horizon` must be a positive integer, got {horizon!r}."
            )
        if horizon < 1:
            raise ValueError(f"`horizon` must be >= 1, got {horizon}.")

        history = np.asarray(self.predict(X), dtype=float)
        p = int(self.ar_order)

        if p <= 0:
            level = float(history[-1]) if history.size else self.y_mean_
            return np.full(horizon, level)

        # Seed most-recent-first: [y[t-1], y[t-2], ...]; pad short history with
        # the frozen training mean.
        if history.size >= p:
            window = list(history[-p:][::-1])
        else:
            window = list(history[::-1]) + [self.y_mean_] * (p - history.size)

        forecasts = []
        for _ in range(horizon):
            nxt = self.ar_intercept_ + float(np.dot(self.ar_coef_, window))
            forecasts.append(nxt)
            window = [nxt] + window[:-1]
        return np.asarray(forecasts, dtype=float)
