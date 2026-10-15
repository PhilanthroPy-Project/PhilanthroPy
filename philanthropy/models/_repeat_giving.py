"""
philanthropy.models._repeat_giving
==================================
Per-donor repeat-giving and gift-size models that need only a gift export
(donor id, date, amount): no labels, no wealth screening.

* :class:`BetaGeoModel` is the BG/NBD "buy till you die" model [1]_. While
  active, each donor gives at a Poisson rate drawn from a gamma(r, alpha)
  distribution; after any gift, the donor drops out for good with a
  probability drawn from a beta(a, b) distribution. Fit on each donor's
  repeat-gift count, recency and age, it scores how likely the donor is to
  still be giving and how many gifts they will make over a horizon.
* :class:`GammaGammaModel` is the gamma-gamma gift-size model [2]_: each
  donor's gifts scatter around a donor-level mean, and that mean is itself
  drawn from a population distribution, so a donor with few gifts is shrunk
  toward the file's average.

:func:`repeat_giving_summary` builds the inputs both models take from a gift
table.

Leakage safety
--------------
Both models learn only a handful of population parameters, by maximum
likelihood on the training donors in :meth:`fit`. They are frozen in
``params_``; every predict method is a closed-form function of those frozen
parameters and the row being scored, so scoring one donor cannot move another
donor's score.

References
----------
.. [1] Fader, P. S., Hardie, B. G. S. and Lee, K. L. (2005). "Counting your
   customers" the easy way: An alternative to the Pareto/NBD model.
   *Marketing Science* 24(2), 275-284.
.. [2] Fader, P. S., Hardie, B. G. S. and Lee, K. L. (2005). RFM and CLV:
   Using iso-value curves for customer base analysis. *Journal of Marketing
   Research* 42(4), 415-430. Model note: Fader and Hardie (2013), "The
   gamma-gamma model of monetary value".
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1
from sklearn.base import BaseEstimator
from sklearn.utils import Tags
from sklearn.utils.validation import check_is_fitted, validate_data

from ..preprocessing import RFMTransformer

_BG = TypeVar("_BG", bound="BetaGeoModel")
_GG = TypeVar("_GG", bound="GammaGammaModel")


def repeat_giving_summary(
    gifts: pd.DataFrame,
    *,
    as_of: Any = None,
    time_unit_days: float = 365.25,
) -> pd.DataFrame:
    """Build per-donor BG/NBD and gamma-gamma inputs from a gift table.

    Gifts on the same day are merged into one gift (amounts summed), as in
    [1]_, so a split payment does not count as a repeat gift.

    Parameters
    ----------
    gifts : pandas.DataFrame
        One row per gift with ``donor_id``, ``gift_date`` and ``gift_amount``,
        the shape every ``philanthropy.ingest`` bridge returns.
    as_of : str, datetime-like or None, default=None
        End of the observation window. Gifts after it are dropped. ``None``
        uses the latest gift date.
    time_unit_days : float, default=365.25
        Days per time unit for ``recency`` and ``T``, so the default unit is
        years and a ``horizon`` of 1 later means "the next year". Use 7 for
        weeks.

    Returns
    -------
    summary : pandas.DataFrame, indexed by ``donor_id``
        * ``frequency``: repeat gifts, i.e. distinct gift days minus one.
        * ``recency``: time from the first gift to the last.
        * ``T``: time from the first gift to the end of the window.
        * ``monetary_value``: mean amount of the repeat gifts (the first gift
          is excluded, as in [2]_); 0 when ``frequency`` is 0.

        ``summary[["frequency", "recency", "T"]]`` is the input to
        :class:`BetaGeoModel` and ``summary[["frequency", "monetary_value"]]``
        the input to :class:`GammaGammaModel`.
    """
    daily = (
        gifts.assign(gift_date=pd.to_datetime(gifts["gift_date"]).dt.normalize())
        .groupby(["donor_id", "gift_date"])["gift_amount"]
        .sum(min_count=1)
        .reset_index()
    )
    # RFMTransformer owns the as_of cut, the NaN-amount handling and the
    # recency / tenure clock; it counts each gift day once because of the
    # merge above.
    rfm = RFMTransformer(
        reference_date=as_of, agg_func="sum", include_tenure=True, as_of=as_of
    ).fit_transform(daily).set_index("donor_id")

    first_amount = (
        daily.dropna(subset=["gift_amount"])
        .sort_values("gift_date")
        .groupby("donor_id")["gift_amount"]
        .first()
        .reindex(rfm.index)
    )
    frequency = (rfm["frequency"] - 1).clip(lower=0)
    repeat_total = rfm["monetary"] - first_amount
    monetary = (repeat_total / frequency.where(frequency > 0)).fillna(0.0)
    return pd.DataFrame({
        "frequency": frequency.astype(float),
        "recency": (rfm["tenure"] - rfm["recency"]) / time_unit_days,
        "T": rfm["tenure"] / time_unit_days,
        "monetary_value": monetary,
    })


def _fit_log_params(
    neg_ll: Callable[[np.ndarray], float], n_params: int, max_iter: int
) -> tuple[np.ndarray, Any]:
    """Minimise ``neg_ll`` over log-parameters, so every parameter stays > 0."""
    def safe(log_params: np.ndarray) -> float:
        # Simplex probes far out in log space overflow; treat that as a bad
        # point rather than letting NaN steer the search.
        with np.errstate(all="ignore"):
            value = neg_ll(log_params)
        return value if np.isfinite(value) else np.inf

    result = minimize(
        safe, np.zeros(n_params), method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 1e-6, "fatol": 1e-8},
    )
    return np.exp(result.x), result


class BetaGeoModel(BaseEstimator):
    """BG/NBD model of repeat giving: who is still giving, and how often.

    Fit by maximum likelihood on one row per donor. No labels needed.

    Parameters
    ----------
    max_iter : int, default=2000
        Maximum optimiser iterations.

    Attributes
    ----------
    params_ : dict
        Fitted ``r``, ``alpha`` (gift-rate gamma shape and scale) and ``a``,
        ``b`` (drop-out beta). On the CDNOW sample the published values are
        r=0.243, alpha=4.414, a=0.793, b=2.426 [1]_.
    log_likelihood_ : float
        Total log-likelihood at ``params_``.
    n_iter_ : int
        Optimiser iterations used.
    n_features_in_ : int
        Always 3.

    Notes
    -----
    ``X`` is ``(frequency, recency, T)`` in that column order, as built by
    :func:`repeat_giving_summary`: repeat-gift count, time from first to last
    gift, and time from first gift to the end of the window, all in one time
    unit. ``recency`` must not exceed ``T``.

    Examples
    --------
    >>> import numpy as np
    >>> from philanthropy.models import BetaGeoModel
    >>> X = np.array([[0, 0, 5], [3, 4, 5], [1, 1, 5], [6, 4.8, 5]] * 10, float)
    >>> model = BetaGeoModel().fit(X)
    >>> model.predict_alive_score(X[:2]).shape
    (2,)
    """

    def __init__(self, max_iter: int = 2000) -> None:
        self.max_iter = max_iter

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.positive_only = True
        return tags

    def _check_X(self, X: Any, reset: bool) -> np.ndarray:
        X = validate_data(self, X, reset=reset, dtype=np.float64)
        if X.shape[1] != 3:
            raise ValueError(
                f"BetaGeoModel expects 3 columns (frequency, recency, T), got "
                f"{X.shape[1]} feature(s)."
            )
        if (X < 0).any():
            raise ValueError("frequency, recency and T must be non-negative.")
        if (X[:, 1] > X[:, 2]).any():
            raise ValueError("recency cannot exceed T.")
        return X

    def fit(self: _BG, X: Any, y: Any = None) -> _BG:
        """Fit r, alpha, a, b by maximum likelihood.

        Parameters
        ----------
        X : array-like of shape (n_donors, 3)
            ``(frequency, recency, T)`` per donor.
        y : ignored

        Returns
        -------
        self : BetaGeoModel
        """
        X = self._check_X(X, reset=True)
        x, t_x, T = X.T

        def neg_ll(log_params: np.ndarray) -> float:
            return -float(self._log_likelihood(np.exp(log_params), x, t_x, T).sum())

        params, result = _fit_log_params(neg_ll, 4, self.max_iter)
        self.params_ = dict(zip(("r", "alpha", "a", "b"), params))
        self.log_likelihood_ = float(-result.fun)
        self.n_iter_ = int(result.nit)
        return self

    @staticmethod
    def _log_likelihood(
        params: np.ndarray, x: np.ndarray, t_x: np.ndarray, T: np.ndarray
    ) -> np.ndarray:
        r, alpha, a, b = params
        a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
        a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
        a3 = -(r + x) * np.log(alpha + T)
        with np.errstate(divide="ignore", invalid="ignore"):
            a4 = np.where(
                x > 0,
                np.log(a) - np.log(b + x - 1) - (r + x) * np.log(alpha + t_x),
                -np.inf,
            )
        return a1 + a2 + np.logaddexp(a3, a4)

    def _dropout_odds(
        self, x: np.ndarray, t_x: np.ndarray, T: np.ndarray
    ) -> np.ndarray:
        """a/(b+x-1) * ((alpha+T)/(alpha+t_x))^(r+x) for x > 0, else 0."""
        p = self.params_
        with np.errstate(divide="ignore", invalid="ignore"):
            odds = (p["a"] / (p["b"] + x - 1)) * (
                (p["alpha"] + T) / (p["alpha"] + t_x)
            ) ** (p["r"] + x)
        return np.where(x > 0, odds, 0.0)

    def predict_alive_score(self, X: Any) -> np.ndarray:
        """P(the donor is still active) at the end of the window, in [0, 1].

        A donor with no repeat gift is always scored 1: the model only lets a
        donor drop out right after a gift.
        """
        check_is_fitted(self)
        x, t_x, T = self._check_X(X, reset=False).T
        return 1.0 / (1.0 + self._dropout_odds(x, t_x, T))

    def predict_gifts_forecast(self, X: Any, horizon: float = 1.0) -> np.ndarray:
        """Expected number of gifts each donor makes over the next ``horizon``.

        Parameters
        ----------
        X : array-like of shape (n_donors, 3)
            ``(frequency, recency, T)`` per donor.
        horizon : float, default=1.0
            Length of the forecast window, in the time unit of ``X`` (one year
            with :func:`repeat_giving_summary` defaults).

        Returns
        -------
        gifts : ndarray of shape (n_donors,)
        """
        check_is_fitted(self)
        if not horizon > 0:
            raise ValueError(f"horizon must be positive, got {horizon!r}.")
        x, t_x, T = self._check_X(X, reset=False).T
        p = self.params_
        r, alpha, a, b = p["r"], p["alpha"], p["a"], p["b"]
        hyp = hyp2f1(r + x, b + x, a + b + x - 1, horizon / (alpha + T + horizon))
        numer = (a + b + x - 1) / (a - 1) * (
            1 - ((alpha + T) / (alpha + T + horizon)) ** (r + x) * hyp
        )
        return numer / (1.0 + self._dropout_odds(x, t_x, T))


class GammaGammaModel(BaseEstimator):
    """Gamma-gamma model of gift size: each donor's expected gift amount.

    Fit by maximum likelihood on donors with at least one repeat gift; their
    mean repeat gift is shrunk toward the population mean, more so the fewer
    gifts they have. Assumes gift size is independent of how often a donor
    gives.

    Parameters
    ----------
    max_iter : int, default=2000
        Maximum optimiser iterations.

    Attributes
    ----------
    params_ : dict
        Fitted ``p``, ``q`` and ``v``. On the CDNOW sample the published
        values are p=6.25, q=3.74, v=15.44 [2]_.
    log_likelihood_ : float
        Total log-likelihood at ``params_``.
    n_iter_ : int
        Optimiser iterations used.
    n_features_in_ : int
        Always 2.

    Notes
    -----
    ``X`` is ``(frequency, monetary_value)`` in that column order, as built by
    :func:`repeat_giving_summary`. Rows with ``frequency`` 0 are ignored in
    :meth:`fit` and scored at the population mean.

    Examples
    --------
    >>> import numpy as np
    >>> from philanthropy.models import GammaGammaModel
    >>> rng = np.random.default_rng(0)
    >>> X = np.column_stack([rng.integers(1, 6, 200), rng.gamma(3, 30, 200)])
    >>> GammaGammaModel().fit(X).predict_gift_size_forecast(X[:3]).shape
    (3,)
    """

    def __init__(self, max_iter: int = 2000) -> None:
        self.max_iter = max_iter

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.positive_only = True
        return tags

    def _check_X(self, X: Any, reset: bool) -> np.ndarray:
        X = validate_data(self, X, reset=reset, dtype=np.float64)
        if X.shape[1] != 2:
            raise ValueError(
                f"GammaGammaModel expects 2 columns (frequency, "
                f"monetary_value), got {X.shape[1]} feature(s)."
            )
        if (X < 0).any():
            raise ValueError("frequency and monetary_value must be non-negative.")
        return X

    def fit(self: _GG, X: Any, y: Any = None) -> _GG:
        """Fit p, q, v by maximum likelihood.

        Parameters
        ----------
        X : array-like of shape (n_donors, 2)
            ``(frequency, monetary_value)`` per donor.
        y : ignored

        Returns
        -------
        self : GammaGammaModel
        """
        X = self._check_X(X, reset=True)
        repeat = (X[:, 0] > 0) & (X[:, 1] > 0)
        if not repeat.any():
            raise ValueError(
                "GammaGammaModel needs at least one donor with a repeat gift "
                "and a positive monetary_value."
            )
        x, m = X[repeat].T

        def neg_ll(log_params: np.ndarray) -> float:
            p, q, v = np.exp(log_params)
            px = p * x
            return -float(np.sum(
                gammaln(px + q) - gammaln(px) - gammaln(q) + q * np.log(v)
                + (px - 1) * np.log(m) + px * np.log(x) - (px + q) * np.log(v + m * x)
            ))

        params, result = _fit_log_params(neg_ll, 3, self.max_iter)
        self.params_ = dict(zip(("p", "q", "v"), params))
        self.log_likelihood_ = float(-result.fun)
        self.n_iter_ = int(result.nit)
        return self

    def predict_gift_size_forecast(self, X: Any) -> np.ndarray:
        """Expected size of each donor's future gifts.

        ``p (v + x m) / (p x + q - 1)``: a weighted average of the population
        mean ``p v / (q - 1)`` and the donor's own mean ``m``. Infinite when
        ``q <= 1``, where the population mean does not exist.
        """
        check_is_fitted(self)
        x, m = self._check_X(X, reset=False).T
        p, q, v = (self.params_[k] for k in ("p", "q", "v"))
        m = np.where(x > 0, m, 0.0)
        return p * (v + x * m) / (p * x + q - 1)
