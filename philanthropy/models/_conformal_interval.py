"""
philanthropy.models._conformal_interval
=======================================
Distribution-free intervals on a dollar amount.

A gift officer handed an ask amount gets one number. The number is wrong, and
the useful question is by how much: an ask of $25,000 that could plausibly have
been $18,000–$34,000 is a different conversation from one that could plausibly
have been $600–$900,000. Nothing else in this package answers that question.
:class:`~philanthropy.models.AskAmountRecommender`,
:class:`~philanthropy.models.ShareOfWalletRegressor` and
:class:`~philanthropy.models.FinancialForecastModel` are all regressors and all
return a point.

``GiftIntervalCalibrator`` wraps one of them, already fitted, and turns held-out
rows into an interval. The construction is split conformal prediction: rank the
held-out conformity scores, take one order statistic, and the rank itself is the
coverage guarantee. It holds in finite samples, for any regressor, under
exchangeability alone.

Five things a default implementation gets wrong
----------------------------------------------
**It returns an interval it cannot certify.** One order statistic needs
``n >= 1/alpha - 1`` calibration rows: 19 at the 95 % level. Below that the
required rank exceeds the largest score there is, and the honest answer is a
refusal. :meth:`GiftIntervalCalibrator.fit` raises. There is no flag to switch
the check off.

**It computes that floor in floating point.** The floor is a ceiling, so
``int(1 / alpha - 1)`` truncates: at ``alpha = 0.07`` it reports 13 where the
floor is 14, and 13 rows cannot certify the level. Every rank here comes from
:class:`fractions.Fraction`. A ``float`` alpha is read as the decimal it prints
as, so ``0.05`` means one twentieth. An alpha with no finite decimal form has to
be spelled exactly, as ``Fraction(1, 3)``; ``Fraction(1 / 3)`` and
``Fraction(str(1 / 3))`` are both the binary approximation, both strictly above
one third, and both move the floor.

**It reports the level that was asked for.** The attained level is
``r / (n + 1)``, and only lands on the request when ``1 / alpha - 1`` divides
``n + 1``. Ask for 0.95 with 30 calibration rows and the interval covers at
0.9677; with 20 rows, 0.9524. Consumers downstream of a service boundary see
what the library resolved, not what they typed, so
:class:`GiftInterval` carries ``attained_level`` as a field.

**It ignores the support.** A gift cannot be negative. ``y >= 0`` and
``y in [l, u]`` together give ``y in [max(l, 0), u]``, so intersecting with the
support leaves coverage bit-identical and strictly narrows the interval. Free
width, no assumption.

**It calibrates on the whole file.** A pooled calibration set is dominated by
whichever segment supplies most of the rows, and the segments it does not
represent are under-covered however much marginal data you add. Pass ``groups``
and each segment is calibrated on its own rows; a segment below the per-group
floor is refused rather than quietly pooled with a segment at a different
capacity level.

Choosing the calibration rows
-----------------------------
Calibration rows must be held out of the regressor's training data and
exchangeable with the rows being scored, the same contract
:func:`~philanthropy.metrics.conformal_pvalue` states for its calibration
scores. On a donor panel that means splitting by donor, not by row: a donor with
rows on both sides of the split carries their own capacity level across it, and
every coverage number you measure is flattered.
:class:`~philanthropy.model_selection.FiscalYearGroupedSplitter` with
``drop_repeat_donors=True`` does that carve. ``tests/test_leakage.py`` measures
the difference.

Examples
--------
>>> import numpy as np
>>> from sklearn.ensemble import HistGradientBoostingRegressor
>>> from philanthropy.models import GiftIntervalCalibrator
>>> rng = np.random.default_rng(0)
>>> X = rng.uniform(0, 1, (200, 3))
>>> y = 5_000 + 20_000 * X[:, 0] + rng.normal(0, 1_500, 200)
>>> model = HistGradientBoostingRegressor(max_iter=30, random_state=0)
>>> _ = model.fit(X[:120], y[:120])
>>> cal = GiftIntervalCalibrator(model, alpha=0.05).fit(X[120:], y[120:])
>>> cal.n_calibration_
80
>>> round(cal.attained_level_, 4)
0.9506

The requested level was 0.95 and the interval covers at 0.9506, because 80
calibration rows put the rank at 77 and ``77 / 81`` is what one order statistic
can deliver:

>>> cal.rank_
77
>>> interval = cal.predict_gift_interval(X[:5])
>>> bool((interval.lower <= interval.upper).all())
True
>>> bool((interval.lower >= 0).all())
True

See Also
--------
philanthropy.metrics.interval_report :
    Coverage, interval score and the width-to-target ratio for a set of
    intervals.
philanthropy.metrics.conformal_pvalue :
    The selection-side construction: p-values that bound a selection rate on a
    donor score rather than bounding a dollar amount.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted, validate_data

_SCORES = ("absolute", "difficulty", "log")


@dataclass(frozen=True)
class GiftInterval:
    """A calibrated interval on a dollar amount, with the level it attained.

    Attributes
    ----------
    lower, upper : ndarray of shape (n_samples,)
        The interval, in the target's units. ``lower`` is intersected with the
        support unless ``lower_bound=None`` was passed to the calibrator.
    rank : ndarray of int, shape (n_samples,)
        The order statistic ``r`` used for each row. Constant unless the
        calibrator was fitted with ``groups``, in which case it varies with the
        group's calibration size.
    attained_level : ndarray of shape (n_samples,)
        ``r / (n + 1)``: the coverage this interval actually certifies. Read
        this, not ``requested_level``.
    requested_level : float
        ``1 - alpha``, what the caller asked for.
    """

    lower: np.ndarray
    upper: np.ndarray
    rank: np.ndarray
    attained_level: np.ndarray
    requested_level: float


def _exact_alpha(alpha) -> Fraction:
    """Read ``alpha`` as an exact rational.

    A ``float`` is read as the shortest decimal that round-trips to it, so
    ``0.05`` is one twentieth rather than the binary value just above it. A
    ``Fraction`` is taken as given: ``Fraction(1, 3)`` is one third, and
    ``Fraction(1 / 3)`` is the double, which is a different number.
    """
    if isinstance(alpha, Fraction):
        return alpha
    if isinstance(alpha, Decimal):
        return Fraction(alpha)
    if isinstance(alpha, (int, np.integer)) and not isinstance(alpha, bool):
        return Fraction(int(alpha))
    if isinstance(alpha, (float, np.floating)):
        return Fraction(str(float(alpha)))
    raise TypeError(
        "alpha must be a float, int, Decimal or Fraction, got "
        f"{type(alpha).__name__}."
    )


def _min_calibration_size(alpha) -> int:
    """Smallest calibration set that certifies ``1 - alpha`` from one rank.

    ``ceil((n + 1) * (1 - alpha)) <= n`` reduces to ``n >= 1 / alpha - 1``, so
    the floor is ``ceil(1 / alpha) - 1``. It is a ceiling: ``int(1 / alpha - 1)``
    truncates and reports one row too few whenever ``1 / alpha`` is not an
    integer.
    """
    return math.ceil(1 / _exact_alpha(alpha)) - 1


def _rank(n: int, alpha: Fraction) -> int:
    """The order statistic that carries the guarantee at ``n`` rows."""
    return math.ceil((n + 1) * (1 - alpha))


def _calibrate(scores: np.ndarray, alpha: Fraction, where: str = "") -> tuple:
    """Return ``(quantile, rank)``, or refuse below the floor."""
    n = int(scores.size)
    floor = _min_calibration_size(alpha)
    if n < floor:
        raise ValueError(
            f"{where}{n} calibration row(s) cannot certify a "
            f"{float(1 - alpha):.6g} interval from one order statistic. The "
            f"floor is {floor} (n >= 1/alpha - 1, in exact rational "
            "arithmetic): below it the required rank exceeds the largest "
            "calibration score there is. Hold out more rows, or ask for a "
            "lower level."
        )
    r = _rank(n, alpha)
    return float(np.sort(scores)[r - 1]), r


def _key(value):
    """Hashable Python key for a group label that may be a numpy scalar."""
    return value.item() if isinstance(value, np.generic) else value


class GiftIntervalCalibrator(RegressorMixin, BaseEstimator):
    """Turn a fitted dollar-valued regressor into calibrated intervals.

    Split conformal prediction over one order statistic. ``fit`` calibrates on
    held-out rows and never touches ``estimator``, which must already be fitted;
    :meth:`predict` forwards to it unchanged, so the point prediction a gift
    officer sees does not move when an interval is added around it.

    Parameters
    ----------
    estimator : object
        A **fitted** regressor whose ``predict`` returns a dollar amount. Fitting
        it on the rows passed to :meth:`fit` would make the conformity scores
        in-sample and void the guarantee, so this class checks and refuses.
    alpha : float, Fraction, Decimal or int, default=0.05
        Miscoverage. The interval targets ``1 - alpha`` and reports what it
        attains. Read exactly: ``0.05`` means one twentieth, and an alpha with
        no finite decimal form must be passed as ``Fraction(1, 3)`` rather than
        ``1 / 3``.
    score : {"absolute", "difficulty", "log"}, default="absolute"
        The conformity score, all three of them one-rank:

        ``"absolute"``
            ``|y - yhat|``. Constant width, and the width every other score is
            judged against.
        ``"difficulty"``
            ``|y - yhat| / sigma(X)``, with ``sigma`` from
            ``difficulty_estimator``. Width scales with the difficulty estimate,
            so a well-understood annual donor gets a narrower interval than an
            unscreened prospect.
        ``"log"``
            ``|log1p(y) - log1p(yhat)|``, inverted back to dollars. Width scales
            with the amount, which is the right shape for a right-skewed gift
            distribution. ``log1p`` rather than ``log`` so a $0 outcome stays in
            the domain.

        Equal-tailed two-rank intervals are deliberately absent: two order
        statistics at ``alpha / 2`` double the floor (39 rows at the 95 % level,
        against 19) and buy nothing these three do not.
    difficulty_estimator : object or callable, default=None
        Required when ``score="difficulty"``. Either an object with ``predict``
        or a callable, mapping ``X`` to strictly positive scale estimates. Fit it
        on the regressor's training rows, not on the calibration rows.
    lower_bound : float or None, default=0.0
        Intersect the interval with ``[lower_bound, inf)``. A gift amount cannot
        be negative, and clipping to a bound the target respects leaves coverage
        unchanged while strictly narrowing width. ``None`` disables it, for a
        target that can legitimately go negative (a net change, a refund-adjusted
        total). Calibration targets below the bound raise, because they are
        evidence the bound is wrong.

    Attributes
    ----------
    quantile_ : float or dict
        The calibrated score at rank ``rank_``. A ``dict`` keyed by group label
        when ``fit`` received ``groups``.
    rank_ : int or dict
        ``r = ceil((n + 1) * (1 - alpha))``.
    attained_level_ : float or dict
        ``r / (n + 1)``. Not ``1 - alpha``; see the module docstring.
    n_calibration_ : int or dict
        Calibration rows used.
    requested_level_ : float
        ``1 - alpha``.
    groups_ : ndarray or None
        The distinct group labels calibrated for, or ``None`` when pooled.
    n_features_in_ : int
        Features seen during :meth:`fit`.

    Raises
    ------
    sklearn.exceptions.NotFittedError
        If ``estimator`` is not fitted when :meth:`fit` is called.
    ValueError
        If the calibration set, or any group in it, is below the floor for the
        requested level; if ``score="difficulty"`` without a positive
        ``difficulty_estimator``; if a calibration target falls below
        ``lower_bound``; or if a group at predict time was not calibrated for.

    Examples
    --------
    **Pooled calibration:**

    >>> import numpy as np
    >>> from philanthropy.models import AskAmountRecommender, GiftIntervalCalibrator
    >>> rng = np.random.default_rng(3)
    >>> X = rng.uniform(0, 1, (240, 4))
    >>> y = 20_000 + 40_000 * X[:, 0] + rng.normal(0, 3_000, 240)
    >>> ask = AskAmountRecommender(max_iter=40, random_state=3).fit(X[:140], y[:140])
    >>> cal = GiftIntervalCalibrator(ask, alpha=0.1).fit(X[140:200], y[140:200])
    >>> cal.rank_, cal.n_calibration_
    (55, 60)
    >>> round(cal.attained_level_, 4)
    0.9016

    **Per-segment calibration.** Segments are calibrated independently, so the
    rank and the attained level differ between them:

    >>> seg = np.where(X[140:200, 1] > 0.5, "principal", "annual")
    >>> cal = GiftIntervalCalibrator(ask, alpha=0.1).fit(X[140:200], y[140:200], groups=seg)
    >>> sorted(cal.n_calibration_.items())
    [('annual', 33), ('principal', 27)]
    >>> interval = cal.predict_gift_interval(X[200:], groups=np.where(
    ...     X[200:, 1] > 0.5, "principal", "annual"))
    >>> bool((interval.upper >= interval.lower).all())
    True
    >>> sorted(set(interval.rank.tolist()))
    [26, 31]

    Notes
    -----
    **Why this is not a ``check_estimator``-compliant estimator.** The battery
    clones with default parameters and calls ``fit(X, y)``. This class cannot
    satisfy that: calibrating and training on the same rows is the one thing it
    exists to prevent. It is exempted in ``tests/test_sklearn_compliance.py``
    with that reason, rather than gaining a parameter that carves training rows
    out of the calibration set to satisfy a test.

    **What exchangeability buys and what it does not.** The guarantee is
    marginal over the calibration draw: coverage holds on average across
    calibration sets, not conditionally on a donor's features. Per-group
    calibration is the practical answer to that gap, and is why ``groups``
    exists.

    See Also
    --------
    philanthropy.metrics.interval_report :
        Whether the interval is narrow enough to act on.
    """

    # Scalar when calibration is pooled, dict keyed by group label when it is
    # not; ``groups_`` says which. Annotated loosely because the shape is a
    # function of how ``fit`` was called, not of the class.
    quantile_: Any
    rank_: Any
    attained_level_: Any
    n_calibration_: Any

    def __init__(
        self,
        estimator,
        alpha: Union[float, Fraction, Decimal, int] = 0.05,
        score: str = "absolute",
        difficulty_estimator=None,
        lower_bound: Optional[float] = 0.0,
    ) -> None:
        # scikit-learn rule: __init__ stores parameters and does NO logic.
        self.estimator = estimator
        self.alpha = alpha
        self.score = score
        self.difficulty_estimator = difficulty_estimator
        self.lower_bound = lower_bound

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.regressor_tags.poor_score = True
        return tags

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y, groups=None) -> "GiftIntervalCalibrator":
        """Calibrate on held-out rows.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Calibration features. Held out of ``estimator``'s training data and
            exchangeable with the rows to be scored. May contain ``NaN`` if
            ``estimator`` accepts it.
        y : array-like of shape (n_samples,)
            Realised dollar amounts for those rows.
        groups : array-like of shape (n_samples,), optional
            Segment label to calibrate within: capacity tier, sector, giving
            society, business unit. **Not** the donor identifier and not the
            fiscal year. Every distinct label is calibrated on its own rows and
            must clear the floor on its own; passing donor ids here gives one
            row per group and is refused.

        Returns
        -------
        self : GiftIntervalCalibrator
        """
        alpha = _exact_alpha(self.alpha)
        if not 0 < alpha < 1:
            raise ValueError(
                f"alpha must satisfy 0 < alpha < 1, got {self.alpha!r}."
            )
        if self.score not in _SCORES:
            raise ValueError(
                f"score must be one of {_SCORES}, got {self.score!r}."
            )
        self._check_prefit()

        _, y_valid = validate_data(
            self, X, y, ensure_all_finite="allow-nan", reset=True
        )
        y_cal = np.asarray(y_valid, dtype=float).ravel()

        if self.lower_bound is not None:
            below = int(np.count_nonzero(y_cal < self.lower_bound))
            if below:
                raise ValueError(
                    f"{below} calibration target(s) fall below "
                    f"lower_bound={self.lower_bound!r}. Clipping the interval "
                    "to a bound the target does not respect changes coverage "
                    "instead of leaving it alone. Pass lower_bound=None if the "
                    "target can legitimately go below it."
                )

        yhat = self._point(X, y_cal.size)
        scores, _ = self._conformity_scores(X, yhat, y_cal)

        self.requested_level_ = float(1 - alpha)
        self._alpha_ = alpha

        if groups is None:
            self.groups_ = None
            self.quantile_, self.rank_ = _calibrate(scores, alpha)
            self.n_calibration_ = int(scores.size)
            self.attained_level_ = float(Fraction(self.rank_, scores.size + 1))
            return self

        labels = self._group_labels(groups, y_cal.size)
        keys, counts = np.unique(labels, return_counts=True)
        floor = _min_calibration_size(alpha)
        short = [(_key(k), int(c)) for k, c in zip(keys, counts) if c < floor]
        if short:
            raise ValueError(
                "per-group calibration will not pool a group that cannot "
                f"certify a {self.requested_level_:.6g} interval on its own "
                f"({floor} row(s) needed per group). Short: "
                + ", ".join(f"{k!r} ({c} row(s))" for k, c in short)
                + ". Pooling them with a larger group calibrates them at "
                "another segment's capacity level, which is the failure this "
                "argument exists to avoid: drop the segment, merge it into "
                "another deliberately, or ask for a lower level."
            )

        self.groups_ = keys
        self.quantile_, self.rank_ = {}, {}
        self.n_calibration_, self.attained_level_ = {}, {}
        for k in keys:
            in_group = labels == k
            q, r = _calibrate(scores[in_group], alpha, where=f"group {_key(k)!r}: ")
            n_g = int(in_group.sum())
            self.quantile_[_key(k)] = q
            self.rank_[_key(k)] = r
            self.n_calibration_[_key(k)] = n_g
            self.attained_level_[_key(k)] = float(Fraction(r, n_g + 1))
        return self

    def predict(self, X) -> np.ndarray:
        """Return ``estimator``'s point prediction, unchanged."""
        check_is_fitted(self, ["quantile_"])
        validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
        return self._point(X, None)

    def predict_gift_interval(self, X, groups=None) -> GiftInterval:
        """Return a calibrated interval on the dollar amount for each row.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features for the rows to be scored.
        groups : array-like of shape (n_samples,), optional
            Required, and only accepted, when :meth:`fit` received ``groups``.
            A label that was not calibrated for raises rather than falling back
            to a pooled quantile.

        Returns
        -------
        GiftInterval
            ``lower``, ``upper``, and the rank and attained level behind them.
        """
        check_is_fitted(self, ["quantile_"])
        validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
        n = _n_rows(X)
        yhat = self._point(X, n)

        if self.groups_ is None:
            if groups is not None:
                raise ValueError(
                    "this calibrator was fitted without groups, so a pooled "
                    "quantile is all it has; passing groups here would imply a "
                    "per-segment guarantee it cannot make. Refit with groups."
                )
            q = np.full(n, self.quantile_, dtype=float)
            rank = np.full(n, self.rank_, dtype=int)
            level = np.full(n, self.attained_level_, dtype=float)
        else:
            if groups is None:
                raise ValueError(
                    "this calibrator was fitted with groups, so every row needs "
                    "the segment it belongs to. Pass groups= to "
                    "predict_gift_interval."
                )
            labels = self._group_labels(groups, n)
            unseen = {_key(v) for v in labels} - set(self.quantile_)
            if unseen:
                raise ValueError(
                    "no calibration rows for group(s) "
                    f"{sorted(unseen, key=repr)}; the calibrated segments are "
                    f"{sorted(self.quantile_, key=repr)}. A segment with no "
                    "calibration data has no certified interval, and borrowing "
                    "another segment's quantile is the pooling this argument "
                    "exists to prevent."
                )
            keys = [_key(v) for v in labels]
            q = np.array([self.quantile_[k] for k in keys], dtype=float)
            rank = np.array([self.rank_[k] for k in keys], dtype=int)
            level = np.array([self.attained_level_[k] for k in keys], dtype=float)

        lower, upper = self._bounds(X, yhat, q)
        if self.lower_bound is not None:
            lower = np.maximum(lower, float(self.lower_bound))
        return GiftInterval(
            lower=lower,
            upper=upper,
            rank=rank,
            attained_level=level,
            requested_level=self.requested_level_,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_prefit(self) -> None:
        try:
            check_is_fitted(self.estimator)
        except NotFittedError:
            raise NotFittedError(
                "GiftIntervalCalibrator calibrates a regressor that is already "
                "fitted: fit `estimator` on training rows first, then pass "
                "held-out rows to this fit. Fitting both on the same rows would "
                "make every conformity score in-sample and the coverage "
                "guarantee void."
            ) from None
        if not hasattr(self.estimator, "predict"):
            raise TypeError(
                "estimator must have a predict method returning a dollar "
                f"amount; {type(self.estimator).__name__} does not."
            )

    def _point(self, X, n: Optional[int]) -> np.ndarray:
        """Delegate to ``estimator``, passing X through untouched.

        ``validate_data`` has already checked the width; the original object
        goes to the inner estimator so a DataFrame keeps its column names.
        """
        yhat = np.asarray(self.estimator.predict(X), dtype=float).ravel()
        if n is not None and yhat.shape != (n,):
            raise ValueError(
                f"estimator.predict returned {yhat.shape} for {n} row(s); "
                "GiftIntervalCalibrator wraps per-row regressors only. "
                "FinancialForecastModel.predict_revenue_forecast, for instance, "
                "returns one value per horizon step rather than per row."
            )
        return yhat

    def _difficulty(self, X, n: int) -> np.ndarray:
        est = self.difficulty_estimator
        if est is None:
            raise ValueError(
                'score="difficulty" needs a difficulty_estimator: an object '
                "with predict, or a callable, mapping X to a strictly positive "
                "scale. Fit it on the regressor's training rows."
            )
        raw = est.predict(X) if hasattr(est, "predict") else est(X)
        sigma = np.asarray(raw, dtype=float).ravel()
        if sigma.shape != (n,):
            raise ValueError(
                f"difficulty_estimator returned {sigma.shape} for {n} row(s); "
                "one positive scale per row is required."
            )
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError(
                "difficulty_estimator must return strictly positive finite "
                "scales: a zero divides, a negative one inverts the interval. "
                f"Got min {np.nanmin(sigma)!r}."
            )
        return sigma

    def _conformity_scores(self, X, yhat: np.ndarray, y: np.ndarray) -> tuple:
        if self.score == "absolute":
            return np.abs(y - yhat), None
        if self.score == "difficulty":
            sigma = self._difficulty(X, y.size)
            return np.abs(y - yhat) / sigma, sigma
        self._check_log_domain(yhat, y)
        return np.abs(np.log1p(y) - np.log1p(yhat)), None

    def _check_log_domain(self, yhat: np.ndarray, y: Optional[np.ndarray]) -> None:
        bad = np.any(yhat < 0) or (y is not None and np.any(y < 0))
        if bad:
            raise ValueError(
                'score="log" works on log1p dollars, so it needs the point '
                "predictions and the targets to be non-negative. Clip the "
                "regressor (AskAmountRecommender has ask_floor), or use "
                'score="absolute" for a target that can go negative.'
            )

    def _bounds(self, X, yhat: np.ndarray, q: np.ndarray) -> tuple:
        if self.score == "log":
            self._check_log_domain(yhat, None)
            base = np.log1p(yhat)
            return np.expm1(base - q), np.expm1(base + q)
        half = q * self._difficulty(X, yhat.size) if self.score == "difficulty" else q
        return yhat - half, yhat + half

    @staticmethod
    def _group_labels(groups, n: int) -> np.ndarray:
        labels = np.asarray(groups)
        if labels.ndim != 1:
            raise ValueError(
                "groups must be one-dimensional: one segment label per row, "
                f"got shape {labels.shape}."
            )
        if labels.size != n:
            raise ValueError(
                f"groups has {labels.size} label(s) for {n} row(s)."
            )
        return labels


def _n_rows(X) -> int:
    try:
        return int(X.shape[0])
    except AttributeError:
        return len(X)
