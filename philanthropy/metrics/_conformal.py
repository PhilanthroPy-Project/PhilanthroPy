"""
philanthropy.metrics._conformal
================================
Distribution-free p-values for donor scores.

A donor score is consumed as a threshold crossing: a solicitation fires when
the score clears a cut point. Picking that cut point from a calibrated
probability ("contact everyone above 0.8") fixes no error rate at all. The
split-conformal p-value fixes one: rank a scored donor against a held-out
calibration set and the result is super-uniform under exchangeability, so
thresholding it at ``alpha`` bounds the expected **selection rate** at ``alpha``
in finite samples, with no distributional assumption.

Selection rate, not false-positive rate. The bound is on the fraction of
*exchangeable* donors the threshold picks. Reading it as a false-positive rate
requires the calibration set to contain only nulls (donors who will not give),
which is the outlier-detection construction of Bates et al. (2023); a
calibration set drawn from a mixed held-out population does not give you that
reading, and this function cannot tell which one you passed.

Only the non-smoothed form is implemented, eq. (3) of Bates et al. (2023):

    p = (1 + |{i : s_i >= s}|) / (n + 1)

The ``1 +`` in the numerator and the ``+ 1`` in the denominator are the test
point itself. Both are load-bearing. Dropping either one produces a statistic
that is not a valid p-value, and dropping only the numerator's can return a
value above 1.

Evaluating intervals
--------------------
:func:`interval_score` and :func:`interval_report` grade the intervals
:class:`~philanthropy.models.GiftIntervalCalibrator` produces. Coverage on its
own is not enough to grade them: ``[0, inf)`` covers everything and says
nothing. Two more numbers decide whether an interval is worth reading.

The interval score ``(u - l) + (2/alpha)(l - y)+ + (2/alpha)(y - u)+`` is proper
for a central interval, so it cannot be gamed by widening. On a right-skewed
gift distribution it is also a mean of a heavy-tailed loss that a handful of
donors can carry, which is why :func:`interval_report` puts a median and a
trimmed mean beside the mean rather than reporting the mean alone.

Width relative to the target is the other one. A median width 39 times the
median gift is a valid interval and a useless one, and the ratio is what
separates the two. Both live on :class:`IntervalReport`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Collection, Tuple

import numpy as np


def conformal_pvalue(calibration_scores: Collection, scores: Collection) -> np.ndarray:
    """Split-conformal p-value of each score against a calibration set.

    Small p-values mean the score is high relative to the calibration donors,
    so ``conformal_pvalue(...) <= alpha`` selects the donors whose scores are
    extreme at level ``alpha``, and the expected selection rate among
    exchangeable donors is at most ``alpha``. The calibration scores must come
    from donors held out of training, exchangeable with the ones being scored;
    reusing training rows breaks the guarantee exactly the way refitting a
    transformer on test data does. To read ``alpha`` as a false-positive rate
    the calibration set must contain only donors who did not give; see the
    module docstring.

    Parameters
    ----------
    calibration_scores : array-like of shape (n_calibration,)
        Scores of held-out donors, higher meaning more likely to give. Must be
        non-empty and finite; ``NaN`` and infinities raise rather than being
        dropped, because the denominator ``n + 1`` counts them.
    scores : array-like of shape (n_samples,)
        Scores to test. May be a scalar-like sequence of any length, including
        empty. ``NaN`` entries yield ``NaN`` p-values.

    Returns
    -------
    ndarray of shape (n_samples,)
        P-values in ``[1 / (n_calibration + 1), 1.0]``. Never 0, never above 1.

    Raises
    ------
    ValueError
        If ``calibration_scores`` is empty, not one-dimensional, or contains
        non-finite values.

    Examples
    --------
    >>> import numpy as np
    >>> from philanthropy.metrics import conformal_pvalue
    >>> calibration = np.arange(9, dtype=float)          # 0 .. 8, n = 9
    >>> conformal_pvalue(calibration, [8.5, 4.0, -1.0])
    array([0.1, 0.6, 1. ])

    A score above every calibration point still gets ``1 / (n + 1)``, not 0:

    >>> float(conformal_pvalue(calibration, [1e6])[0])
    0.1
    """
    cal = np.asarray(calibration_scores, dtype=float)
    if cal.ndim != 1:
        raise ValueError("calibration_scores must be one-dimensional.")
    if cal.size == 0:
        raise ValueError("calibration_scores must be non-empty.")
    if not np.all(np.isfinite(cal)):
        raise ValueError("calibration_scores must be finite (no NaN or inf).")

    s = np.asarray(scores, dtype=float)
    n = cal.size
    cal_sorted = np.sort(cal)
    # |{i : cal_i >= s}|; 'left' so calibration points equal to s are counted.
    n_ge = n - np.searchsorted(cal_sorted, s, side="left")
    p = (1.0 + n_ge) / (n + 1.0)
    return np.where(np.isnan(s), np.nan, p)


@dataclass(frozen=True)
class IntervalReport:
    """Everything needed to judge a set of intervals, not just certify them.

    Attributes
    ----------
    n : int
        Rows scored.
    coverage : float
        Fraction of targets inside their interval.
    requested_level : float
        ``1 - alpha``. Compare against the calibrator's ``attained_level_``,
        which is what a one-rank construction can actually deliver, rather than
        against this.
    score_mean, score_median, score_trimmed_mean : float
        The interval score aggregated three ways. The mean is the proper score;
        the other two are there because on gift amounts a handful of donors can
        carry it. A ranking that flips between them is a ranking of the tail.
    median_width : float
        Median ``upper - lower``.
    median_target : float
        Median ``y_true``.
    width_ratio : float
        ``median_width / median_target``. The regime indicator: near 2 is an
        interval a gift officer can work from, near 40 is one that is valid and
        carries no information. ``inf`` when the median target is zero.
    median_bound_ratio : float
        Median ``upper / lower`` over rows with a positive lower bound, so a
        clipped-to-zero interval does not read as infinitely wide. ``nan`` when
        no row has one. A ratio of 3,627 is an interval from ``yhat / 60`` to
        ``60 * yhat``.
    """

    n: int
    coverage: float
    requested_level: float
    score_mean: float
    score_median: float
    score_trimmed_mean: float
    median_width: float
    median_target: float
    width_ratio: float
    median_bound_ratio: float


def _interval_inputs(
    y_true: Collection,
    lower: Collection,
    upper: Collection,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    y = np.asarray(y_true, dtype=float).ravel()
    lo = np.asarray(lower, dtype=float).ravel()
    hi = np.asarray(upper, dtype=float).ravel()
    if not (y.shape == lo.shape == hi.shape):
        raise ValueError(
            "y_true, lower and upper must have the same shape, got "
            f"{y.shape}, {lo.shape}, {hi.shape}."
        )
    if y.size == 0:
        raise ValueError("y_true must be non-empty.")
    a = float(alpha)
    if not 0 < a < 1:
        raise ValueError(f"alpha must satisfy 0 < alpha < 1, got {alpha!r}.")
    if np.any(hi < lo):
        raise ValueError(
            "upper must be >= lower for every row; the interval score of an "
            "inverted interval is not interpretable."
        )
    return y, lo, hi, a


def _interval_score_per_row(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray, a: float
) -> np.ndarray:
    return (
        (hi - lo)
        + (2.0 / a) * np.maximum(lo - y, 0.0)
        + (2.0 / a) * np.maximum(y - hi, 0.0)
    )


def interval_score(
    y_true: Collection,
    lower: Collection,
    upper: Collection,
    alpha: float = 0.05,
) -> float:
    """Mean interval score of a central ``1 - alpha`` interval.

    ``(u - l) + (2/alpha)(l - y)+ + (2/alpha)(y - u)+``, averaged. Lower is
    better. Proper for a central interval, so widening to buy coverage costs
    more than it gains and the score cannot be gamed.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Realised amounts.
    lower, upper : array-like of shape (n_samples,)
        The interval bounds.
    alpha : float, default=0.05
        The miscoverage the interval was built for. This is the score's penalty
        weight, not a level to be attained, so the caller's requested ``alpha``
        is the right value here even when the attained level differs.

    Returns
    -------
    float
        The mean score, in the target's units.

    Raises
    ------
    ValueError
        If the three arrays disagree in shape, are empty, ``alpha`` is outside
        ``(0, 1)``, or any ``upper`` is below its ``lower``.

    Examples
    --------
    A covered row costs its width; a missed row adds ``2 / alpha`` per dollar
    of miss:

    >>> from philanthropy.metrics import interval_score
    >>> interval_score([10.0, 30.0], [5.0, 5.0], [20.0, 20.0], alpha=0.5)
    35.0
    """
    y, lo, hi, a = _interval_inputs(y_true, lower, upper, alpha)
    return float(np.mean(_interval_score_per_row(y, lo, hi, a)))


def interval_report(
    y_true: Collection,
    lower: Collection,
    upper: Collection,
    alpha: float = 0.05,
    trim: float = 0.1,
) -> IntervalReport:
    """Coverage, interval score and width-to-target for a set of intervals.

    Coverage alone cannot tell a useful interval from ``[0, inf)``. This returns
    the three things that can: the proper score aggregated robustly, the width
    relative to the amounts being predicted, and the ratio between the bounds.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Realised amounts.
    lower, upper : array-like of shape (n_samples,)
        The interval bounds.
    alpha : float, default=0.05
        Miscoverage the intervals were built for; sets both the score's penalty
        weight and ``requested_level``.
    trim : float, default=0.1
        Fraction of rows dropped from **each** end before ``score_trimmed_mean``.
        Must be in ``[0, 0.5)``.

    Returns
    -------
    IntervalReport

    Raises
    ------
    ValueError
        On the same input problems as :func:`interval_score`, or ``trim``
        outside ``[0, 0.5)``.

    Examples
    --------
    >>> from philanthropy.metrics import interval_report
    >>> report = interval_report([10.0, 30.0], [5.0, 5.0], [20.0, 20.0], alpha=0.5)
    >>> report.coverage, report.median_width, report.median_target
    (0.5, 15.0, 20.0)
    >>> report.width_ratio, report.median_bound_ratio
    (0.75, 4.0)

    The score is heavy-tailed on gift amounts, so read all three:

    >>> report.score_mean, report.score_median
    (35.0, 35.0)
    """
    y, lo, hi, a = _interval_inputs(y_true, lower, upper, alpha)
    if not 0 <= trim < 0.5:
        raise ValueError(f"trim must satisfy 0 <= trim < 0.5, got {trim!r}.")

    per_row = _interval_score_per_row(y, lo, hi, a)
    width = hi - lo
    median_width = float(np.median(width))
    median_target = float(np.median(y))
    positive = lo > 0

    return IntervalReport(
        n=int(y.size),
        coverage=float(np.mean((y >= lo) & (y <= hi))),
        requested_level=1.0 - a,
        score_mean=float(np.mean(per_row)),
        score_median=float(np.median(per_row)),
        score_trimmed_mean=_trimmed_mean(per_row, trim),
        median_width=median_width,
        median_target=median_target,
        width_ratio=(
            median_width / median_target
            if median_target != 0
            else (np.inf if median_width > 0 else np.nan)
        ),
        median_bound_ratio=(
            float(np.median(hi[positive] / lo[positive]))
            if positive.any()
            else np.nan
        ),
    )


def _trimmed_mean(values: np.ndarray, trim: float) -> float:
    """Mean after dropping ``floor(trim * n)`` rows from each end."""
    ordered = np.sort(values)
    k = min(int(math.floor(trim * ordered.size)), (ordered.size - 1) // 2)
    kept = ordered[k: ordered.size - k] if k else ordered
    return float(np.mean(kept))
