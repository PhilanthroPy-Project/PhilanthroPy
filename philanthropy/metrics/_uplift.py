"""
philanthropy.metrics._uplift
============================
Qini curve and Qini area for evaluating uplift scores on a randomised appeal.

An uplift score (e.g. ``UpliftTLearner.predict_uplift_score``) cannot be
checked with ROC AUC: nobody is observed both solicited and not solicited. The
Qini curve instead ranks donors by score and, at each cut-off, compares the
responses of the treated and control donors above it. It needs a randomised
``treatment`` column; on non-random assignment it measures selection bias.
"""

from __future__ import annotations

from typing import Collection, Tuple

import numpy as np


def qini_curve(
    y_true: Collection,
    uplift: Collection,
    treatment: Collection,
) -> Tuple[np.ndarray, np.ndarray]:
    """Qini curve: cumulative incremental responses by fraction targeted.

    Donors are sorted by ``uplift``, highest first. At each cut-off holding
    the top ``k`` donors, the incremental gain is (Radcliffe 2007, and the
    definition ``scikit-uplift`` uses)::

        Y_t(k) - Y_c(k) * N_t(k) / N_c(k)

    where ``Y_t`` / ``Y_c`` are the summed outcomes of the treated / control
    donors in the top ``k``, and ``N_t`` / ``N_c`` their counts. It reads as
    "extra responders (or dollars) the appeal produced among the donors
    targeted so far", with the control response rescaled to the treated
    count. While no control donor has been reached the control term is 0.

    Tied scores form one cut-off, so the curve does not depend on how ties
    happen to be ordered.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Observed outcome: 1/0 for gave, or a gift amount.
    uplift : array-like of shape (n_samples,)
        Predicted uplift score; higher means target first.
    treatment : array-like of shape (n_samples,)
        1 if the donor was solicited, 0 if held out. Assignment must have
        been random.

    Returns
    -------
    fraction_targeted : ndarray
        Fraction of donors targeted at each cut-off, starting at 0.0 and
        ending at 1.0.
    gains : ndarray
        Cumulative incremental gain at each cut-off, starting at 0.0.

    Raises
    ------
    ValueError
        If inputs are empty, of mismatched length, non-finite, if
        ``treatment`` is not binary 0/1, or if either arm is empty.
    """
    y = np.asarray(y_true, dtype=float)
    score = np.asarray(uplift, dtype=float)
    t = np.asarray(treatment)
    if not (y.shape[0] == score.shape[0] == t.shape[0]):
        raise ValueError(
            f"y_true, uplift and treatment must be the same length, got "
            f"{y.shape[0]}, {score.shape[0]} and {t.shape[0]}."
        )
    if y.shape[0] == 0:
        raise ValueError("y_true is empty; nothing to score.")
    if not (np.isfinite(y).all() and np.isfinite(score).all()):
        raise ValueError("y_true and uplift must be finite (no NaN or inf).")
    if not np.isin(t, [0, 1]).all():
        raise ValueError("treatment must be binary 0/1.")
    t = t.astype(float)
    if t.sum() == 0 or t.sum() == t.shape[0]:
        raise ValueError(
            "Both arms must be present: treatment needs at least one 0 and "
            "one 1."
        )

    order = np.argsort(score, kind="mergesort")[::-1]
    y, score, t = y[order], score[order], t[order]

    n_t = np.cumsum(t)
    n_c = np.cumsum(1.0 - t)
    y_t = np.cumsum(y * t)
    y_c = np.cumsum(y * (1.0 - t))

    # Last index of each run of tied scores, plus the final row.
    cut = np.r_[np.flatnonzero(np.diff(score)), score.shape[0] - 1]
    ratio = np.divide(
        n_t[cut], n_c[cut], out=np.zeros(cut.shape[0]), where=n_c[cut] > 0
    )
    gains = y_t[cut] - y_c[cut] * ratio

    fraction = (cut + 1) / score.shape[0]
    return np.r_[0.0, fraction], np.r_[0.0, gains]


def qini_auc_score(
    y_true: Collection,
    uplift: Collection,
    treatment: Collection,
) -> float:
    """Area between the Qini curve and the random-targeting line.

    The random line runs straight from ``(0, 0)`` to the curve's end point
    ``(1, gains[-1])``: what targeting in random order would earn on
    average. The score is the trapezoidal area under :func:`qini_curve`
    minus the area under that line.

    The score is **not normalised** (``scikit-uplift``'s ``qini_auc_score``
    divides by the perfect model's area by default; this does not). Its unit
    is incremental responses (or dollars, for amount outcomes) times
    fraction targeted, so it scales with sample size and outcome scale. Use
    it to compare models scored on the same holdout, not across datasets.

    Positive means the ranking beats random targeting, about 0 means no
    better than random, negative means worse than random.

    Parameters
    ----------
    y_true, uplift, treatment
        As for :func:`qini_curve`.

    Returns
    -------
    float
        Area between the model's Qini curve and the random line.

    Raises
    ------
    ValueError
        As for :func:`qini_curve`.
    """
    x, gains = qini_curve(y_true, uplift, treatment)
    area = float(np.sum((x[1:] - x[:-1]) * (gains[1:] + gains[:-1]) / 2.0))
    return area - float(gains[-1]) / 2.0
