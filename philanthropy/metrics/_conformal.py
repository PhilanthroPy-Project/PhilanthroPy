"""
philanthropy.metrics._conformal
================================
Distribution-free p-values for donor scores.

A donor score is consumed as a threshold crossing: a solicitation fires when
the score clears a cut point. Picking that cut point from a calibrated
probability ("contact everyone above 0.8") fixes no error rate at all. The
split-conformal p-value does: rank a scored donor against a held-out
calibration set and the result is uniform on the lattice under exchangeability,
so thresholding it at ``alpha`` bounds the false-positive rate at ``alpha`` in
finite samples, with no distributional assumption.

Only the non-smoothed form is implemented, eq. (3) of Bates et al. (2023):

    p = (1 + |{i : s_i >= s}|) / (n + 1)

The ``1 +`` in the numerator and the ``+ 1`` in the denominator are the test
point itself. Both are load-bearing. Dropping either one produces a statistic
that is not a valid p-value, and dropping only the numerator's can return a
value above 1.
"""

from __future__ import annotations

from typing import Collection

import numpy as np


def conformal_pvalue(calibration_scores: Collection, scores: Collection) -> np.ndarray:
    """Split-conformal p-value of each score against a calibration set.

    Small p-values mean the score is high relative to the calibration donors,
    so ``conformal_pvalue(...) <= alpha`` selects the donors whose scores are
    extreme at level ``alpha``. The calibration scores must come from donors
    held out of training, exchangeable with the ones being scored; reusing
    training rows breaks the guarantee exactly the way refitting a transformer
    on test data does.

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
