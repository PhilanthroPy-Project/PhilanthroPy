"""
philanthropy.metrics._concentration
====================================
Donor-base concentration diagnostics.

Boards and campaign leads routinely ask "how dependent are we on a handful of
donors?". These two stateless helpers answer it: the Gini coefficient of the
gift distribution and the revenue share captured by the top slice of donors.
A highly concentrated base (few donors, most of the money) is a retention and
succession risk; a flat base may signal under-cultivated major-gift capacity.
"""

from __future__ import annotations

from typing import Collection

import numpy as np


def _clean_nonneg_amounts(amounts: Collection) -> np.ndarray:
    """Return finite, non-NaN amounts as a float array; reject negatives."""
    a = np.asarray(amounts, dtype=float)
    a = a[~np.isnan(a)]
    if np.any(a < 0):
        raise ValueError("gift amounts must be non-negative.")
    return a


def gift_concentration_gini(amounts: Collection) -> float:
    """Gini coefficient of a set of donor gift amounts.

    ``0.0`` is perfect equality (every donor gives the same); values approaching
    ``1.0`` mean revenue is concentrated in a few donors.

    Parameters
    ----------
    amounts : array-like of shape (n_donors,)
        Per-donor total giving. ``NaN`` entries are dropped; negatives raise.

    Returns
    -------
    float
        Gini coefficient in ``[0.0, 1.0]``. Returns ``0.0`` for an empty input
        or an all-zero total (no distribution to measure).
    """
    a = _clean_nonneg_amounts(amounts)
    total = a.sum()
    if a.size == 0 or total == 0:
        return 0.0
    a = np.sort(a)
    n = a.size
    index = np.arange(1, n + 1)
    # Mean-absolute-difference form of the Gini coefficient.
    return float((2.0 * np.sum(index * a)) / (n * total) - (n + 1.0) / n)


def top_donor_share(amounts: Collection, top_fraction: float = 0.1) -> float:
    """Fraction of total revenue contributed by the top ``top_fraction`` donors.

    Parameters
    ----------
    amounts : array-like of shape (n_donors,)
        Per-donor total giving. ``NaN`` entries are dropped; negatives raise.
    top_fraction : float, default=0.1
        Slice of donors (ranked by giving, descending) to sum. Must be in
        ``(0.0, 1.0]``. At least one donor is always counted.

    Returns
    -------
    float
        Share in ``[0.0, 1.0]``. Returns ``0.0`` for an empty input or an
        all-zero total.
    """
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0.0, 1.0].")
    a = _clean_nonneg_amounts(amounts)
    total = a.sum()
    if a.size == 0 or total == 0:
        return 0.0
    a = np.sort(a)[::-1]
    k = max(1, int(np.ceil(top_fraction * a.size)))
    return float(a[:k].sum() / total)
