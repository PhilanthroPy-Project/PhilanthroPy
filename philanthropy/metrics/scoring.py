"""
philanthropy.metrics.scoring
=============================
"""

import numpy as np
from typing import Collection


def donor_retention_rate(
    current_donors: Collection,
    prior_donors: Collection,
) -> float:
    """Share of the prior period's donors who gave again this period.

    Returns a fraction in ``[0.0, 1.0]``; ``0.0`` when ``prior_donors`` is
    empty (no base to retain from).
    """
    current_set = set(current_donors)
    prior_set = set(prior_donors)

    if not prior_set:
        return 0.0

    retained = current_set & prior_set
    return len(retained) / len(prior_set)


def donor_acquisition_cost(
    total_fundraising_expense: float,
    new_donors_acquired: int,
) -> float:
    """Average spend to acquire one new donor.

    Returns ``np.inf`` when ``new_donors_acquired`` is 0 (spend with nothing
    acquired), so the result is always safe to compare or plot.
    """
    if new_donors_acquired == 0:
        return np.inf

    return total_fundraising_expense / new_donors_acquired
