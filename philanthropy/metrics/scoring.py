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


def cost_per_dollar_raised(
    total_fundraising_expense: float,
    total_raised: float,
) -> float:
    """Fundraising expense per dollar of revenue raised.

    A headline efficiency KPI: values below ~0.20 are typically healthy for a
    mature program. Returns ``np.inf`` when ``total_raised`` is 0 (spend with
    nothing raised), so the result is always safe to compare or plot.
    """
    if total_raised == 0:
        return np.inf

    return total_fundraising_expense / total_raised


def fundraising_roi(
    total_raised: float,
    total_fundraising_expense: float,
) -> float:
    """Net return on fundraising investment, ``(raised - expense) / expense``.

    ``0.0`` means the program broke even; ``3.0`` means every dollar spent
    returned three dollars of net revenue. Returns ``np.inf`` when
    ``total_fundraising_expense`` is 0 (revenue with no spend).
    """
    if total_fundraising_expense == 0:
        return np.inf

    return (total_raised - total_fundraising_expense) / total_fundraising_expense
