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
    if new_donors_acquired == 0:
        return np.inf

    return total_fundraising_expense / new_donors_acquired
