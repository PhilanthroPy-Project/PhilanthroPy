"""
philanthropy.metrics
====================
Donor KPI calculators.
"""

from ._scoring import (
    donor_retention_rate,
    donor_acquisition_cost,
    cost_per_dollar_raised,
    fundraising_roi,
)
from ._financial import donor_lifetime_value
from ._fairness import disparate_impact_ratio, selection_rate_by_group
from ._concentration import gift_concentration_gini, top_donor_share
from ._conformal import conformal_pvalue

__all__ = [
    "donor_retention_rate",
    "donor_acquisition_cost",
    "cost_per_dollar_raised",
    "fundraising_roi",
    "donor_lifetime_value",
    "disparate_impact_ratio",
    "selection_rate_by_group",
    "gift_concentration_gini",
    "top_donor_share",
    "conformal_pvalue",
]
