"""
philanthropy.metrics
====================
Donor KPI calculators.
"""

from .scoring import donor_retention_rate, donor_acquisition_cost
from ._financial import donor_lifetime_value
from ._fairness import disparate_impact_ratio, selection_rate_by_group

__all__ = [
    "donor_retention_rate",
    "donor_acquisition_cost",
    "donor_lifetime_value",
    "disparate_impact_ratio",
    "selection_rate_by_group",
]
