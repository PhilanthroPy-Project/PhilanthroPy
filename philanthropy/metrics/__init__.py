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
from ._fairness import (
    demographic_parity_difference,
    disparate_impact_ratio,
    metric_by_group,
    selection_rate_by_group,
)
from ._uplift import qini_auc_score, qini_curve
from ._concentration import gift_concentration_gini, top_donor_share
from ._conformal import (
    IntervalReport,
    conformal_pvalue,
    interval_report,
    interval_score,
)

__all__ = [
    "IntervalReport",
    "interval_report",
    "interval_score",
    "donor_retention_rate",
    "donor_acquisition_cost",
    "cost_per_dollar_raised",
    "fundraising_roi",
    "donor_lifetime_value",
    "disparate_impact_ratio",
    "demographic_parity_difference",
    "selection_rate_by_group",
    "metric_by_group",
    "qini_curve",
    "qini_auc_score",
    "gift_concentration_gini",
    "top_donor_share",
    "conformal_pvalue",
]
