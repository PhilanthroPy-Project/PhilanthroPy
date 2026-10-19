"""
philanthropy.model_selection
============================
Fiscal-year–aware cross-validation for donor analytics, and the per-file
model-vs-rule check.
"""

from ._temporal_donor_splitter import FiscalYearGroupedSplitter
from ._rule_check import compare_to_rule

__all__ = ["FiscalYearGroupedSplitter", "compare_to_rule"]
