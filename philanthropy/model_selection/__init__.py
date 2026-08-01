"""
philanthropy.model_selection
============================
Fiscal-year–aware cross-validation for donor analytics.
"""

from ._temporal_donor_splitter import FiscalYearGroupedSplitter

__all__ = ["FiscalYearGroupedSplitter"]
