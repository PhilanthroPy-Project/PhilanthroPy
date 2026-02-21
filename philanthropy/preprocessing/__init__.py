"""
philanthropy.preprocessing
==========================
CRM data cleaning, Fiscal Year–aware feature engineering, and
clinical-encounter feature engineering for medical philanthropy.
"""

from .transformers import FiscalYearTransformer, CRMCleaner
from ._wealth import WealthScreeningImputer
from ._encounters import EncounterTransformer
from ._rfm import RFMTransformer
from ._planned_giving import PlannedGivingIndicator
from ._wealth_percentile import WealthPercentileTransformer


__all__ = [
    "FiscalYearTransformer",
    "CRMCleaner",
    "WealthScreeningImputer",
    "EncounterTransformer",
    "RFMTransformer",
    "PlannedGivingIndicator",
    "WealthPercentileTransformer",
]
