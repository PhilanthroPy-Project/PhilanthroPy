"""
philanthropy.models
===================
Donor propensity, lapse prediction, and share-of-wallet capacity models.
"""

from .propensity import PropensityScorer
from ._propensity import DonorPropensityModel, MajorGiftClassifier
from ._wallet import ShareOfWalletRegressor
from ._moves import MovesManagementClassifier
from ._lapse import LapsePredictor
from ._planned_giving import PlannedGivingIntentScorer

__all__ = [
    "DonorPropensityModel",
    "LapsePredictor",
    "MajorGiftClassifier",
    "MovesManagementClassifier",
    "PropensityScorer",
    "ShareOfWalletRegressor",
    "PlannedGivingIntentScorer",
]
