"""
philanthropy.models
===================
Donor propensity, lapse prediction, and share-of-wallet capacity models.
"""

from ._propensity_baseline import PropensityScorer
from ._ask import AskAmountRecommender
from ._propensity import DonorPropensityModel, MajorGiftClassifier
from ._wallet import ShareOfWalletRegressor
from ._moves import MovesManagementClassifier
from ._lapse import LapsePredictor
from ._planned_giving import PlannedGivingIntentScorer
from ._forecast import FinancialForecastModel
from ._conformal_interval import GiftInterval, GiftIntervalCalibrator

__all__ = [
    "AskAmountRecommender",
    "GiftInterval",
    "GiftIntervalCalibrator",
    "DonorPropensityModel",
    "FinancialForecastModel",
    "LapsePredictor",
    "MajorGiftClassifier",
    "MovesManagementClassifier",
    "PropensityScorer",
    "ShareOfWalletRegressor",
    "PlannedGivingIntentScorer",
]
