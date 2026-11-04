"""
philanthropy.models
===================
Donor propensity, lapse prediction, share-of-wallet capacity, and
repeat-giving (BG/NBD, gamma-gamma) models.
"""

from ._propensity_baseline import PropensityScorer
from ._ask import AskAmountRecommender, suggest_ask
from ._propensity import DonorPropensityModel, MajorGiftClassifier
from ._wallet import ShareOfWalletRegressor
from ._moves import MovesManagementClassifier
from ._lapse import LapsePredictor
from ._planned_giving import PlannedGivingIntentScorer
from ._forecast import FinancialForecastModel
from ._conformal_interval import GiftInterval, GiftIntervalCalibrator
from ._upgrade import score_leadership_prospects
from ._repeat_giving import BetaGeoModel, GammaGammaModel, repeat_giving_summary

__all__ = [
    "AskAmountRecommender",
    "BetaGeoModel",
    "GiftInterval",
    "GiftIntervalCalibrator",
    "DonorPropensityModel",
    "FinancialForecastModel",
    "GammaGammaModel",
    "LapsePredictor",
    "MajorGiftClassifier",
    "MovesManagementClassifier",
    "PropensityScorer",
    "ShareOfWalletRegressor",
    "PlannedGivingIntentScorer",
    "repeat_giving_summary",
    "score_leadership_prospects",
    "suggest_ask",
]
