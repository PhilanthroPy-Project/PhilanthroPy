"""
Isolated sklearn check_estimator compliance suite.
Run FIRST in CI — any failure here is a pipeline-breaking regression.
pytest tests/test_sklearn_compliance.py -v
"""
from sklearn.utils.estimator_checks import parametrize_with_checks

from philanthropy.models import (
    DonorPropensityModel,
    MajorGiftClassifier,
    PropensityScorer,
    ShareOfWalletRegressor,
    MovesManagementClassifier,
)
from philanthropy.preprocessing import (
    WealthScreeningImputer,
    WealthPercentileTransformer,
    PlannedGivingIndicator,
    RFMTransformer,
    FiscalYearTransformer,
    CRMCleaner,
)

_ALL_ESTIMATORS = [
    DonorPropensityModel(),
    MajorGiftClassifier(),
    PropensityScorer(),
    ShareOfWalletRegressor(),
    MovesManagementClassifier(),
    WealthScreeningImputer(),
    WealthPercentileTransformer(),
    PlannedGivingIndicator(),
    RFMTransformer(),
    FiscalYearTransformer(),
    CRMCleaner(),
]

@parametrize_with_checks(_ALL_ESTIMATORS)
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)
