import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks
from philanthropy.preprocessing import (
    EncounterRecencyTransformer,
    WealthScreeningImputer,
    ShareOfWalletScorer,
    CRMCleaner,
    FiscalYearTransformer
)
from philanthropy.model_selection import TemporalDonorSplitter

ALL_ESTIMATORS = [
    EncounterRecencyTransformer(),
    WealthScreeningImputer(),
    ShareOfWalletScorer(),
    CRMCleaner(),
    FiscalYearTransformer(),
]

@parametrize_with_checks(ALL_ESTIMATORS)
def test_sklearn_compatible_estimators(estimator, check):
    # Some domain transformers (like CRMCleaner) might need specific skips
    # or handle string inputs which check_estimator doesn't always like.
    check(estimator)
