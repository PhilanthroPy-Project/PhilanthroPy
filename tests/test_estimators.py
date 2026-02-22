import pytest
from sklearn.utils.estimator_checks import check_estimator
from philanthropy.models import MajorGiftClassifier, LapsePredictor
import numpy as np

def test_major_gift_classifier_sklearn_compatible():
    # We test it with default parameters to ensure basic compatibility
    estimator = MajorGiftClassifier()
    check_estimator(estimator)

@pytest.mark.xfail(
    reason=(
        "LapsePredictor.fit() has a non-standard signature: fit(X, gift_dates, reference_date=None). "
        "The second argument is 'gift_dates' not 'y', which violates sklearn's estimator contract. "
        "check_estimator's check_fit_score_takes_y test requires the second parameter to be named 'y' or 'Y'. "
        "This is a known API design tradeoff: the gift_dates parameter has domain-specific meaning "
        "and cannot be renamed without losing clarity. See GitHub issue #42."
    ),
    strict=True,
)
def test_lapse_predictor_sklearn_compatible():
    # Test LapsePredictor - xfail because of non-standard fit signature
    estimator = LapsePredictor()
    check_estimator(estimator)

def test_major_gift_predict_affinity_score():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    clf = MajorGiftClassifier(random_state=42)
    clf.fit(X, y)

    scores = clf.predict_affinity_score(X)
    assert scores.shape == (100,)
    assert (scores >= 0).all() and (scores <= 100).all()
    # predict_affinity_score returns float64 (np.round returns float, not int)
    assert np.issubdtype(scores.dtype, np.floating)
