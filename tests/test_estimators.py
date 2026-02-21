import pytest
from sklearn.utils.estimator_checks import check_estimator
from philanthropy.models import MajorGiftClassifier, LapsePredictor
import numpy as np

def test_major_gift_classifier_sklearn_compatible():
    # We test it with default parameters to ensure basic compatibility
    estimator = MajorGiftClassifier()
    check_estimator(estimator)

def test_lapse_predictor_sklearn_compatible():
    # Test LapsePredictor
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
    assert scores.dtype == int
