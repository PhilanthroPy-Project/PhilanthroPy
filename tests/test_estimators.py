"""Behavioural checks for MajorGiftClassifier.

The check_estimator battery for every compliant estimator lives in exactly one
place: tests/test_sklearn_compliance.py::_STANDARD_ESTIMATORS.
"""

import numpy as np

from philanthropy.models import MajorGiftClassifier


def test_major_gift_predict_affinity_score():
    rng = np.random.default_rng(0)
    X = rng.random((100, 5))
    y = rng.integers(0, 2, 100)

    clf = MajorGiftClassifier(random_state=42)
    clf.fit(X, y)

    scores = clf.predict_affinity_score(X)
    assert scores.shape == (100,)
    assert (scores >= 0).all() and (scores <= 100).all()
    # predict_affinity_score returns float64 (np.round returns float, not int)
    assert np.issubdtype(scores.dtype, np.floating)
