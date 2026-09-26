import numpy as np

from philanthropy.models import LapsePredictor


def _dummy_X(n):
    rng = np.random.RandomState(0)
    return rng.rand(n, 3)


def test_lapse_predictor_binary_classes():
    X = _dummy_X(20)
    y = np.array([0, 1] * 10)
    clf = LapsePredictor(n_estimators=5, random_state=0)
    clf.fit(X, y)
    assert list(clf.classes_) == [0, 1]


def test_lapse_predictor_classes_built_with_unique_labels():
    # LapsePredictor.classes_ is now built with sklearn.utils.multiclass's
    # unique_labels, matching PropensityScorer and the other sibling
    # classifiers, instead of a bare np.unique(y). For the plain integer
    # binary target this classifier is documented for, the two calls agree
    # on the result; this test locks in that the fitted classes_ still comes
    # out sorted and matches unique_labels' own output exactly, so a future
    # edit can't silently drift the two apart again.
    from sklearn.utils.multiclass import unique_labels

    X = _dummy_X(20)
    y = np.array([1, 0] * 10)
    clf = LapsePredictor(n_estimators=5, random_state=0)
    clf.fit(X, y)
    assert np.array_equal(clf.classes_, unique_labels(y))
