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


def test_retention_score_is_complement_of_lapse_score():
    X = _dummy_X(20)
    y = np.array([0, 1] * 10)
    clf = LapsePredictor(n_estimators=5, random_state=0).fit(X, y)
    lapse = clf.predict_lapse_score(X)
    retention = clf.predict_retention_score(X)
    np.testing.assert_allclose(lapse + retention, 100.0, atol=1e-9)


def test_retention_read_flag_low_lapse_rate():
    X = _dummy_X(20)
    y = np.array([0] * 15 + [1] * 5)  # 25% lapse rate
    clf = LapsePredictor(n_estimators=5, random_state=0).fit(X, y)
    assert clf.training_lapse_rate_ == 0.25
    assert clf.retention_read_ is False


def test_retention_read_flag_high_lapse_rate():
    X = _dummy_X(20)
    y = np.array([1] * 17 + [0] * 3)  # 85% lapse rate
    clf = LapsePredictor(n_estimators=5, random_state=0).fit(X, y)
    assert clf.training_lapse_rate_ == 0.85
    assert clf.retention_read_ is True


def test_hist_gradient_boosting_backend_fits_and_predicts():
    X = _dummy_X(40)
    y = np.array([0, 1] * 20)
    clf = LapsePredictor(backend="hist_gradient_boosting", random_state=0).fit(X, y)
    score = clf.predict_lapse_score(X)
    assert score.shape == (40,)
    assert ((score >= 0) & (score <= 100)).all()


def test_unknown_backend_raises():
    import pytest

    X = _dummy_X(10)
    y = np.array([0, 1] * 5)
    with pytest.raises(ValueError):
        LapsePredictor(backend="bogus", random_state=0).fit(X, y)
