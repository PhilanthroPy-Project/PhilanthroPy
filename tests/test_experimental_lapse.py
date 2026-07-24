import numpy as np
from philanthropy.experimental import LapsePredictor


def test_experimental_lapse_fit_predict_smoke():
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)

    clf = LapsePredictor(n_estimators=10, random_state=0)
    assert clf.fit(X, y) is clf
    assert clf.n_features_in_ == 4
    assert set(clf.classes_) == {0, 1}

    preds = clf.predict(X)
    assert preds.shape == (100,)
    assert set(np.unique(preds)).issubset({0, 1})

    proba = clf.predict_proba(X)
    assert proba.shape == (100, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
