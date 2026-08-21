"""Tests for save_model / load_model bundle helpers."""

import warnings

import joblib
import numpy as np
import pytest

from philanthropy.models import DonorPropensityModel
from philanthropy.utils import save_model, load_model


def _fit_model():
    rng = np.random.default_rng(0)
    X = rng.random((60, 3))
    y = (X[:, 0] > 0.5).astype(int)
    return DonorPropensityModel(n_estimators=10, random_state=0).fit(X, y)


def test_save_load_round_trip(tmp_path):
    model = _fit_model()
    path = tmp_path / "model.joblib"
    save_model(model, path, features=["a", "b", "c"], target="is_major_donor")

    bundle = load_model(path)
    assert bundle["model"] is not None
    assert bundle["features"] == ["a", "b", "c"]
    assert bundle["target"] == "is_major_donor"
    assert "philanthropy_version" in bundle
    assert "sklearn_version" in bundle


def test_load_warns_on_version_mismatch(tmp_path):
    bundle = {
        "model": object(),
        "philanthropy_version": "0.0.1",
        "sklearn_version": "0.0.1",
    }
    path = tmp_path / "old.joblib"
    joblib.dump(bundle, path)

    with pytest.warns(UserWarning):
        load_model(path)


def test_load_no_warning_when_versions_match(tmp_path):
    model = _fit_model()
    path = tmp_path / "model.joblib"
    save_model(model, path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_model(path)

    # Scope this to the warnings load_model is responsible for. A blanket
    # simplefilter("error") also fails on unrelated third-party warnings raised
    # during unpickling: numpy 2.5 makes joblib/numpy_pickle.py emit a
    # DeprecationWarning, which has nothing to do with version matching.
    mismatch = [w for w in caught if "was saved with" in str(w.message)]
    assert not mismatch, [str(w.message) for w in mismatch]


def test_load_rejects_non_bundle(tmp_path):
    path = tmp_path / "not_a_bundle.joblib"
    joblib.dump([1, 2, 3], path)
    with pytest.raises(ValueError):
        load_model(path)
