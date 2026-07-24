"""tests/test_inspection.py — permutation feature importance helper."""

import numpy as np
import pytest

from philanthropy.datasets import generate_synthetic_donor_data
from philanthropy.inspection import donor_feature_importance
from philanthropy.models import DonorPropensityModel, MajorGiftClassifier

FEATURES = ["total_gift_amount", "years_active", "event_attendance_count"]


def _xy(n=300):
    df = generate_synthetic_donor_data(n_samples=n, random_state=0)
    return df[FEATURES].to_numpy(), df["is_major_donor"].to_numpy()


def test_returns_ranked_frame():
    X, y = _xy()
    model = DonorPropensityModel(n_estimators=50, random_state=0).fit(X, y)
    imp = donor_feature_importance(
        model, X, y, feature_names=FEATURES, random_state=0
    )
    assert list(imp.columns) == ["feature", "importance_mean", "importance_std"]
    assert len(imp) == len(FEATURES)
    assert set(imp["feature"]) == set(FEATURES)
    # sorted descending by mean importance
    assert imp["importance_mean"].is_monotonic_decreasing
    assert np.isfinite(imp["importance_mean"]).all()


def test_infers_names_from_dataframe():
    df = generate_synthetic_donor_data(n_samples=300, random_state=0)
    X = df[FEATURES]  # DataFrame
    y = df["is_major_donor"].to_numpy()
    model = DonorPropensityModel(n_estimators=50, random_state=0).fit(X, y)
    imp = donor_feature_importance(model, X, y, random_state=0)
    assert set(imp["feature"]) == set(FEATURES)


def test_works_on_calibrated_model_without_feature_importances():
    # MajorGiftClassifier wraps CalibratedClassifierCV and exposes no
    # feature_importances_, so permutation importance is the only option.
    X, y = _xy()
    model = MajorGiftClassifier(random_state=0).fit(X, y)
    assert not hasattr(model, "feature_importances_")
    imp = donor_feature_importance(model, X, y, feature_names=FEATURES, random_state=0)
    assert len(imp) == len(FEATURES)


def test_feature_names_length_mismatch_raises():
    X, y = _xy()
    model = DonorPropensityModel(n_estimators=10, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="feature_names"):
        donor_feature_importance(model, X, y, feature_names=["only_one"])


def test_defaults_to_positional_names():
    X, y = _xy()
    model = DonorPropensityModel(n_estimators=10, random_state=0).fit(X, y)
    imp = donor_feature_importance(model, X, y, random_state=0)
    assert set(imp["feature"]) == {"x0", "x1", "x2"}
