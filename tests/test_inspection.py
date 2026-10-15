"""tests/test_inspection.py: permutation feature importance helper."""

import numpy as np
import pandas as pd
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


# ---------------------------------------------------------------------------
# explain_donor_scores: per-donor median-substitution contributions
# ---------------------------------------------------------------------------

from sklearn.compose import make_column_transformer  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

from philanthropy.inspection import explain_donor_scores  # noqa: E402
from philanthropy.models import (  # noqa: E402
    AskAmountRecommender,
    FinancialForecastModel,
    LapsePredictor,
    MovesManagementClassifier,
    PlannedGivingIntentScorer,
    PropensityScorer,
    ShareOfWalletRegressor,
)


def _driver_data(n=400):
    # Only column 2 drives the target; the rest are noise.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 4))
    return X, (X[:, 2] > 0).astype(int), 5 * X[:, 2] + 0.1 * X[:, 0]


def test_explain_ranks_driving_feature_first_linear():
    X, _, y = _driver_data()
    model = LinearRegression().fit(X, y)
    out = explain_donor_scores(model, X, "predict", baseline=X, top_k=2)
    assert list(out.columns) == [
        "row", "rank", "feature", "value", "baseline", "contribution"
    ]
    assert len(out) == 2 * len(X)
    top = out[out["rank"] == 1]
    # wherever the driver is not itself near-typical, it ranks first
    off_median = np.abs(X[:, 2] - np.median(X[:, 2])) > 0.1
    assert (top["feature"].to_numpy()[off_median] == 2).all()
    # additive model: contribution is exactly coef * (value - median)
    full = explain_donor_scores(model, X, "predict", baseline=X, top_k=None)
    driver = full[full["feature"] == 2]
    expected = 5 * (X[:, 2] - np.median(X[:, 2]))
    np.testing.assert_allclose(driver["contribution"], expected, atol=1e-6)


def test_explain_ranks_driving_feature_first_classifier():
    X, y, _ = _driver_data()
    model = MajorGiftClassifier(random_state=0).fit(X, y)
    out = explain_donor_scores(model, X, "predict_affinity_score", baseline=X)
    top = out[out["rank"] == 1].set_index("row")
    # Donors scored high are high because of the driver: setting it back to
    # the (negative-side) median flips them, so it ranks first, with a
    # positive sign. Low scorers already sit on the median's side, so
    # substitution leaves them unchanged and carries no signal.
    high = X[:, 2] > 0.1
    assert (top.loc[high, "feature"] == 2).all()
    assert (top.loc[high, "contribution"] > 0).all()


SCORERS = [
    (DonorPropensityModel(n_estimators=20, random_state=0), "predict_affinity_score"),
    (DonorPropensityModel(n_estimators=20, random_state=0), "decision_function"),
    (MajorGiftClassifier(random_state=0), "predict_proba"),
    (LapsePredictor(n_estimators=20, random_state=0), "predict_lapse_score"),
    (LapsePredictor(n_estimators=20, random_state=0), "predict_retention_score"),
    (PlannedGivingIntentScorer(n_estimators=20, random_state=0), "predict_intent_score"),
    (PropensityScorer(), "predict_proba"),
    (MovesManagementClassifier(max_iter=20, random_state=0), "predict_proba"),
    (AskAmountRecommender(max_iter=20, random_state=0), "predict"),
    (ShareOfWalletRegressor(max_iter=20, random_state=0), "predict"),
    (FinancialForecastModel(random_state=0), "predict"),
]


@pytest.mark.parametrize("model,method", SCORERS, ids=lambda p: getattr(p, "__name__", str(p)))
def test_explain_works_with_every_scorer(model, method):
    X, y, y_reg = _driver_data(200)
    if isinstance(model, MovesManagementClassifier):
        target = np.digitize(X[:, 2], [-0.5, 0.5])  # three stages
    elif hasattr(model, "predict_proba"):
        target = y
    else:
        target = np.abs(y_reg) + 1
    model.fit(X, target)
    out = explain_donor_scores(model, X[:10], method, baseline=X)
    assert len(out) == 30
    assert np.isfinite(out["contribution"].astype(float)).all()


def test_explain_uses_stored_fill_values():
    X, _, y = _driver_data(200)
    model = FinancialForecastModel(random_state=0).fit(X, y)
    out = explain_donor_scores(model, X[:5], "predict", top_k=None)
    np.testing.assert_allclose(
        out.sort_values(["row", "feature"])["baseline"].to_numpy(dtype=float),
        np.tile(model.fill_values_, 5),
    )


def test_explain_handles_nan():
    X, y, _ = _driver_data()
    model = LapsePredictor(n_estimators=20, random_state=0).fit(X, y)
    Xq = X[:5].copy()
    Xq[0, 2] = np.nan
    out = explain_donor_scores(model, Xq, "predict_lapse_score", baseline=X, top_k=None)
    assert np.isfinite(out["contribution"]).all()
    assert np.isnan(out.query("row == 0 and feature == 2")["value"].iloc[0])


def test_explain_categorical_dataframe_pipeline():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "region": rng.choice(["north", "south", "east"], size=300),
            "years_active": rng.integers(0, 20, size=300),
        }
    )
    y = (df["region"] == "north").astype(int)
    pipe = make_pipeline(
        make_column_transformer(
            (OneHotEncoder(), ["region"]), remainder="passthrough"
        ),
        LapsePredictor(n_estimators=20, random_state=0),
    ).fit(df, y)
    out = explain_donor_scores(pipe, df.iloc[:20], baseline=df, top_k=1)
    mode = df["region"].mode().iloc[0]
    assert set(out["baseline"]) <= {mode, df["years_active"].median()}
    # rows whose region differs from the mode are explained by region
    differs = (df["region"].iloc[:20] != mode).to_numpy()
    assert (out.loc[differs, "feature"] == "region").all()
    assert list(out["row"]) == list(df.index[:20])


def test_explain_missing_baseline_raises():
    X, y, _ = _driver_data(100)
    model = LapsePredictor(n_estimators=5, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="baseline=X_train"):
        explain_donor_scores(model, X, "predict_lapse_score")
    with pytest.raises(ValueError, match="baseline has 2"):
        explain_donor_scores(model, X, "predict_lapse_score", baseline=[0, 0])


def test_explain_class_index_override():
    X, _, _ = _driver_data(200)
    target = np.digitize(X[:, 2], [-0.5, 0.5])
    model = MovesManagementClassifier(max_iter=20, random_state=0).fit(X, target)
    a = explain_donor_scores(model, X[:5], baseline=X, class_index=0, top_k=None)
    b = explain_donor_scores(model, X[:5], baseline=X, class_index=2, top_k=None)
    assert not np.allclose(a["contribution"], b["contribution"])


def test_explain_int_array_keeps_fractional_median():
    # An integer array (gift counts) with median 2.5 must not substitute 2.
    X = np.array([[1, 0], [2, 0], [3, 0], [4, 0]])
    model = LinearRegression().fit(X, 2.0 * X[:, 0])
    out = explain_donor_scores(model, X, "predict", baseline=X, top_k=1)
    np.testing.assert_allclose(out["baseline"].astype(float), 2.5)
    np.testing.assert_allclose(out["contribution"], 2.0 * (X[:, 0] - 2.5))
