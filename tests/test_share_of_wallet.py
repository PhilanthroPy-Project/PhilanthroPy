"""
tests/test_share_of_wallet.py
Test suite for ShareOfWalletRegressor.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from philanthropy.models import ShareOfWalletRegressor
from philanthropy.preprocessing import ShareOfWalletScorer, WealthScreeningImputerKNN


@pytest.fixture
def wallet_Xy():
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1e6, (100, 5))
    y = rng.uniform(1e4, 5e6, 100)
    return X, y


def test_fit_predict_shapes(wallet_Xy):
    X, y = wallet_Xy
    model = ShareOfWalletRegressor(max_iter=20, random_state=0)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (100,)
    assert preds.dtype in (np.float32, np.float64)


def test_fit_predict_dtype(wallet_Xy):
    X, y = wallet_Xy
    model = ShareOfWalletRegressor(max_iter=20, random_state=0)
    model.fit(X, y)
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert np.issubdtype(preds.dtype, np.floating)


def test_capacity_floor_clipping(wallet_Xy):
    X, y = wallet_Xy
    model = ShareOfWalletRegressor(capacity_floor=5000.0, max_iter=20, random_state=0)
    model.fit(X, y)
    preds = model.predict(X)
    assert (preds >= 5000.0).all()


def test_predict_with_nan_inputs_no_error(wallet_Xy):
    X, y = wallet_Xy
    X_nan = X.copy()
    X_nan[0:10, 0] = np.nan
    model = ShareOfWalletRegressor(max_iter=20, random_state=0)
    model.fit(X_nan, y)
    preds = model.predict(X_nan)
    assert preds.shape == (100,)
    assert not np.any(np.isnan(preds))


def test_predict_capacity_ratio_formula(wallet_Xy):
    X, y = wallet_Xy
    model = ShareOfWalletRegressor(max_iter=20, random_state=0)
    model.fit(X, y)
    hist = np.array([100.0, 200.0, 300.0])
    ratios = model.predict_capacity_ratio(X[:3], historical_giving=hist)
    expected = model.predict(X[:3]) / np.maximum(hist, 1.0)
    np.testing.assert_array_almost_equal(ratios, expected)


def test_predict_capacity_ratio_shape(wallet_Xy):
    X, y = wallet_Xy
    model = ShareOfWalletRegressor(max_iter=20, random_state=0)
    model.fit(X, y)
    hist = np.ones(100) * 1000.0
    ratios = model.predict_capacity_ratio(X, historical_giving=hist)
    assert ratios.shape == (100,)


def test_predict_capacity_ratio_dtype(wallet_Xy):
    X, y = wallet_Xy
    model = ShareOfWalletRegressor(max_iter=20, random_state=0)
    model.fit(X, y)
    hist = np.ones(100) * 1000.0
    ratios = model.predict_capacity_ratio(X, historical_giving=hist)
    assert np.issubdtype(ratios.dtype, np.floating)


def test_predict_capacity_ratio_zero_historical_denominator_floor(wallet_Xy):
    X, y = wallet_Xy
    model = ShareOfWalletRegressor(max_iter=20, random_state=0)
    model.fit(X, y)
    hist = np.zeros(3)
    ratios = model.predict_capacity_ratio(X[:3], historical_giving=hist)
    assert ratios.shape == (3,)
    assert not np.any(np.isnan(ratios))
    assert not np.any(np.isinf(ratios))


def test_not_fitted_raises_predict(wallet_Xy):
    X, y = wallet_Xy
    model = ShareOfWalletRegressor()
    with pytest.raises(NotFittedError):
        model.predict(X)


def test_not_fitted_raises_predict_capacity_ratio(wallet_Xy):
    X, y = wallet_Xy
    model = ShareOfWalletRegressor()
    with pytest.raises(NotFittedError):
        model.predict_capacity_ratio(X, historical_giving=np.ones(100))


def test_pipeline_compatibility(wallet_Xy):
    X, y = wallet_Xy
    pipe = Pipeline([
        ("model", ShareOfWalletRegressor(max_iter=20, random_state=0)),
    ])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (100,)


def test_clone_compatibility(wallet_Xy):
    X, y = wallet_Xy
    model = ShareOfWalletRegressor()
    model.fit(X, y)
    cloned = clone(model)
    assert not hasattr(cloned, "estimator_")


def test_get_params_set_params_round_trip():
    model = ShareOfWalletRegressor(max_iter=50, capacity_floor=100.0)
    params = model.get_params()
    assert params["max_iter"] == 50
    assert params["capacity_floor"] == 100.0
    model.set_params(max_iter=30, capacity_floor=200.0)
    assert model.max_iter == 30
    assert model.capacity_floor == 200.0


def test_random_state_reproducibility(wallet_Xy):
    X, y = wallet_Xy
    m1 = ShareOfWalletRegressor(max_iter=20, random_state=42)
    m2 = ShareOfWalletRegressor(max_iter=20, random_state=42)
    m1.fit(X, y)
    m2.fit(X, y)
    np.testing.assert_array_almost_equal(m1.predict(X), m2.predict(X))


def test_capacity_floor_respected(wallet_Xy):
    X, y = wallet_Xy
    floor = 1e6
    model = ShareOfWalletRegressor(capacity_floor=floor, max_iter=20, random_state=0)
    model.fit(X, y)
    preds = model.predict(X)
    assert (preds >= floor).all()


def test_l2_regularization_reaches_the_inner_estimator(wallet_Xy):
    X, y = wallet_Xy
    model = ShareOfWalletRegressor(l2_regularization=1.0, max_iter=20, random_state=0)
    model.fit(X, y)
    # A shape assertion passes whether or not the param is wired through.
    assert model.estimator_.l2_regularization == 1.0


def test_max_iter_reaches_the_inner_estimator(wallet_Xy):
    X, y = wallet_Xy
    model = ShareOfWalletRegressor(max_iter=50, random_state=0)
    model.fit(X, y)
    assert model.estimator_.max_iter == 50
    assert model.n_iter_ <= 50


def test_estimator_attribute_accessible(wallet_Xy):
    X, y = wallet_Xy
    model = ShareOfWalletRegressor(max_iter=20, random_state=0)
    model.fit(X, y)
    assert hasattr(model, "estimator_")
    from sklearn.ensemble import HistGradientBoostingRegressor
    assert isinstance(model.estimator_, HistGradientBoostingRegressor)


def test_predict_capacity_ratio_length_mismatch_raises(wallet_Xy):
    X, y = wallet_Xy
    model = ShareOfWalletRegressor(max_iter=20, random_state=0)
    model.fit(X, y)
    with pytest.raises(ValueError, match="same length"):
        model.predict_capacity_ratio(X, historical_giving=np.ones(50))


def test_fit_returns_self(wallet_Xy):
    X, y = wallet_Xy
    model = ShareOfWalletRegressor(max_iter=5, random_state=0)
    result = model.fit(X, y)
    assert result is model


def test_n_features_in_after_fit(wallet_Xy):
    X, y = wallet_Xy
    model = ShareOfWalletRegressor(max_iter=5, random_state=0)
    model.fit(X, y)
    assert model.n_features_in_ == 5


def test_predict_single_sample(wallet_Xy):
    X, y = wallet_Xy
    model = ShareOfWalletRegressor(max_iter=20, random_state=0)
    model.fit(X, y)
    pred = model.predict(X[:1])
    assert pred.shape == (1,)


def test_predict_capacity_ratio_single_sample(wallet_Xy):
    X, y = wallet_Xy
    model = ShareOfWalletRegressor(max_iter=20, random_state=0)
    model.fit(X, y)
    ratios = model.predict_capacity_ratio(X[:1], historical_giving=np.array([1000.0]))
    assert ratios.shape == (1,)


def test_n_iter_property(wallet_Xy):
    X, y = wallet_Xy
    model = ShareOfWalletRegressor(max_iter=20, random_state=0)
    model.fit(X, y)
    n_iter = model.n_iter_
    assert n_iter >= 1
    assert isinstance(n_iter, (int, np.integer))


def test_negative_historical_giving_uses_floor(wallet_Xy):
    X, y = wallet_Xy
    model = ShareOfWalletRegressor(max_iter=20, random_state=0)
    model.fit(X, y)
    hist = np.array([-100.0, -200.0, -300.0])
    ratios = model.predict_capacity_ratio(X[:3], historical_giving=hist)
    expected = model.predict(X[:3]) / 1.0
    np.testing.assert_array_almost_equal(ratios, expected)


# --------------------------------------------------------------------------- #
# WealthScreeningImputerKNN — the feature-name contract on money columns
# (philanthropy/preprocessing/_share_of_wallet.py:279-292 was never executed)
# --------------------------------------------------------------------------- #
def _knn_frame():
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        rng.uniform(0, 1e6, (40, 3)),
        columns=["estimated_net_worth", "real_estate_value", "years_active"],
    )
    df.loc[df.index[:8], "estimated_net_worth"] = np.nan
    df.loc[df.index[5:12], "real_estate_value"] = np.nan
    return df


def test_knn_feature_names_out_width_matches_transform():
    df = _knn_frame()
    imp = WealthScreeningImputerKNN(strategy="knn", n_neighbors=3).fit(df)
    names = imp.get_feature_names_out()
    out = imp.transform(df)
    assert len(names) == out.shape[1]


def test_knn_feature_names_out_appends_one_indicator_per_imputed_col():
    df = _knn_frame()
    imp = WealthScreeningImputerKNN(strategy="knn", n_neighbors=3).fit(df)
    names = list(imp.get_feature_names_out())
    assert names[:3] == list(df.columns)
    assert names[3:] == [f"{c}__was_missing" for c in imp.imputed_cols_]
    # `years_active` is not a wealth column, so it gets no indicator.
    assert "years_active__was_missing" not in names


def test_knn_feature_names_out_has_no_indicators_when_disabled():
    df = _knn_frame()
    imp = WealthScreeningImputerKNN(
        strategy="knn", n_neighbors=3, add_indicator=False
    ).fit(df)
    assert list(imp.get_feature_names_out()) == list(df.columns)


def test_knn_feature_names_out_honours_input_features():
    df = _knn_frame()
    imp = WealthScreeningImputerKNN(
        strategy="knn", n_neighbors=3, add_indicator=False
    ).fit(df)
    renamed = ["a", "b", "c"]
    assert list(imp.get_feature_names_out(renamed)) == renamed


def test_knn_warns_for_a_requested_wealth_column_that_is_absent():
    df = _knn_frame()
    imp = WealthScreeningImputerKNN(
        wealth_cols=["estimated_net_worth", "stock_holdings"],
        strategy="knn",
        n_neighbors=3,
    )
    with pytest.warns(UserWarning, match="'stock_holdings' not found in X"):
        imp.fit(df)
    assert "stock_holdings" not in imp.imputed_cols_


# --------------------------------------------------------------------------- #
# Parameter validation (moved here from the deleted tests/test_coverage_boost.py,
# with each `match=` tightened to the full message).
# --------------------------------------------------------------------------- #
def test_knn_invalid_strategy_lists_the_valid_ones():
    with pytest.raises(
        ValueError,
        match=r"`strategy` must be one of \['knn', 'mean', 'median', 'zero'\], "
              r"got 'invalid'\.",
    ):
        WealthScreeningImputerKNN(strategy="invalid").fit(
            pd.DataFrame({"net_worth": [1.0, 2.0]})
        )


def test_knn_n_neighbors_below_one_raises():
    with pytest.raises(ValueError, match=r"`n_neighbors` must be >= 1, got 0\."):
        WealthScreeningImputerKNN(strategy="knn", n_neighbors=0).fit(
            pd.DataFrame({"net_worth": [1.0, 2.0]})
        )


def test_knn_resolves_wealth_columns_by_substring_when_unset():
    X = pd.DataFrame({"donor_net_worth": [1e6, np.nan], "other": [1.0, 2.0]})
    imp = WealthScreeningImputerKNN(wealth_cols=None).fit(X)
    assert "donor_net_worth" in imp.imputed_cols_
    assert "other" not in imp.imputed_cols_


@pytest.mark.parametrize("strategy", ["median", "mean", "zero"])
def test_knn_non_knn_strategies_still_append_indicators(strategy):
    X = pd.DataFrame({
        "real_estate": [100.0, np.nan, 300.0],
        "net_worth": [1.0, 2.0, 3.0],
    })
    out = WealthScreeningImputerKNN(
        strategy=strategy, add_indicator=True
    ).fit_transform(X)
    assert out.shape[1] == 4  # 2 original + 2 indicators (both match by substring)


def test_share_of_wallet_scorer_negative_epsilon_raises():
    with pytest.raises(ValueError, match=r"`epsilon` must be >= 0, got -1\."):
        ShareOfWalletScorer(epsilon=-1).fit(np.ones((5, 2)))


def test_share_of_wallet_scorer_negative_capacity_col_idx_raises():
    with pytest.raises(ValueError, match=r"`capacity_col_idx` must be >= 0, got -1\."):
        ShareOfWalletScorer(capacity_col_idx=-1).fit(np.ones((5, 2)))


def test_share_of_wallet_scorer_tier_labels_are_exact():
    # The old assertion was an `or` over all three labels, so it could not fail.
    X = np.array([[100.0, 100.0], [10.0, 1_000_000.0], [50.0, 100.0]])
    scorer = ShareOfWalletScorer(capacity_col_idx=0).fit(X)
    np.testing.assert_array_equal(
        scorer.get_tier_labels(X), np.array(["Major", "Leadership", "Leadership"],
                                            dtype=object)
    )
    np.testing.assert_array_equal(scorer.transform(X)[:, 1], [1.0, 0.0, 0.0])
