"""Per-group KNN imputation on WealthScreeningImputerKNN.

`group_col_idx` was documented, stored and never read. Now it stratifies the
KNN fit, so a donor's missing wealth is filled from neighbours inside their own
group rather than from the whole database.
"""

import numpy as np
import pandas as pd
import pytest

from philanthropy.preprocessing import WealthScreeningImputerKNN

KW = dict(strategy="knn", n_neighbors=5, add_indicator=False)


def _two_group_pool(seed=0, n=40):
    """Two groups an order of magnitude apart, one missing wealth in each."""
    rng = np.random.default_rng(seed)
    lo = np.column_stack([rng.normal(50_000, 3_000, n), np.zeros(n)])
    hi = np.column_stack([rng.normal(5_000_000, 200_000, n), np.ones(n)])
    X = np.vstack([lo, hi])
    X[0, 0] = np.nan     # low group, missing
    X[n, 0] = np.nan     # high group, missing
    return X, n


def test_grouped_fill_comes_from_inside_the_group():
    # The bounds alone are a weak assertion: plain KNN also satisfies them,
    # because a donor's nearest neighbours in feature space tend to share their
    # group anyway. So assert the bounds AND that grouping moved the answer.
    X, n = _two_group_pool()
    grouped = WealthScreeningImputerKNN(group_col_idx=1, **KW).fit(X).transform(X)
    plain = WealthScreeningImputerKNN(**KW).fit(X).transform(X)

    assert grouped[0, 0] < 100_000
    assert grouped[n, 0] > 1_000_000
    assert not np.isnan(grouped).any()
    assert grouped[0, 0] != plain[0, 0]
    assert grouped[n, 0] != plain[n, 0]


def test_column_all_missing_within_a_group_does_not_become_zero():
    # The bug this guard exists for. KNNImputer(keep_empty_features=True) fills a
    # wholly-missing column with a hard 0.0 rather than NaN, so a NaN check at
    # transform time cannot see it. For a wealth column, 0.0 reads as "no
    # capacity": a materially wrong number, silently, for every donor in that
    # group. Those columns must defer to the global imputer.
    n = 20
    X = np.vstack([
        np.column_stack([np.full(n, np.nan), np.linspace(1e3, 2e3, n), np.zeros(n)]),
        np.column_stack([np.linspace(4e6, 6e6, n), np.linspace(5e5, 7e5, n), np.ones(n)]),
    ])
    grouped = WealthScreeningImputerKNN(group_col_idx=2, **KW).fit(X).transform(X)
    plain = WealthScreeningImputerKNN(**KW).fit(X).transform(X)

    assert not np.any(grouped[:n, 0] == 0.0)
    np.testing.assert_allclose(grouped[:n, 0], plain[:n, 0])


def test_grouping_changes_the_result():
    # If grouping made no difference the parameter would be pointless.
    X, _ = _two_group_pool()
    grouped = WealthScreeningImputerKNN(group_col_idx=1, **KW).fit(X).transform(X)
    plain = WealthScreeningImputerKNN(**KW).fit(X).transform(X)
    assert not np.allclose(grouped, plain)


def test_group_smaller_than_n_neighbors_falls_back_to_global():
    # KNN over fewer neighbours than requested is worse than the global fit, so
    # a small group deliberately gets no imputer of its own.
    X, _ = _two_group_pool()
    tiny = np.column_stack([np.full(3, 9e6), np.full(3, 2.0)])
    tiny[-1, 0] = np.nan
    X = np.vstack([X, tiny])

    model = WealthScreeningImputerKNN(group_col_idx=1, **KW).fit(X)
    assert 2.0 not in model.group_imputers_        # too small, excluded
    assert {0.0, 1.0} <= set(model.group_imputers_)
    assert not np.isnan(model.transform(X)).any()  # still imputed, via global


def test_group_unseen_at_fit_uses_the_global_imputer():
    # The leakage-safe choice. The alternative is fitting at transform time.
    X, _ = _two_group_pool()
    model = WealthScreeningImputerKNN(group_col_idx=1, **KW).fit(X)
    unseen = np.array([[np.nan, 99.0]])
    out = model.transform(unseen)
    assert np.isfinite(out[0, 0])


def test_missing_group_label_uses_the_global_imputer():
    X, _ = _two_group_pool()
    model = WealthScreeningImputerKNN(group_col_idx=1, **KW).fit(X)
    out = model.transform(np.array([[np.nan, np.nan]]))
    assert np.isfinite(out[0, 0])


def test_transform_is_idempotent_and_fits_nothing():
    # The leakage contract: repeated transforms must agree, and transforming a
    # batch must not change what a later transform of the same rows returns.
    X, _ = _two_group_pool()
    model = WealthScreeningImputerKNN(group_col_idx=1, **KW).fit(X)
    first = model.transform(X)
    fitted_groups = set(model.group_imputers_)     # snapshot the keys themselves
    assert fitted_groups                           # and it is not vacuously empty
    model.transform(np.vstack([X, X]))             # a bigger, different batch
    np.testing.assert_array_equal(first, model.transform(X))
    assert set(model.group_imputers_) == fitted_groups


@pytest.mark.parametrize("strategy", ["median", "mean", "zero"])
def test_ignored_for_non_knn_strategies(strategy):
    X, _ = _two_group_pool()
    df = pd.DataFrame(X, columns=["net_worth", "zip_group"])
    grouped = WealthScreeningImputerKNN(
        strategy=strategy, group_col_idx=1, add_indicator=False
    ).fit(df).transform(df)
    plain = WealthScreeningImputerKNN(
        strategy=strategy, add_indicator=False
    ).fit(df).transform(df)
    np.testing.assert_array_equal(grouped, plain)


@pytest.mark.parametrize("bad", [99, -99])
def test_out_of_range_group_col_idx_raises(bad):
    X, _ = _two_group_pool()
    with pytest.raises(ValueError, match="out of range"):
        WealthScreeningImputerKNN(group_col_idx=bad, **KW).fit(X)


def test_group_col_idx_still_round_trips_through_get_params():
    assert WealthScreeningImputerKNN(group_col_idx=3).get_params()["group_col_idx"] == 3
