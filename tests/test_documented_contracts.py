"""Locks behaviour that a docstring previously described incorrectly.

Each test asserts what the code actually does, so the next person to reword a
docstring has to reword it to match a passing test rather than to match intent.
"""

import numpy as np
import pandas as pd

from philanthropy.preprocessing import (
    FiscalYearTransformer,
    WealthScreeningImputerKNN,
)


def test_fiscal_year_transformer_replaces_rather_than_appends():
    # The class and transform docstrings used to say the two fiscal columns were
    # "appended"; get_feature_names_out has always said otherwise. transform
    # returns exactly those two columns and drops the input.
    X = pd.DataFrame({
        "gift_date": ["2023-01-15", "2023-09-01"],
        "gift_amount": [100.0, 250.0],
        "appeal_code": ["A", "B"],
    })
    t = FiscalYearTransformer(date_col="gift_date").fit(X)

    out = t.transform(X)
    assert out.shape == (2, 2)
    assert list(t.get_feature_names_out()) == ["fiscal_year", "fiscal_quarter"]

    frame = t.set_output(transform="pandas").transform(X)
    assert list(frame.columns) == ["fiscal_year", "fiscal_quarter"]
    assert "gift_amount" not in frame.columns


def test_fiscal_year_transformer_all_nan_when_date_col_absent():
    # Documented in transform's Returns section: no date_col means both columns
    # come back NaN for every row rather than raising.
    X = pd.DataFrame({"gift_amount": [100.0, 250.0]})
    out = FiscalYearTransformer(date_col="gift_date").fit(X).transform(X)
    assert out.shape == (2, 2)
    assert np.isnan(out).all()


def test_group_col_idx_is_inert():
    # Documented as stratifying KNN imputation per group; it is stored and never
    # read. The docstring now says "ignored", so lock that in: passing it must
    # not change the output.
    rng = np.random.default_rng(0)
    X = rng.random((30, 4))
    X[X < 0.2] = np.nan

    kwargs = dict(strategy="knn", n_neighbors=3, add_indicator=False)
    without = WealthScreeningImputerKNN(**kwargs).fit_transform(X)
    with_group = WealthScreeningImputerKNN(group_col_idx=3, **kwargs).fit_transform(X)

    np.testing.assert_array_equal(without, with_group)
    # Still round-trips through get_params, which is why the parameter stays.
    assert WealthScreeningImputerKNN(group_col_idx=3).get_params()["group_col_idx"] == 3
