"""Locks behaviour that a docstring previously described incorrectly.

Each test asserts what the code actually does, so the next person to reword a
docstring has to reword it to match a passing test rather than to match intent.
"""

import numpy as np
import pandas as pd
import pytest

from philanthropy.preprocessing import (
    EncounterTransformer,
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


@pytest.mark.filterwarnings(
    "ignore:WealthScreeningImputerKNN\\(group_col_idx:DeprecationWarning"
)
def test_group_col_idx_is_wired_up_not_ignored():
    # This test previously asserted the opposite, locking in "group_col_idx is
    # stored and never read" from when the docstring said "ignored".
    #
    # It deliberately does NOT assert that grouping changes the imputed values.
    # Measured across several synthetic setups, that difference is small and not
    # reliably reproducible: a donor's nearest neighbours by feature distance
    # usually share their group already, so the grouped and global fits often
    # agree exactly. What is reliable, and what this locks in, is that the
    # parameter is honoured rather than discarded.
    rng = np.random.default_rng(0)
    n = 40
    X = np.column_stack([
        np.r_[rng.normal(5e4, 2e3, n), rng.normal(5e6, 2e5, n)],
        np.r_[np.zeros(n), np.ones(n)],
    ])
    X[0, 0] = np.nan
    X[n, 0] = np.nan

    kwargs = dict(strategy="knn", n_neighbors=5, add_indicator=False)
    model = WealthScreeningImputerKNN(group_col_idx=1, **kwargs).fit(X)

    # A per-group imputer exists for each qualifying group, which is the thing
    # that was previously absent entirely.
    assert set(model.group_imputers_) == {0.0, 1.0}
    out = model.transform(X)
    assert not np.isnan(out).any()
    assert out[0, 0] < 1e5 and out[n, 0] > 1e6


def test_days_since_last_discharge_is_float_and_carries_nan():
    # Documented as an "Integer number of days". It is float64, and it has to be:
    # a donor absent from the encounter table gets NaN, which an integer dtype
    # cannot represent. Casting to int here would silently destroy missingness.
    enc = pd.DataFrame({"donor_id": [1], "discharge_date": ["2019-01-01"]})
    X = pd.DataFrame({
        "donor_id": [1, 2],
        "gift_date": ["2021-01-01", "2021-01-01"],
        "amt": [1.0, 2.0],
    })
    out = (
        EncounterTransformer(encounter_df=enc)
        .set_output(transform="pandas")
        .fit(X)
        .transform(X)
    )
    col = out["days_since_last_discharge"]
    assert col.dtype == np.float64
    assert col.iloc[0] == 731.0          # donor 1, a real gap
    assert np.isnan(col.iloc[1])         # donor 2, no encounter at all


def test_encounter_frequency_score_counts_rows_not_distinct_encounters():
    # Documented as a "Log-scaled count of distinct encounter records". It is
    # log1p of the ROW count: donor 1 has three rows on two distinct dates and
    # scores log1p(3), not log1p(2).
    enc = pd.DataFrame({
        "donor_id": [1, 1, 1],
        "discharge_date": ["2019-01-01", "2019-06-01", "2019-06-01"],
    })
    X = pd.DataFrame({"donor_id": [1], "gift_date": ["2021-01-01"], "amt": [1.0]})
    out = (
        EncounterTransformer(encounter_df=enc)
        .set_output(transform="pandas")
        .fit(X)
        .transform(X)
    )
    score = out["encounter_frequency_score"].iloc[0]
    assert score == pytest.approx(np.log1p(3))
    assert score != pytest.approx(np.log1p(2))
