"""
tests/test_karlan_list.py

The real file is not shipped; these tests write a tiny Stata file with the
published variable names.
"""

import numpy as np
import pandas as pd
import pytest

from philanthropy.datasets import load_karlan_list


@pytest.fixture
def dta(tmp_path):
    nan = np.nan
    raw = pd.DataFrame(
        {
            "gave": [1, 0, 0, nan],
            "amount": [50, 0, 0, nan],
            "treatment": [1, 0, 1, nan],
            "control": [0, 1, 0, nan],
            "ratio": [2, 0, 1, nan],
            "size": [4, 0, 1, nan],
            "ask": [3, 0, 1, nan],
            "freq": [3, 10, 1, nan],
            "HPA": [40, 25, 100, nan],
            "MRM2": [5, 30, nan, nan],
            "years": [4, 12, 1, nan],
            "female": [1, nan, 0, nan],
            "couple": [0, 0, nan, nan],
            "red0": [1, 0, nan, nan],
            "redcty": [1, 0, 1, nan],
        }
    ).astype("float32")
    path = tmp_path / "AERtables1-5.dta"
    raw.to_stata(path, write_index=False)
    return str(path)


def test_renames_drops_empty_rows_and_types(dta):
    df = load_karlan_list(dta)
    assert len(df) == 3  # the all-missing row is gone
    assert list(df.columns) == [
        "gave", "amount", "matched", "match_ratio", "match_cap", "ask_multiple",
        "prior_gifts", "highest_previous_amount", "months_since_last_gift",
        "years_since_first_gift", "female", "couple", "red_state", "red_county",
    ]
    for col in ["gave", "matched", "match_ratio", "match_cap", "ask_multiple"]:
        assert df[col].dtype == np.int64
    assert df["amount"].tolist() == [50.0, 0.0, 0.0]
    assert df["matched"].tolist() == [1, 0, 1]
    assert df["highest_previous_amount"].tolist() == [40.0, 25.0, 100.0]


def test_partial_missing_values_pass_through(dta):
    df = load_karlan_list(dta)
    assert np.isnan(df.loc[2, "months_since_last_gift"])
    assert np.isnan(df.loc[1, "female"])
