"""
tests/test_matching_gift.py
============================
Unit tests for philanthropy.preprocessing.MatchingGiftFeaturizer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from philanthropy.preprocessing import MatchingGiftFeaturizer


@pytest.fixture
def X_donors():
    """Donor rows: present employer, null, empty string, whitespace, unknown."""
    return pd.DataFrame({
        "employer": ["Boeing", None, "", "  ", "Acme Corp", "microsoft"],
        "gift_amount": [100.0, 50.0, 25.0, 10.0, np.nan, 200.0],
    })


@pytest.fixture
def ratios():
    return {"Boeing": 1.0, "Microsoft": 2.0}


@pytest.fixture
def fitted(X_donors, ratios):
    return MatchingGiftFeaturizer(match_ratios=ratios).fit(X_donors)


def test_fit_returns_self(X_donors, ratios):
    feat = MatchingGiftFeaturizer(match_ratios=ratios)
    assert feat.fit(X_donors) is feat


def test_output_shape_and_type(fitted, X_donors):
    out = fitted.transform(X_donors)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert out.shape == (len(X_donors), 3)


def test_has_employer_flag(fitted, X_donors):
    # col 0: Boeing=1, None=0, ""=0, "  "=0, "Acme Corp"=1, "microsoft"=1
    expected = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(fitted.transform(X_donors)[:, 0], expected)


def test_match_ratio_case_insensitive_and_unknown(fitted, X_donors):
    out = fitted.transform(X_donors)
    # col 1: Boeing=1.0, unknown None/""/"  "=0.0, "Acme Corp"=0.0,
    # "microsoft" matches "Microsoft" key -> 2.0
    expected = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    np.testing.assert_array_equal(out[:, 1], expected)


def test_potential_matched_amount(fitted, X_donors):
    out = fitted.transform(X_donors)
    # col 2 = gift (NaN->0) * match_ratio
    # Boeing 100*1=100; microsoft 200*2=400; Acme NaN->0*0=0
    expected = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 400.0])
    np.testing.assert_array_equal(out[:, 2], expected)


def test_nan_gift_yields_zero_matched():
    X = pd.DataFrame({"employer": ["Boeing"], "gift_amount": [np.nan]})
    feat = MatchingGiftFeaturizer(match_ratios={"Boeing": 1.0}).fit(X)
    out = feat.transform(X)
    assert out[0, 0] == 1.0  # employer present
    assert out[0, 1] == 1.0  # ratio known
    assert out[0, 2] == 0.0  # NaN gift -> 0 matched


def test_none_ratios_gives_zero_everywhere():
    X = pd.DataFrame({"employer": ["Boeing"], "gift_amount": [100.0]})
    out = MatchingGiftFeaturizer(match_ratios=None).fit(X).transform(X)
    assert out[0, 1] == 0.0
    assert out[0, 2] == 0.0


def test_transform_idempotent(fitted, X_donors):
    first = fitted.transform(X_donors)
    second = fitted.transform(X_donors)
    np.testing.assert_array_equal(first, second)


def test_leakage_output_independent_of_batch(X_donors, ratios):
    """A row's output must not depend on which other rows share the batch."""
    feat = MatchingGiftFeaturizer(match_ratios=ratios).fit(X_donors)

    single = feat.transform(X_donors.iloc[[0]])

    # Append extra rows with different employers/gifts to a fresh batch.
    bigger = pd.concat(
        [
            X_donors.iloc[[0]],
            pd.DataFrame({
                "employer": ["Microsoft", "Nowhere Inc"],
                "gift_amount": [999.0, 1.0],
            }),
        ],
        ignore_index=True,
    )
    batch = feat.transform(bigger)

    np.testing.assert_array_equal(single[0], batch[0])


def test_get_feature_names_out(fitted):
    names = fitted.get_feature_names_out()
    np.testing.assert_array_equal(
        names,
        np.array(
            ["has_employer", "match_ratio", "potential_matched_amount"],
            dtype=object,
        ),
    )


def test_not_fitted_raises(X_donors, ratios):
    feat = MatchingGiftFeaturizer(match_ratios=ratios)
    with pytest.raises(NotFittedError):
        feat.transform(X_donors)


def test_non_dataframe_raises(ratios):
    feat = MatchingGiftFeaturizer(match_ratios=ratios)
    with pytest.raises(TypeError, match="pandas DataFrame"):
        feat.fit([[1, 2], [3, 4]])


def test_transform_non_dataframe_raises(fitted):
    with pytest.raises(TypeError, match="pandas DataFrame"):
        fitted.transform(np.array([[1.0, 2.0]]))


def test_missing_columns_raises():
    feat = MatchingGiftFeaturizer()
    with pytest.raises(ValueError, match="missing required columns"):
        feat.fit(pd.DataFrame({"employer": ["Boeing"]}))


def test_clone_drops_fitted_state(X_donors, ratios):
    feat = MatchingGiftFeaturizer(match_ratios=ratios).fit(X_donors)
    cloned = clone(feat)
    assert not hasattr(cloned, "match_ratios_")
