"""
tests/test_fairness.py: disparate-impact diagnostics.
"""

import math
import numpy as np
import pytest
from philanthropy.metrics import (
    demographic_parity_difference,
    disparate_impact_ratio,
    selection_rate_by_group,
)


def test_selection_rate_by_group_basic():
    rates = selection_rate_by_group([1, 0, 1, 1], ["a", "a", "b", "b"])
    assert math.isclose(rates["a"], 0.5)
    assert math.isclose(rates["b"], 1.0)


def test_disparate_impact_ratio_basic():
    ratio = disparate_impact_ratio([1, 0, 1, 1], ["a", "a", "b", "b"])
    assert math.isclose(ratio, 0.5)  # 0.5 / 1.0


def test_disparate_impact_ratio_parity_is_one():
    assert disparate_impact_ratio([1, 1], ["a", "b"]) == 1.0


def test_disparate_impact_ratio_single_group_is_one():
    assert disparate_impact_ratio([1, 0, 1], ["a", "a", "a"]) == 1.0


def test_disparate_impact_ratio_no_selection_is_one():
    # No one selected in any group -> no disparity to measure.
    assert disparate_impact_ratio([0, 0, 0], ["a", "b", "c"]) == 1.0


def test_disparate_impact_ratio_custom_pos_label():
    ratio = disparate_impact_ratio(
        ["yes", "no", "yes", "yes"], ["a", "a", "b", "b"], pos_label="yes"
    )
    assert math.isclose(ratio, 0.5)


def test_fairness_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        disparate_impact_ratio([1, 0], ["a", "a", "b"])


def test_fairness_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        disparate_impact_ratio([], [])


def test_fairness_accepts_numpy_arrays():
    ratio = disparate_impact_ratio(
        np.array([1, 0, 1, 1]), np.array([0, 0, 1, 1])
    )
    assert math.isclose(ratio, 0.5)


def test_demographic_parity_difference_basic():
    diff = demographic_parity_difference([1, 0, 1, 1], ["a", "a", "b", "b"])
    assert math.isclose(diff, 0.5)  # 1.0 - 0.5


def test_demographic_parity_difference_parity_is_zero():
    assert demographic_parity_difference([1, 1], ["a", "b"]) == 0.0


def test_demographic_parity_difference_single_group_is_zero():
    assert demographic_parity_difference([1, 0, 1], ["a", "a", "a"]) == 0.0


def test_demographic_parity_difference_small_rates_not_masked_by_ratio():
    # 0.01 vs 0.02 gives a ratio of 0.5 (looks severe) but a difference of
    # 0.01 (looks negligible); the two diagnostics answer different questions.
    y_pred = [1] + [0] * 99 + [1, 1] + [0] * 98
    groups = ["a"] * 100 + ["b"] * 100
    assert math.isclose(disparate_impact_ratio(y_pred, groups), 0.5)
    assert math.isclose(demographic_parity_difference(y_pred, groups), 0.01)


def test_demographic_parity_difference_custom_pos_label():
    diff = demographic_parity_difference(
        ["yes", "no", "yes", "yes"], ["a", "a", "b", "b"], pos_label="yes"
    )
    assert math.isclose(diff, 0.5)
