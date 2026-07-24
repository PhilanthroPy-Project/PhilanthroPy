"""
tests/test_fairness.py — disparate-impact diagnostics.
"""

import math
import numpy as np
import pytest
from philanthropy.metrics import disparate_impact_ratio, selection_rate_by_group


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
