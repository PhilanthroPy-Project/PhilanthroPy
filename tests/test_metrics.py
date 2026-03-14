"""
tests/test_metrics.py
"""

import math
import numpy as np
import pytest
from philanthropy.metrics import (
    donor_retention_rate,
    donor_acquisition_cost,
    donor_lifetime_value,
)


def test_retention_rate_basic():
    prior = ["A", "B", "C", "D"]
    current = ["A", "B", "E"]
    rate = donor_retention_rate(current, prior)
    assert math.isclose(rate, 0.5)


def test_retention_rate_perfect():
    donors = ["X", "Y", "Z"]
    assert donor_retention_rate(donors, donors) == 1.0


def test_retention_rate_zero():
    assert donor_retention_rate(["A"], ["B", "C"]) == 0.0


def test_retention_rate_empty_prior():
    assert donor_retention_rate(["A", "B"], []) == 0.0


def test_dac_basic():
    assert math.isclose(donor_acquisition_cost(50_000, 200), 250.0)


def test_dac_zero_donors():
    assert donor_acquisition_cost(10_000, 0) == np.inf


def test_retention_rate_empty_current():
    assert donor_retention_rate([], ["A", "B", "C"]) == 0.0


def test_retention_rate_identical_sets():
    donors = ["A", "B", "C"]
    assert donor_retention_rate(donors, donors) == 1.0


def test_retention_rate_completely_disjoint():
    assert donor_retention_rate(["X", "Y"], ["A", "B", "C"]) == 0.0


def test_ltv_basic_npv():
    avg, lifespan, r = 1000.0, 5.0, 0.05
    ltv = donor_lifetime_value(avg, lifespan, discount_rate=r)
    expected = 1000 * (1 - (1.05**-5)) / 0.05
    assert math.isclose(ltv, expected)


def test_ltv_discount_rate_zero():
    ltv = donor_lifetime_value(1000.0, 10.0, discount_rate=0.0)
    assert math.isclose(ltv, 1000.0 * 10.0)


def test_ltv_retention_rate_overrides_lifespan():
    ltv = donor_lifetime_value(
        1000.0, lifespan_years=5.0, retention_rate=0.5, discount_rate=0.0
    )
    expected_lifespan = 1 / (1 - 0.5)
    assert math.isclose(ltv, 1000.0 * expected_lifespan)


def test_ltv_retention_rate_one_perpetuity():
    ltv = donor_lifetime_value(1000.0, lifespan_years=5.0, retention_rate=1.0)
    assert ltv == float('inf')


def test_ltv_zero_average_donation():
    assert donor_lifetime_value(0.0, 10.0) == 0.0


def test_ltv_negative_discount_rate_raises():
    with pytest.raises(ValueError, match="discount_rate"):
        donor_lifetime_value(1000.0, 10.0, discount_rate=-0.05)


def test_dac_negative_expense_returns_negative():
    result = donor_acquisition_cost(-1000, 50)
    assert result < 0


def test_ltv_retention_rate_negative_raises():
    with pytest.raises(ValueError, match="retention_rate"):
        donor_lifetime_value(1000.0, 10.0, retention_rate=-0.1)


def test_ltv_discount_rate_zero_with_retention():
    ltv = donor_lifetime_value(
        500.0, lifespan_years=3.0, retention_rate=0.8, discount_rate=0.0
    )
    expected_lifespan = 1 / (1 - 0.8)
    assert math.isclose(ltv, 500.0 * expected_lifespan)
