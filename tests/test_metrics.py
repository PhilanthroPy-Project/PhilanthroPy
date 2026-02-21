"""
tests/test_metrics.py
"""

import math
import numpy as np
import pytest
from philanthropy.metrics import donor_retention_rate, donor_acquisition_cost


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
