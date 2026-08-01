"""Tests for donor-base concentration and campaign-efficiency metrics."""

import numpy as np
import pytest

from philanthropy.metrics import (
    gift_concentration_gini,
    top_donor_share,
    cost_per_dollar_raised,
    fundraising_roi,
)


def test_gini_perfect_equality_is_zero():
    assert gift_concentration_gini([100, 100, 100, 100]) == pytest.approx(0.0, abs=1e-9)


def test_gini_high_concentration():
    # One donor holds everything → strong concentration.
    assert gift_concentration_gini([0, 0, 0, 100]) == pytest.approx(0.75)


def test_gini_empty_and_all_zero_return_zero():
    assert gift_concentration_gini([]) == 0.0
    assert gift_concentration_gini([0, 0, 0]) == 0.0


def test_gini_drops_nan():
    assert gift_concentration_gini([100, 100, np.nan]) == pytest.approx(0.0, abs=1e-9)


def test_gini_rejects_negative():
    with pytest.raises(ValueError):
        gift_concentration_gini([1, -2, 3])


def test_top_donor_share_top_decile():
    # 10 donors; top 10% = 1 donor holding 91 of 100 total.
    amounts = [1] * 9 + [91]
    assert top_donor_share(amounts, 0.1) == pytest.approx(0.91)


def test_top_donor_share_full_fraction_is_one():
    assert top_donor_share([1, 2, 3], 1.0) == pytest.approx(1.0)


def test_top_donor_share_rejects_bad_fraction():
    with pytest.raises(ValueError):
        top_donor_share([1, 2], 0.0)
    with pytest.raises(ValueError):
        top_donor_share([1, 2], 1.5)


def test_top_donor_share_empty_and_zero():
    assert top_donor_share([]) == 0.0
    assert top_donor_share([0, 0]) == 0.0


def test_cost_per_dollar_raised():
    assert cost_per_dollar_raised(total_fundraising_expense=20, total_raised=100) == pytest.approx(0.2)
    assert cost_per_dollar_raised(total_fundraising_expense=20, total_raised=0) == np.inf


def test_fundraising_roi():
    assert fundraising_roi(total_raised=400, total_fundraising_expense=100) == pytest.approx(3.0)
    assert fundraising_roi(total_raised=100, total_fundraising_expense=0) == np.inf
