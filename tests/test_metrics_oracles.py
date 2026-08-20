"""tests/test_metrics_oracles.py

Closed-form oracles for the money metrics.

The rest of the metrics suite checks ranges, monotonicity and error paths, all
of which a subtly wrong formula still satisfies. These assert each function
against an independent source: the textbook definition of the Gini coefficient,
the discounted-annuity NPV written out term by term, and the EEOC's own
four-fifths-rule worked example.
"""

import numpy as np
import pytest

from philanthropy.metrics import (
    cost_per_dollar_raised,
    disparate_impact_ratio,
    donor_acquisition_cost,
    donor_lifetime_value,
    donor_retention_rate,
    fundraising_roi,
    gift_concentration_gini,
    top_donor_share,
)


# ---------------------------------------------------------------------------
# gift_concentration_gini
# ---------------------------------------------------------------------------

def test_gini_is_zero_under_perfect_equality():
    assert gift_concentration_gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert gift_concentration_gini([500.0] * 9) == pytest.approx(0.0)


def test_gini_of_a_single_giver_is_n_minus_one_over_n():
    # One donor holds the entire pot: the maximum a sample Gini can reach.
    for n in (2, 4, 10, 100):
        amounts = [0.0] * (n - 1) + [4.0]
        assert gift_concentration_gini(amounts) == pytest.approx((n - 1) / n)


def test_gini_matches_the_mean_absolute_difference_definition():
    # Population Gini: G = sum_i sum_j |x_i - x_j| / (2 * n^2 * mean(x)).
    # This is the convention the implementation uses, consistent with the
    # (n-1)/n maximum asserted above, not the sample-corrected n/(n-1) variant.
    rng = np.random.default_rng(0)
    x = rng.uniform(10.0, 10_000.0, 40)
    n = x.size
    mad = np.abs(x[:, None] - x[None, :]).sum()
    expected = mad / (2.0 * n * n * x.mean())
    assert gift_concentration_gini(x) == pytest.approx(expected, rel=1e-9)


def test_gini_is_scale_invariant():
    x = [100.0, 250.0, 900.0, 4000.0]
    assert gift_concentration_gini(x) == pytest.approx(
        gift_concentration_gini([v * 1000 for v in x])
    )


def test_top_donor_share_of_a_known_split():
    # Ten donors, one gives 500 and nine give 500/9 each: top 10% == 50%.
    amounts = [500.0] + [500.0 / 9.0] * 9
    assert top_donor_share(amounts, top_fraction=0.1) == pytest.approx(0.5)
    assert top_donor_share(amounts, top_fraction=1.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# donor_lifetime_value
# ---------------------------------------------------------------------------

def test_ltv_matches_a_discounted_annuity_summed_term_by_term():
    # NPV of an ordinary annuity: sum_{t=1..n} a / (1 + r)^t
    a, r, n = 250.0, 0.05, 10
    expected = sum(a / (1 + r) ** t for t in range(1, n + 1))
    assert donor_lifetime_value(a, n, discount_rate=r) == pytest.approx(expected)


@pytest.mark.parametrize("rate", [0.0, 0.03, 0.07, 0.12])
def test_ltv_annuity_holds_across_discount_rates(rate):
    a, n = 1000.0, 7
    if rate == 0.0:
        expected = a * n
    else:
        expected = sum(a / (1 + rate) ** t for t in range(1, n + 1))
    assert donor_lifetime_value(a, n, discount_rate=rate) == pytest.approx(expected)


def test_ltv_geometric_lifetime_is_not_the_annuity_at_the_mean_lifespan():
    # This test previously asserted the two were EQUAL, which is the bug. An 80%
    # retention rate does imply a 1/(1-0.8) = 5-year expected lifespan, but the
    # annuity is concave in L, so by Jensen NPV(E[L]) > E[NPV(L)] and
    # substituting the mean lifespan overstates value. Closed form: m/(1+d-r).
    a, d, r = 400.0, 0.05, 0.8
    by_retention = donor_lifetime_value(a, 999, discount_rate=d, retention_rate=r)
    annuity_at_mean = donor_lifetime_value(a, 5, discount_rate=d)

    assert by_retention == pytest.approx(a / (1 + d - r))
    assert by_retention < annuity_at_mean
    # The overstatement is ~8.2% at these parameters, not a rounding artefact.
    assert annuity_at_mean / by_retention == pytest.approx(1.082, abs=1e-3)


def test_ltv_geometric_lifetime_matches_the_term_by_term_expectation():
    # E[NPV] = m * sum_{t>=1} P(alive at t) / (1+d)^t, with P(alive at t) = r^(t-1).
    a, d, r = 250.0, 0.07, 0.75
    expected = sum(a * r ** (t - 1) / (1 + d) ** t for t in range(1, 4000))
    assert donor_lifetime_value(a, 999, discount_rate=d, retention_rate=r) == (
        pytest.approx(expected)
    )


def test_ltv_geometric_and_fixed_horizon_agree_at_zero_retention():
    # r = 0 is one gift, one year out. Both modes must give m/(1+d).
    a, d = 100.0, 0.05
    assert donor_lifetime_value(a, 999, discount_rate=d, retention_rate=0.0) == (
        pytest.approx(a / (1 + d))
    )
    assert donor_lifetime_value(a, 1, discount_rate=d) == pytest.approx(a / (1 + d))


def test_ltv_never_lapsing_donor_is_a_perpetuity():
    # m / d with a discount rate, unbounded only without one.
    assert donor_lifetime_value(100.0, 10, retention_rate=1.0) == pytest.approx(2000.0)
    assert donor_lifetime_value(
        100.0, 10, discount_rate=0.0, retention_rate=1.0
    ) == float("inf")


def test_ltv_rejects_retention_rate_above_one():
    with pytest.raises(ValueError, match="cannot exceed 1"):
        donor_lifetime_value(100.0, 10, retention_rate=1.5)


# ---------------------------------------------------------------------------
# disparate_impact_ratio: the EEOC four-fifths worked example
# ---------------------------------------------------------------------------

def test_disparate_impact_matches_the_eeoc_four_fifths_example():
    # The EEOC's canonical illustration: 80/100 of group A selected (0.80) and
    # 40/100 of group B (0.40) gives a ratio of 0.50, adverse impact.
    y_pred = [1] * 80 + [0] * 20 + [1] * 40 + [0] * 60
    groups = ["A"] * 100 + ["B"] * 100
    assert disparate_impact_ratio(y_pred, groups) == pytest.approx(0.5)


def test_disparate_impact_of_exactly_four_fifths_is_the_threshold():
    # 50/100 (0.50) versus 40/100 (0.40) is exactly 0.8, the EEOC boundary.
    y_pred = [1] * 50 + [0] * 50 + [1] * 40 + [0] * 60
    groups = ["A"] * 100 + ["B"] * 100
    assert disparate_impact_ratio(y_pred, groups) == pytest.approx(0.8)


def test_disparate_impact_of_exact_parity_is_one():
    y_pred = [1] * 30 + [0] * 70 + [1] * 30 + [0] * 70
    groups = ["A"] * 100 + ["B"] * 100
    assert disparate_impact_ratio(y_pred, groups) == pytest.approx(1.0)


def test_disparate_impact_is_min_over_max_across_three_groups():
    # Rates 0.9 / 0.6 / 0.3 → 0.3 / 0.9.
    y_pred = ([1] * 9 + [0]) + ([1] * 6 + [0] * 4) + ([1] * 3 + [0] * 7)
    groups = ["A"] * 10 + ["B"] * 10 + ["C"] * 10
    assert disparate_impact_ratio(y_pred, groups) == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# Campaign efficiency: arithmetic identities
# ---------------------------------------------------------------------------

def test_campaign_efficiency_metrics_are_reciprocal_and_consistent():
    raised, expense, new_donors = 1_000_000.0, 250_000.0, 500

    cpdr = cost_per_dollar_raised(
        total_fundraising_expense=expense, total_raised=raised
    )
    roi = fundraising_roi(
        total_raised=raised, total_fundraising_expense=expense
    )
    assert cpdr == pytest.approx(0.25)
    # fundraising_roi is NET return, (raised - expense) / expense, so it is
    # one less than the reciprocal of cost-per-dollar-raised.
    assert roi == pytest.approx(3.0)
    assert roi == pytest.approx(1.0 / cpdr - 1.0)
    assert donor_acquisition_cost(
        total_fundraising_expense=expense, new_donors_acquired=new_donors
    ) == pytest.approx(500.0)


def test_fundraising_roi_is_zero_at_break_even():
    assert fundraising_roi(
        total_raised=100_000.0, total_fundraising_expense=100_000.0
    ) == pytest.approx(0.0)


@pytest.mark.parametrize("fn, args", [
    (donor_acquisition_cost, (50_000.0, 200)),
    (cost_per_dollar_raised, (250_000.0, 1_000_000.0)),
    (fundraising_roi, (1_000_000.0, 250_000.0)),
])
def test_money_metrics_reject_positional_arguments(fn, args):
    # These three do not share an argument order: cost_per_dollar_raised takes
    # expense first, fundraising_roi takes raised first. Positional calls used
    # to be silently accepted and return a plausible wrong number. Keyword-only
    # since 0.7.0.
    with pytest.raises(TypeError, match="positional"):
        fn(*args)


def test_donor_retention_rate_is_the_overlap_over_the_prior_cohort():
    prior = [1, 2, 3, 4, 5]
    current = [3, 4, 5, 6, 7]        # 3 of the 5 prior donors renewed
    assert donor_retention_rate(current, prior) == pytest.approx(0.6)
