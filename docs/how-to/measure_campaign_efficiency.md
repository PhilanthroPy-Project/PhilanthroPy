# Measure campaign efficiency

Four numbers answer "was this campaign worth running?": cost per dollar raised, net ROI, donor acquisition cost, and how concentrated the revenue was. `philanthropy.metrics` computes all four from plain totals; no estimator required.

## Pass arguments by keyword

The efficiency functions do not share an argument order: `cost_per_dollar_raised` takes expense first, `fundraising_roi` takes raised first. Swapping them is silently accepted and returns a plausible wrong number. These three are **keyword-only** as of 0.7.0, so a positional call is a `TypeError` rather than a plausible wrong number.

```python
from philanthropy.metrics import (
    cost_per_dollar_raised,
    donor_acquisition_cost,
    fundraising_roi,
)

total_raised = 1_000_000.0
total_fundraising_expense = 250_000.0
new_donors_acquired = 500

cpdr = cost_per_dollar_raised(
    total_fundraising_expense=total_fundraising_expense,
    total_raised=total_raised,
)
roi = fundraising_roi(
    total_raised=total_raised,
    total_fundraising_expense=total_fundraising_expense,
)
cac = donor_acquisition_cost(
    total_fundraising_expense=total_fundraising_expense,
    new_donors_acquired=new_donors_acquired,
)

print(f"cost per dollar raised: ${cpdr:.2f}")
print(f"net ROI:                {roi:.1f}x")
print(f"donor acquisition cost: ${cac:,.0f}")

assert cpdr == 0.25
assert roi == 3.0            # NET return: (raised - expense) / expense
assert cac == 500.0
```

`fundraising_roi` is **net**, not gross: `(raised − expense) / expense`. A campaign that exactly breaks even scores `0.0`, not `1.0`. It is therefore one less than the reciprocal of cost per dollar raised.

```python
assert fundraising_roi(total_raised=100_000.0, total_fundraising_expense=100_000.0) == 0.0
assert roi == 1.0 / cpdr - 1.0
```

Both functions return `np.inf` rather than raising when their denominator is zero, so a campaign with revenue and no recorded spend is safe to sort and plot.

## Retention

`donor_retention_rate` is the share of last period's donors who gave again. It takes two donor-id collections, not counts.

```python
from philanthropy.metrics import donor_retention_rate

prior_year = [101, 102, 103, 104, 105]
this_year = [103, 104, 105, 106, 107]

print(f"retention: {donor_retention_rate(this_year, prior_year):.0%}")
assert donor_retention_rate(this_year, prior_year) == 0.6
```

## How concentrated was the revenue?

A campaign that raised its target from three donors is a different risk profile from one that raised it from three thousand. `gift_concentration_gini` and `top_donor_share` quantify that.

```python
import numpy as np

from philanthropy.metrics import gift_concentration_gini, top_donor_share

rng = np.random.default_rng(0)
broad = rng.lognormal(6, 0.4, 1000)                       # many similar gifts
concentrated = np.concatenate([rng.lognormal(5, 0.3, 990), rng.lognormal(13, 0.5, 10)])

for label, gifts in [("broad", broad), ("concentrated", concentrated)]:
    print(
        f"{label:13s} gini={gift_concentration_gini(gifts):.3f} "
        f"top-1%={top_donor_share(gifts, top_fraction=0.01):.1%}"
    )

assert gift_concentration_gini(concentrated) > gift_concentration_gini(broad)
```

The Gini coefficient is `0.0` under perfect equality and approaches `1.0` as one donor holds everything; for `n` donors where exactly one gives, it is exactly `(n-1)/n`.

```python
assert gift_concentration_gini([1, 1, 1, 1]) == 0.0
assert gift_concentration_gini([0, 0, 0, 4]) == 0.75
assert top_donor_share([500.0] + [500.0 / 9.0] * 9, top_fraction=0.1) == 0.5
```

## Long-run value of an acquired donor

Acquisition cost only means something against what the donor is worth. `donor_lifetime_value` has two modes: the net present value of a discounted annuity over a **fixed horizon**, or the *expected* net present value over a **geometric lifetime** implied by a retention rate. They are different calculations, not the same one with a substituted lifespan.

```python
from philanthropy.metrics import donor_lifetime_value

fixed = donor_lifetime_value(250.0, 10, discount_rate=0.05)
from_retention = donor_lifetime_value(250.0, 999, discount_rate=0.05, retention_rate=0.8)

print(f"LTV over 10 years:        ${fixed:,.0f}")
print(f"LTV at 80% retention:     ${from_retention:,.0f}")

# 80% annual retention does imply a 1 / (1 - 0.8) = 5-year expected lifespan,
# but the expected NPV is NOT the 5-year annuity. The annuity is concave in the
# lifespan, so by Jensen's inequality plugging in the mean overstates value:
# E[NPV(L)] < NPV(E[L]). The retention mode uses the closed form G / (1 + d - r).
assert from_retention == 250.0 / (1 + 0.05 - 0.8)
assert from_retention < donor_lifetime_value(250.0, 5, discount_rate=0.05)
assert fixed > cac  # the acquisition pays for itself
```

## A real registry, and what it cannot tell you

`load_ciob_fundraising` ships a real open-government dataset: every not-for-profit a New York City agency reported soliciting for, under the Conflicts of Interest Board's disclosure mandate.

```python
from philanthropy.datasets import load_ciob_fundraising

ciob = load_ciob_fundraising()
print(ciob.shape, sorted(ciob["year"].unique()))
print(ciob["name_of_not_for_profit"].value_counts().head(5))
```

It is an **affiliation registry**, one row per `(year, agency, nonprofit)` link, with no gift amounts, donor records, or engagement labels. It supports honest questions about who solicits for whom:

```python
breadth = ciob.groupby("agency")["name_of_not_for_profit"].nunique().sort_values()
print(breadth.tail(5))
assert set(ciob.columns) == {"year", "agency", "name_of_not_for_profit"}
```

It does **not** support the efficiency metrics above or the RFM/propensity modelling elsewhere in the library; there are no dollars in it. Use `generate_synthetic_donor_data` for those, or your own CRM export.
