# Fundraising metrics

Accuracy and R-Squared measure a model. They don't measure whether fundraising worked. PhilanthroPy provides metrics that speak the language of development officers and CFOs.

## Lifetime value analysis

### Donor lifetime value (DLV)

The North Star metric for long-term donor health.

The `donor_lifetime_value` function computes the Net Present Value (NPV) of all future expected gifts from a donor.

**The formulas.** There are two, because a *known* horizon and an *uncertain*
lifetime are different problems.

Over a fixed horizon of `L` years, the NPV of an ordinary annuity:

```math
DLV = \frac{G \times (1 - (1+d)^{-L})}{d}
```

Given an annual retention rate `r` instead, the donor's lifetime is geometric
with `E[L] = 1 / (1 - r)`, and the *expected* NPV is:

```math
E[DLV] = \frac{G}{1 + d - r}
```

Where:

* **G**: Average Annual Giving.
* **d**: Discount Rate (Time Value of Money).
* **L**: Donor Lifespan, when it is known in advance.
* **r**: Annual Retention Rate, when the lifespan is uncertain.

**Do not substitute `E[L]` into the first formula.** The annuity is concave in
`L`, so by Jensen's inequality `NPV(E[L]) >= E[NPV(L)]`: plugging the expected
lifespan into the fixed-horizon formula overstates lifetime value every time,
by 8.2% at `r = 0.8, d = 0.05` and by 22.9% at `r = 0.9, d = 0.10`. The error is
one-signed, so it never averages out across a portfolio. PhilanthroPy made
exactly this mistake before version 0.7.0.

**Why it matters.** DLV lets you justify higher Acquisition Costs for high-value segments (like grateful patients) even when the initial gift is small.

## Efficiency & ROI

### Donor acquisition cost (DAC)

The cost of bringing in one new donor.

**Calculation.** Total Campaign Spend / Number of New Donors Acquired.

**Strategic benchmarking.** Ideally, a donor's Year 1 gift should cover their DAC, or their 3-year DLV should be at least 3x their DAC.

### Retention rate

The percentage of donors from Period A who gave again in Period B.

Retention is the single biggest lever for total revenue growth. PhilanthroPy's metrics module lets you segment retention by acquisition channel, so you can see which sources (Direct Mail versus High-Touch Events, say) produce the most loyal donors.
