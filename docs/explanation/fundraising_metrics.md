# Fundraising metrics

Accuracy and R-Squared measure a model. They don't measure whether fundraising worked. PhilanthroPy provides metrics that speak the language of development officers and CFOs.

## Lifetime value analysis

### Donor lifetime value (DLV)

The North Star metric for long-term donor health.

The `donor_lifetime_value` function computes the Net Present Value (NPV) of all future expected gifts from a donor.

**The formula.** PhilanthroPy uses a discounted flow model:

```math
DLV = \frac{G \times (1 - (1+d)^{-L})}{d}
```

Where:

* **G**: Average Annual Giving.
* **d**: Discount Rate (Time Value of Money).
* **L**: Donor Lifespan (often derived as `1 / (1 - Retention Rate)`).

**Why it matters.** DLV lets you justify higher Acquisition Costs for high-value segments — like grateful patients — even when the initial gift is small.

## Efficiency & ROI

### Donor acquisition cost (DAC)

The cost of bringing in one new donor.

**Calculation.** Total Campaign Spend / Number of New Donors Acquired.

**Strategic benchmarking.** Ideally, a donor's Year 1 gift should cover their DAC, or their 3-year DLV should be at least 3x their DAC.

### Retention rate

The percentage of donors from Period A who gave again in Period B.

Retention is the single biggest lever for total revenue growth. PhilanthroPy's metrics module lets you segment retention by acquisition channel, so you can see which sources — Direct Mail versus High-Touch Events, say — produce the most loyal donors.
