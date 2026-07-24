# Capacity and loyalty models

PhilanthroPy's models are estimators built for fundraising. They follow the scikit-learn estimator interface and add behavior specific to the domain.

## Capacity & Potential

### The Share of Wallet Concept

Many donors give $1,000 but have the capacity to give $1,000,000. The gap between the two is where major-gift work happens. `ShareOfWalletScorer` computes a normalized score from 0 to 1 that compares estimated capacity to modeled or known wealth. It tells a gift officer how much room there is to grow a donor's giving.

Gift officers often need to justify the score to leadership. It is defined as:

```text
SoW = predicted_capacity / estimated_total_philanthropic_capacity
```

This ratio is rooted in industry standards and aligns with benchmarks established by organizations like Blackbaud and EAB.

### Capacity Tiers

Categories are easier to act on than raw numbers, so PhilanthroPy sorts donors into operational tiers automatically:

* **Principal**: High Share of Wallet potential.
* **Major**: Medium Share of Wallet potential.
* **Leadership**: Currently maxing out given known capacity constraints.

## Retention & Loyalty

### Lapse Prediction

Find donors at risk of "walking out the back door." Churn models focus on the **12-to-24 month window** post-gift, statistically the high-risk "cliff" for donor attrition.

### Moves Management

Development officers manage large portfolios of prospects. Moves Management transitions a donor from *Identify* to *Qualify*, *Cultivate*, *Solicit*, and *Steward*.

`MovesManagementClassifier` is not a binary "will they give?" model. It is a multi-class predictive tool that infers a donor's optimal lifecycle stage from recent engagement patterns, which optimizes the workflows of entire fundraising departments.
