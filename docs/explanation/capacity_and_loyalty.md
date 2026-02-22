# Capacity and Loyalty Models

PhilanthroPy models are specialized estimators for the fundraising domain. They follow the scikit-learn estimator interface but add fundraising-specific enhancements.

## Capacity & Potential

### The Share of Wallet Concept
Many donors give $1,000 but have the capacity to give $1,000,000. Finding this gap is critical. The `ShareOfWalletScorer` computes a normalized score (0 to 1) comparing an estimated capacity capability to modeled or known wealth. It effectively tells a gift officer how much room there is to grow a donor's giving.

### Capacity Tiers
It's easier for human gift officers to interpret categories rather than numbers. PhilanthroPy can automatically categorize donors into operational tiers:
* **Principal**: High Share of Wallet potential.
* **Major**: Medium Share of Wallet potential.
* **Leadership**: Currently maxing out given known capacity constraints.

## Retention & Loyalty

### Lapse Prediction
Identifying donors at risk of "walking out the back door." Models focusing on churn specifically zero in on the **12-to-24 month window** post-gift, which is statistically the high-risk "cliff" for donor attrition.

### Moves Management
Development officers manage large portfolios of prospects. Moves Management involves transitioning a donor from *Identify* to *Qualify*, *Cultivate*, *Solicit*, and *Steward*.

Unlike a binary "will they give?" model, PhilanthroPy's `MovesManagementClassifier` is a multi-class predictive tool that infers the optimal lifecycle stage of a donor based on recent engagement patterns. This optimizes the workflows of entire fundraising departments.
