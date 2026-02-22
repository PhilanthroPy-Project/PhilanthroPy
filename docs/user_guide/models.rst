Predictive Modeling
===================

PhilanthroPy models are specialized estimators for the fundraising domain. They follow the scikit-learn estimator interface (fit/predict/predict_proba) but add fundraising-specific enhancements.

.. currentmodule:: philanthropy.models

Propensity & Affinity
---------------------

DonorPropensityModel
~~~~~~~~~~~~~~~~~~~~
A Random Forest ensemble tuned for "Major Gift" classification (typically defined as gifts ≥ $25k).

**Affinity Scoring (0-100):**
While a typical classifier returns a probability (e.g., 0.12), the ``DonorPropensityModel`` provides ``predict_affinity_score()``. This maps probabilities to a 0–100 scale using a calibration curve. In the fundraising world, a score of "85" is much more actionable for a gift officer than "0.08".

MajorGiftClassifier
~~~~~~~~~~~~~~~~~~~
A high-performance Gradient Boosting model (based on ``HistGradientBoostingClassifier``).

**NaN-Native Processing:**
Unlike many models, the ``MajorGiftClassifier`` handles missing wealth screening features internally. It learns the optimal branch to take for missing values, which often captures the subtle "signal of absence" better than explicit imputation.

Capacity & Potential
--------------------

ShareOfWalletRegressor
~~~~~~~~~~~~~~~~~~~~~~
Estimating how much *more* a donor could give.

Many donors give $1,000 but have the capacity to give $1,000,000. The ``ShareOfWalletRegressor`` uses regression to estimate total giving capacity and produces a ``capacity_ratio`` (Current Giving / Estimated Total Capacity).

**Capacity Tiers:**
The regressor automatically categorizes donors into tiers:
*   **Principal**: High SoW potential.
*   **Major**: Medium SoW potential.
*   **Leadership**: Current baseline levels.

Retention & Loyalty
-------------------

LapsePredictor
~~~~~~~~~~~~~~
Identifying donors at risk of "walking out the back door."

This model is trained on historic donor churn data. It focuses on the **12-to-24 month window** post-gift, which is the high-risk zone for donor attrition.

Moves Management
----------------

MovesManagementClassifier
~~~~~~~~~~~~~~~~~~~~~~~~~
Supporting the "Officer Pipeline."

Unlike a binary "will they give?" model, this is a multi-class classifier that predicts which stage of the moves management cycle a donor should be in (e.g., *Qualification* vs. *Cultivation*). It helps managers balance gift officer portfolios.
