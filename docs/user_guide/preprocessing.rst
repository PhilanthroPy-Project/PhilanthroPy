Preprocessing Deep Dive
=======================

PhilanthroPy's preprocessing module is designed to transform messy clinical and CRM data into a format that machine learning models can learn from, while strictly preventing data leakage.

.. currentmodule:: philanthropy.preprocessing

Data Cleaning & Standardization
-------------------------------

CRMCleaner
~~~~~~~~~~
The foundation of any PhilanthroPy pipeline.

**Key Features:**
*   **Type Coercion**: Automatically converts dates (ISO-8601) and amounts.
*   **Privacy Firewall**: Identifies and drops PII columns (MRN, SSN, DOB) based on naming heuristics.
*   **Whitespace Scrubbing**: Cleans raw CSV/XLSX exports.
*   **Integrated Imputation**: Can wrap a ``WealthScreeningImputer`` to ensure imputation statistics are learned *only* on training data (preventing leakage).

**Advanced Usage:**
If your CRM uses non-standard column names for PII, pass them to ``id_cols_to_drop``.

Wealth Intelligence
-------------------

WealthScreeningImputer
~~~~~~~~~~~~~~~~~~~~~~
Handling the "Missingness Gap" in 3rd party wealth data.

**Strategies:**
*   ``median``: Fill with training-set median (Robust to outliers).
*   ``mean``: Fill with training-set mean.
*   ``zero``: Assume missing data implies zero wealth (Aggressive).

**Missingness Indicators:**
By setting ``add_indicator=True``, the imputer appends a ``<col>__was_missing`` column. This is critical because for many vendors, **missing data is a signal** (e.g., the donor might be a patient who hasn't been screened yet or is new to the database).

WealthPercentileTransformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Normalizing skewed wealth distributions.

Fundraising data is notoriously right-skewed (a few donors have $100M+ net worth). Using raw dollars in a linear model can lead to poor performance. This transformer converts dollars to ranks (0-100).

Medical Philanthropy (Grateful Patients)
----------------------------------------

GratefulPatientFeaturizer
~~~~~~~~~~~~~~~~~~~~~~~~~
Quantifying clinical engagement.

Not all hospital visits are equal for philanthropy. A visit to Oncology or Cardiology often correlates more strongly with grateful patient giving than a visit to urgent care.

**Clinical Gravity Scores:**
You can provide a ``service_line_weights`` dictionary to prioritize specific clinical areas. By default, it uses a balanced "Propensity Map" derived from dozens of AMC datasets.

SolicitationWindowTransformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The "Golden Window" of engagement.

**Midpoint Proximity**: The ``window_score`` feature reaches its maximum (1.0) at the exact center of the your defined solicitation window (e.g., at 15 months if your window is 6–24 months). This allows your model to prioritize patients who are in that optimal "sweet spot."

Temporal Engineering
--------------------

RFMTransformer
~~~~~~~~~~~~~~
* **Recency**: Days since last gift.
* **Frequency**: Count of gifts in life or lookback period.
* **Monetary**: Total or average gift amount.

This transformer computes these three fundamental pillars of fundraising analytics in a single step.
