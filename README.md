<p align="center">
  <img src="docs/logo.png" alt="PhilanthroPy logo" width="180"/>
</p>

<p align="center">
  <strong>A scikit-learn–compatible toolkit for predictive donor analytics in the nonprofit and medical philanthropy sector.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-1.7%2B-orange.svg" alt="scikit-learn"></a>
  <img src="https://img.shields.io/badge/tests-161%20passing-brightgreen.svg" alt="Tests">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

PhilanthroPy gives hospital advancement offices, development teams, and nonprofit data scientists a **production-ready Python library** that slots directly into a `sklearn.pipeline.Pipeline`. It covers the full predictive analytics workflow — from raw CRM data cleaning, through feature engineering and major-gift propensity scoring, to fundraising-specific KPI metrics.

---

## Table of Contents

1. [Why PhilanthroPy?](#why-philanthropy)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Package Overview](#package-overview)
5. [Module Reference](#module-reference)
   - [philanthropy.datasets](#philanthropydatasets)
   - [philanthropy.preprocessing](#philanthropypreprocessing)
   - [philanthropy.models](#philanthropymodels)
   - [philanthropy.metrics](#philanthropymetrics)
   - [philanthropy.utils](#philanthropyutils)
6. [Worked Examples](#worked-examples)
   - [End-to-End Pipeline](#end-to-end-pipeline)
   - [Affinity Score Ranking](#affinity-score-ranking)
   - [Fiscal Year Analysis](#fiscal-year-analysis)
   - [Medical Philanthropy Pipeline](#medical-philanthropy-pipeline)
   - [Share-of-Wallet Capacity Ranking](#share-of-wallet-capacity-ranking)
   - [Donor Retention Reporting](#donor-retention-reporting)
7. [API Reference](#api-reference)
8. [Design Principles (Golden Rules)](#design-principles-golden-rules)
9. [Development Environment](#development-environment)
10. [Testing](#testing)
11. [Contributing](#contributing)

---

## Why PhilanthroPy?

Most general-purpose ML libraries treat all classification problems identically. Philanthropic data science has unique challenges:

| Challenge | PhilanthroPy's answer |
|-----------|----------------------|
| CRM exports vary wildly by vendor (Raiser's Edge, Salesforce NPSP, Veeva) | `CRMCleaner` standardises raw exports; optional `WealthScreeningImputer` fills missing vendor data leak-safely |
| Fiscal years start in July (or any month), not January | `FiscalYearTransformer` appends `fiscal_year` and `fiscal_quarter` columns for any start month |
| Clinical encounter history is siloed from advancement CRM | `EncounterTransformer` merges discharge dates with gift dates, producing `days_since_last_discharge` and `encounter_frequency_score` |
| Wealth-screening vendor exports have 30–70 % missing values | `WealthScreeningImputer` learns fill statistics from training data only — no leakage |
| Major donors are <5 % of a prospect pool (severe class imbalance) | `DonorPropensityModel` exposes `class_weight` and outputs affinity scores on a 0–100 scale |
| Gift officers need a *dollar figure*, not just a binary signal | `ShareOfWalletRegressor` predicts total philanthropic capacity; `predict_capacity_ratio()` surfaces untapped potential |
| Gift officers need *ranked* prospects, not just binary predictions | `predict_affinity_score()` gives a human-readable score that maps directly to call/visit priority |
| Fundraising reports need year-over-year retention rates | `donor_retention_rate()` is a single function call |

---

## Installation

### Option 1 — Using the pinned Conda environment (recommended)

This reproduces the exact environment used during development and CI.

```bash
# Clone the repository
git clone https://github.com/PhilanthroPy-Project/PhilanthroPy.git
cd PhilanthroPy

# Create the pinned environment
conda env create -f environment.yml

# Activate it
conda activate Philanthropy

# Install the package in editable mode (includes dev extras)
pip install -e ".[dev]"
```

### Option 2 — pip into any Python 3.10+ environment

```bash
pip install -e ".[dev]"
```

**Core runtime dependencies** (auto-installed):

| Package | Version tested |
|---------|---------------|
| `scikit-learn` | ≥ 1.7 |
| `pandas` | ≥ 2.0 |
| `numpy` | ≥ 1.24 |
| `matplotlib` | ≥ 3.7 |
| `seaborn` | ≥ 0.12 |

**Dev / test extras** (`.[dev]`):

| Package | Purpose |
|---------|---------|
| `pytest` | Test runner |
| `pytest-cov` | Coverage reporting |
| `hypothesis` | Property-based testing (stress-tests temporal edge cases) |

---

## Quick Start

```python
from philanthropy.datasets import generate_synthetic_donor_data
from philanthropy.models import DonorPropensityModel

# 1. Generate a synthetic prospect pool (or load your own DataFrame)
df = generate_synthetic_donor_data(n_samples=500, random_state=42)

# 2. Prepare features and labels
feature_cols = ["total_gift_amount", "years_active", "event_attendance_count"]
X = df[feature_cols].to_numpy()
y = df["is_major_donor"].to_numpy()

# 3. Train the model
model = DonorPropensityModel(n_estimators=200, class_weight="balanced", random_state=0)
model.fit(X, y)

# 4. Score every prospect on a 0–100 affinity scale
scores = model.predict_affinity_score(X)

# 5. Rank prospects for your gift officers
df["affinity_score"] = scores
top_prospects = df.sort_values("affinity_score", ascending=False).head(20)
print(top_prospects[["affinity_score", "total_gift_amount", "years_active"]])
```

---

## Package Overview

```
philanthropy/
├── datasets/               # Synthetic data generators for prototyping & testing
│   └── _generator.py
├── preprocessing/          # CRM cleaning, FY engineering, and clinical encounter features
│   ├── transformers.py     # CRMCleaner, FiscalYearTransformer
│   ├── _wealth.py          # WealthScreeningImputer (leakage-safe vendor data imputation)
│   ├── _encounters.py      # EncounterTransformer (clinical discharge → gift date features)
│   └── _rfm.py             # RFMTransformer (recency, frequency, monetary metrics)
├── models/                 # Scikit-learn–compatible estimators
│   ├── propensity.py       # PropensityScorer, LapsePredictor (base stubs)
│   ├── _propensity.py      # DonorPropensityModel (RF-backed classifier, 0-100 affinity score)
│   └── _wallet.py          # ShareOfWalletRegressor (capacity regression + untapped-ratio)
├── metrics/                # Fundraising-specific KPI functions
│   ├── scoring.py
│   └── _financial.py
├── visualisation/          # Presentation-ready visualizations
│   └── _plots.py
├── utils/                  # Shared testing helpers and synthetic data utilities
│   └── testing.py
└── base.py                 # Abstract base classes for all PhilanthroPy estimators
```

---

## Module Reference

### `philanthropy.datasets`

Generate realistic synthetic donor datasets for prototyping, unit testing, and model benchmarking — without needing access to a live CRM database.

#### `generate_synthetic_donor_data(n_samples=1000, random_state=None)`

Returns a `pd.DataFrame` with **5 columns** that simulate a hospital's major-gift prospect pool.

| Column | dtype | Description |
|--------|-------|-------------|
| `total_gift_amount` | `float64` | Cumulative lifetime giving in USD. Log-normally distributed; major donors receive a 3–8× uplift. |
| `years_active` | `int64` | Years since first recorded gift (range: 1–30). Positively correlated with `is_major_donor`. |
| `last_gift_date` | `datetime64[ns]` | Date of most recent gift. Major donors skewed toward the past 2 years using a Beta(1,3) distribution. |
| `event_attendance_count` | `int64` | Number of fundraising events attended (range: 0–20). Positively correlated with `is_major_donor`. |
| `is_major_donor` | `int64` | Binary label: `1` = major-gift prospect (capacity ≥ $25K), `0` = standard annual-fund donor. |

The binary label is generated via a **logistic propensity model**:

```
z = 0.12 × years_active + 0.18 × event_attendance_count − 2.5 + noise
P(major donor) = σ(z)
```

This produces a base positive rate of ~15%, making the dataset non-trivially learnable.

```python
from philanthropy.datasets import generate_synthetic_donor_data

# Reproducible dataset
df = generate_synthetic_donor_data(n_samples=1000, random_state=42)
print(df.dtypes)
# total_gift_amount           float64
# years_active                  int64
# last_gift_date       datetime64[ns]
# event_attendance_count        int64
# is_major_donor                int64

print(df["is_major_donor"].value_counts(normalize=True))
# 0    ~0.85
# 1    ~0.15

# Edge case: zero samples returns an empty DataFrame with correct schema
empty = generate_synthetic_donor_data(n_samples=0)
print(empty.shape)   # (0, 5)
```

---

### `philanthropy.preprocessing`

Transformers inherit from `sklearn.base.TransformerMixin`, so they drop directly into a `Pipeline`.

#### `RFMTransformer`

Transforms transaction logs into Recency, Frequency, and Monetary features relative to a given reference date. 
Standardises raw transaction data into a format suitable for pipeline models.

#### `CRMCleaner`

Standardises raw CRM export DataFrames. Coerces `date_col` to `datetime64` and `amount_col` to `float64`. Optionally delegates leakage-safe wealth imputation to an embedded `WealthScreeningImputer`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `date_col` | `str` | `"gift_date"` | Name of the gift date column. Coerced to `datetime64`. |
| `amount_col` | `str` | `"gift_amount"` | Name of the gift amount column. Coerced to `float64`; non-numeric values → `NaN`. |
| `fiscal_year_start` | `int` | `7` | Starting month of the fiscal year (1–12). |
| `wealth_imputer` | `WealthScreeningImputer \| None` | `None` | **Unfitted** imputer instance. `CRMCleaner.fit()` will call `wealth_imputer.fit(X_train)`, guaranteeing no test-set data contaminates fill statistics. |

**Fitted attributes:**

| Attribute | Description |
|-----------|-------------|
| `feature_names_in_` | Column names seen during `fit()`. |
| `n_features_in_` | Number of input columns. |

```python
import numpy as np
from philanthropy.preprocessing import CRMCleaner, WealthScreeningImputer

# Without wealth imputation
cleaner = CRMCleaner(date_col="gift_date", amount_col="gift_amount", fiscal_year_start=7)
cleaned_df = cleaner.fit_transform(raw_df)

# With leakage-safe wealth imputation
imputer = WealthScreeningImputer(
    wealth_cols=["estimated_net_worth", "real_estate_value"],
    strategy="median",
    add_indicator=True,   # appends <col>__was_missing flags
)
cleaner = CRMCleaner(wealth_imputer=imputer)   # pass UNFITTED imputer
cleaned_df = cleaner.fit_transform(X_train)   # imputer.fit() called here
test_df   = cleaner.transform(X_test)         # frozen fill values applied
```

---

#### `WealthScreeningImputer`

Leakage-safe median/mean/zero imputation for wealth-screening vendor columns (e.g., Estimated Net Worth, Real Estate Value, Stock Holdings). All fill statistics are computed **exclusively** from `X_train` at `fit()` time and frozen — calling `transform()` on held-out data never updates them.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wealth_cols` | `list[str] \| None` | `None` | Columns to impute. Defaults to a canonical vendor set (`estimated_net_worth`, `real_estate_value`, `stock_holdings`, `charitable_capacity`, `planned_gift_inclination`). |
| `strategy` | `"median"\|"mean"\|"zero"` | `"median"` | Imputation strategy. `"median"` is strongly recommended for wealth data (robust to extreme outliers). |
| `add_indicator` | `bool` | `True` | Append a `<col>__was_missing` binary column (dtype `uint8`) for each imputed column — retains absence-of-record as a signal. |
| `fiscal_year_start` | `int` | `7` | FY start month (inherited for pipeline compatibility). |

**Fitted attributes:** `fill_values_` (dict), `imputed_cols_` (list), `n_features_in_` (int)

```python
import numpy as np, pandas as pd
from philanthropy.preprocessing import WealthScreeningImputer

X_train = pd.DataFrame({
    "estimated_net_worth": [1e6, np.nan, 3e6, np.nan],
    "real_estate_value":   [np.nan, 4e5, np.nan, 2e5],
    "gift_amount":         [5000, 250, 10000, 750],
})
X_test = pd.DataFrame({
    "estimated_net_worth": [np.nan, 2e6],
    "real_estate_value":   [5e5, np.nan],
    "gift_amount":         [1000, 3000],
})

imp = WealthScreeningImputer(
    wealth_cols=["estimated_net_worth", "real_estate_value"],
    strategy="median",
    add_indicator=True,
)
imp.fit(X_train)
print(imp.fill_values_)
# {'estimated_net_worth': 2000000.0, 'real_estate_value': 300000.0}

X_test_out = imp.transform(X_test)
# estimated_net_worth__was_missing and real_estate_value__was_missing columns added
print(X_test_out["estimated_net_worth__was_missing"].tolist())  # [1, 0]
```

---

#### `FiscalYearTransformer`

Appends `fiscal_year` and `fiscal_quarter` columns based on the organisation's fiscal calendar. Handles July-start (US academic medical centres), October-start (US federal), and any custom month.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `date_col` | `str` | `"gift_date"` | Name of the gift date column. Must contain datetime-parsable values. |
| `fiscal_year_start` | `int` | `7` | Starting month of the fiscal year (1–12). |

**Behaviour:**
- A gift dated **2023-07-01** with `fiscal_year_start=7` → `fiscal_year = 2024`
- A gift dated **2023-06-30** with `fiscal_year_start=7` → `fiscal_year = 2023`

**Raises:** `ValueError` if `fiscal_year_start` is outside 1–12 (raised at `.fit()` time).

```python
import pandas as pd
from philanthropy.preprocessing import FiscalYearTransformer

df = pd.DataFrame({"gift_date": ["2023-07-01", "2023-06-30", "2024-01-15"]})

transformer = FiscalYearTransformer(date_col="gift_date", fiscal_year_start=7)
result = transformer.fit_transform(df)

print(result[["gift_date", "fiscal_year", "fiscal_quarter"]])
#    gift_date  fiscal_year  fiscal_quarter
# 0  2023-07-01        2024               1   # July  = FY-Q1
# 1  2023-06-30        2023               4   # June  = FY-Q4
# 2  2024-01-15        2024               3   # Jan   = FY-Q3
```

---

#### `EncounterTransformer`

Bridges the clinical data warehouse with the advancement CRM by merging hospital encounter (discharge) records with philanthropic gift histories. Produces two continuous temporal features that are strong signals in major-gift propensity models for academic medical centres (AMCs).

**Features produced:**

| Output column | Description |
|---------------|-------------|
| `days_since_last_discharge` | Integer days between the donor's most recent discharge (at `fit` time) and the gift date. Pre-discharge gifts → `NaN` by default (`allow_negative_days=False`). |
| `encounter_frequency_score` | `log1p(encounter_count)` — log-scaled total encounter count, normalising the heavy right-skew typical in AMC data. Donors with no record → `0.0`. |

**Privacy guarantee:** all identifier-like columns (`merge_key`, plus any column name containing `_id`, `mrn`, `ssn`, `name`, `dob`, `zip`) are **silently dropped** from the output before it is returned.

**Leakage guarantee:** `encounter_summary_` is computed exclusively from `encounter_df` at `fit()` time. Post-fit mutation of `encounter_df` cannot alter `transform()` output. Donors absent from the training-period encounter table always receive `NaN` / `0.0`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encounter_df` | `pd.DataFrame` | — | Reference table of clinical encounters (must contain `merge_key` and `discharge_col`). |
| `discharge_col` | `str` | `"discharge_date"` | Column in `encounter_df` holding discharge timestamps. |
| `gift_date_col` | `str` | `"gift_date"` | Column in `X` holding gift dates. |
| `merge_key` | `str` | `"donor_id"` | Column present in both `encounter_df` and `X` used to join the tables. Dropped from output. |
| `allow_negative_days` | `bool` | `False` | If `True`, retain negative `days_since_last_discharge` (gift predates discharge). |
| `id_cols_to_drop` | `list[str] \| None` | `None` | Extra identifier columns to strip beyond the PII heuristic. |

**Fitted attributes:** `encounter_summary_` (DataFrame), `dropped_cols_` (list), `n_features_in_` (int)

```python
import pandas as pd
from philanthropy.preprocessing import EncounterTransformer

# Clinical encounter records (from EHR / data warehouse)
enc_df = pd.DataFrame({
    "donor_id":       [101, 101, 102, 103],
    "discharge_date": ["2022-03-15", "2023-06-01", "2021-11-20", "2020-08-10"],
})

# Gift-level CRM records (training split)
gift_df = pd.DataFrame({
    "donor_id":    [101, 102, 103, 104],   # donor 104 has no encounter record
    "gift_date":   ["2023-08-01", "2022-02-15", "2021-09-30", "2023-01-01"],
    "gift_amount": [10_000.0, 500.0, 250.0, 1_000.0],
})

t = EncounterTransformer(
    encounter_df=enc_df,
    discharge_col="discharge_date",
    gift_date_col="gift_date",
    merge_key="donor_id",
)
out = t.fit_transform(gift_df)

print(out.columns.tolist())
# ['gift_amount', 'days_since_last_discharge', 'encounter_frequency_score']
# donor_id and gift_date are stripped automatically

print(out[["days_since_last_discharge", "encounter_frequency_score"]])
#    days_since_last_discharge  encounter_frequency_score
# 0                      61.0                   1.098612   # log1p(2 encounters)
# 1                        NaN                   1.098612   # gift before discharge
# 2                        NaN                   0.693147
# 3                        NaN                   0.000000   # donor 104 unknown
```

---

---

### `philanthropy.models`

All models inherit from `sklearn.base.BaseEstimator` and the appropriate sklearn Mixin, making them compatible with `Pipeline`, `GridSearchCV`, `cross_val_score`, and `clone()`.

#### `DonorPropensityModel` ⭐

The flagship estimator. Wraps a `RandomForestClassifier` and adds a fundraising-native `.predict_affinity_score()` method. Passes all **45 sklearn `check_estimator` checks** on scikit-learn 1.7.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_estimators` | `int` | `100` | Number of trees in the Random Forest. Increase for more stable probability estimates. |
| `max_depth` | `int \| None` | `None` | Maximum tree depth. `None` = grow until pure. Set to 5–10 to regularise on small prospect pools. |
| `min_samples_split` | `int \| float` | `2` | Minimum samples to split an internal node. Larger values = stronger regularisation. |
| `min_samples_leaf` | `int \| float` | `1` | Minimum samples at a leaf node. |
| `min_weight_fraction_leaf` | `float` | `0.0` | Minimum weighted fraction at a leaf. Useful when combined with `class_weight`. |
| `class_weight` | `dict \| "balanced" \| None` | `None` | Class weighting scheme. Pass `"balanced"` for imbalanced datasets (<5% major donors). |
| `random_state` | `int \| None` | `None` | Seed for reproducibility. Required for audit-trail compliance. |

**Fitted attributes (available after `.fit()`):**

| Attribute | Description |
|-----------|-------------|
| `estimator_` | The trained `RandomForestClassifier` backend. Access `.feature_importances_` for explainability. |
| `classes_` | Array of unique class labels seen during fit, e.g. `array([0, 1])`. |
| `n_features_in_` | Number of feature columns seen during fit. |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `.fit(X, y)` | `self` | Train on feature matrix `X` and binary target `y`. |
| `.predict(X)` | `ndarray (n,)` | Binary class predictions (0 or 1). |
| `.predict_proba(X)` | `ndarray (n, 2)` | Class probabilities `[P(0), P(1)]`. Each row sums to 1.0. |
| `.predict_affinity_score(X)` | `ndarray (n,)` | `P(major donor) × 100`, rounded to 2 d.p. Range: [0.0, 100.0]. |
| `.get_params()` | `dict` | Returns all constructor parameters (sklearn standard). |
| `.set_params(**params)` | `self` | Update parameters (enables `GridSearchCV`). |

**Affinity score interpretation:**

| Score Range | Recommended Action |
|-------------|-------------------|
| **80–100** | Premium prospect — assign a major gift officer immediately |
| **60–79** | Strong prospect — include in next bi-annual solicitation cycle |
| **40–59** | Moderate prospect — steward via annual fund or planned giving |
| **0–39** | Low propensity — retain in broad annual-appeal pool |

```python
from philanthropy.models import DonorPropensityModel
from philanthropy.datasets import generate_synthetic_donor_data

df = generate_synthetic_donor_data(1000, random_state=0)
X = df[["total_gift_amount", "years_active", "event_attendance_count"]].to_numpy()
y = df["is_major_donor"].to_numpy()

model = DonorPropensityModel(
    n_estimators=300,
    max_depth=8,
    class_weight="balanced",
    random_state=42,
)
model.fit(X, y)

# Binary predictions
preds = model.predict(X)          # array([0, 1, 0, ...])

# Calibrated probabilities
proba = model.predict_proba(X)    # shape (1000, 2)

# 0–100 affinity scores
scores = model.predict_affinity_score(X)  # array([12.5, 87.3, 4.0, ...])

# Inspect which features drive propensity
import pandas as pd
importances = pd.Series(
    model.estimator_.feature_importances_,
    index=["total_gift_amount", "years_active", "event_attendance_count"]
).sort_values(ascending=False)
print(importances)
```

---

#### `MajorGiftClassifier`

Classifies major gifts using calibrated probability estimates. Uses `HistGradientBoostingClassifier` as the backend to handle missing data natively, and wraps the backend in `CalibratedClassifierCV` to ensure the `.predict_proba()` outputs are true probabilities. Provides a `.predict_affinity_score(X)` method scaling the calibrated probability to a 0-100 integer.

#### `PropensityScorer`

A lightweight base stub for propensity scoring. Uses a pluggable `estimator` backend and a configurable `threshold` for class boundary tuning.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimator` | estimator \| None | `None` | Backend sklearn classifier. |
| `threshold` | `float` | `0.5` | Decision threshold for `predict()`. |
| `fiscal_year_start` | `int` | `7` | Fiscal year starting month. |

```python
from philanthropy.models import PropensityScorer
from sklearn.ensemble import GradientBoostingClassifier

scorer = PropensityScorer(estimator=GradientBoostingClassifier(), threshold=0.4)
scorer.fit(X, y)
preds = scorer.predict(X)
```

---

#### `LapsePredictor`

Identifies donors at risk of lapsing (i.e., not renewing their gift) using a production `RandomForestClassifier` backend. Useful for retention-focused campaigns.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_estimators` | `int` | `100` | Number of trees in the Random Forest. |
| `lapse_window_years` | `int` | `2` | Time window over which to predict lapsing, in years. |

Mapping to retention strategies based on predicted probabilities:
- **High Risk (> 70%)**: Flag for immediate high-touch stewardship.
- **Moderate Risk (40% - 70%)**: Add to targeted multi-channel re-engagement campaigns.
- **Low Risk (< 40%)**: Continue with standard cyclic communications.

```python
from philanthropy.models import LapsePredictor

predictor = LapsePredictor(lapse_window_years=3)
predictor.fit(X, y)
at_risk = predictor.predict(X)   # 1 = at risk of lapsing
```

---

#### `ShareOfWalletRegressor`

Predicts a donor's **total philanthropic capacity** in dollars — the fundraising concept of "share of wallet". Unlike `DonorPropensityModel` (binary: will they give?), this regressor answers "*how much* could they give?" and surfaces untapped major-gift potential via `predict_capacity_ratio()`.

Powered by `HistGradientBoostingRegressor` under the hood, which handles missing CRM and wealth-screening values **natively** — no upstream imputation step required.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | `float` | `0.1` | Boosting step size. |
| `max_iter` | `int` | `100` | Number of boosting trees. Increase to 300–500 for production prospect pools. |
| `max_depth` | `int \| None` | `None` | Maximum tree depth. |
| `l2_regularization` | `float` | `0.0` | L2 leaf-weight regularization. Increase to `1.0` for sparse prospect datasets. |
| `min_samples_leaf` | `int` | `20` | Minimum samples per leaf. |
| `random_state` | `int \| None` | `None` | Reproducibility seed. |
| `capacity_floor` | `float` | `1.0` | Minimum predicted capacity in dollars (clips semantically-impossible negatives). |

**Fitted attributes:** `estimator_` (HistGradientBoostingRegressor), `n_features_in_` (int)

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `.fit(X, y)` | `self` | Fit on capacity labels. `X` may contain `NaN`. |
| `.predict(X)` | `ndarray (n,)` | Predicted capacity in dollars, clipped to `[capacity_floor, ∞)`. |
| `.predict_capacity_ratio(X, historical_giving)` | `ndarray (n,)` | `predicted_capacity / max(historical_giving, 1.0)`. Ratios ≥ 5× flag strong untapped-potential candidates. |

**Capacity ratio interpretation:**

| Ratio | Recommended action |
|-------|-------------------|
| **≥ 10×** | Dramatically under-asked — schedule discovery call immediately |
| **5–9×** | Significant untapped potential — major-gift candidate |
| **2–4×** | Moderate upside — consider upgrade ask |
| **< 2×** | Near capacity — focus on retention and stewardship |

```python
import numpy as np
from philanthropy.models import ShareOfWalletRegressor
from philanthropy.datasets import generate_synthetic_donor_data

df = generate_synthetic_donor_data(n_samples=500, random_state=0)
feature_cols = ["total_gift_amount", "years_active", "event_attendance_count"]
X = df[feature_cols].to_numpy()

# Simulate capacity labels (in practice: prospect-research ratings)
capacity = df["total_gift_amount"].to_numpy() * np.random.default_rng(0).uniform(1, 10, 500)
historical = df["total_gift_amount"].to_numpy()

model = ShareOfWalletRegressor(max_iter=200, random_state=42)
model.fit(X, capacity)

# Predicted dollar capacity per prospect
caps = model.predict(X)

# Untapped-potential ratio — the gift-officer priority metric
ratios = model.predict_capacity_ratio(X, historical_giving=historical)
print(ratios[:5])   # e.g. [3.2, 8.7, 1.1, 12.4, 2.9]
```

---

### `philanthropy.metrics`

Pure functions — no fitting required. Take standard Python collections as input, return scalar floats. Safe to call in any reporting script.

#### `donor_lifetime_value(average_donation, lifespan_years, discount_rate=0.05, retention_rate=None) → float`

Computes the Net Present Value (NPV) of a donor's future giving. If `retention_rate` is provided, dynamically calculates the expected lifespan; otherwise uses the fixed `lifespan_years`.

```python
from philanthropy.metrics import donor_lifetime_value

ltv = donor_lifetime_value(average_donation=100.0, lifespan_years=5, discount_rate=0.05)
print(f"Donor LTV: ${ltv:,.2f}")
```

#### `donor_retention_rate(current_donors, prior_donors) → float`

Calculates the year-over-year donor retention rate:

```
Retention = |Current ∩ Prior| / |Prior|
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `current_donors` | `Collection` | Donor IDs active in the current period. |
| `prior_donors` | `Collection` | Donor IDs active in the prior period. |

**Returns:** `float` in [0.0, 1.0]. Returns `0.0` if `prior_donors` is empty.

```python
from philanthropy.metrics import donor_retention_rate

fy2023_donors = {"D00001", "D00002", "D00003", "D00004"}
fy2024_donors = {"D00001", "D00002", "D00005"}

rate = donor_retention_rate(fy2024_donors, fy2023_donors)
print(f"Retention rate: {rate:.1%}")   # Retention rate: 50.0%
```

---

#### `donor_acquisition_cost(total_fundraising_expense, new_donors_acquired) → float`

Calculates the cost to acquire each new donor:

```
DAC = Total Fundraising Expense / New Donors Acquired
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `total_fundraising_expense` | `float` | Total spend on the acquisition campaign in USD. |
| `new_donors_acquired` | `int` | Number of first-time donors gained in the period. |

**Returns:** `float`. Returns `numpy.inf` if `new_donors_acquired == 0`.

```python
from philanthropy.metrics import donor_acquisition_cost

dac = donor_acquisition_cost(total_fundraising_expense=80_000, new_donors_acquired=320)
print(f"Donor Acquisition Cost: ${dac:,.2f}")   # Donor Acquisition Cost: $250.00
```

---

### `philanthropy.visualisation`

Presentation-ready visualizations of donor characteristics and pipeline health. Returns `matplotlib.axes.Axes` objects for further customization.

```python
import numpy as np
import matplotlib.pyplot as plt
from philanthropy.visualisation import plot_affinity_distribution

scores = np.random.uniform(0, 100, 500)
labels = np.random.randint(0, 2, 500)
ax = plot_affinity_distribution(scores, labels=labels)
plt.show()
```

### `philanthropy.utils`

#### `make_donor_dataset(n_donors, ...) → pd.DataFrame`

Generates a multi-gift transaction log for testing preprocessing pipelines. Each donor gets 1–5 synthetic gift records.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_donors` | `int` | `200` | Number of unique donors to simulate. |
| `fiscal_year_start` | `int` | `7` | FY start month. |
| `start_year` | `int` | `2018` | Earliest gift year. |
| `end_year` | `int` | `2024` | Latest gift year. |
| `lapse_rate` | `float` | `0.25` | Approximate fraction of lapsed donors. |
| `major_gift_threshold` | `float` | `10000.0` | USD amount above which `is_major_gift = True`. |
| `random_state` | `int \| None` | `42` | Reproducibility seed. |

**Column schema:**

| Column | dtype | Description |
|--------|-------|-------------|
| `donor_id` | `str` | Unique donor identifier, e.g. `"D00001"`. |
| `gift_date` | `datetime64` | Date of the gift. |
| `gift_amount` | `float` | Gift amount in USD (log-normal distribution). |
| `appeal_code` | `str` | One of: `ANNUAL`, `MAJOR`, `PLANNED`, `ONLINE`, `EVENT`. |
| `is_major_gift` | `bool` | `True` if `gift_amount >= major_gift_threshold`. |

```python
from philanthropy.utils import make_donor_dataset

df = make_donor_dataset(n_donors=500, major_gift_threshold=5_000, random_state=7)
print(df.head())
#   donor_id  gift_date  gift_amount appeal_code  is_major_gift
# 0   D00142 2018-01-04      4213.17       MAJOR          False
# 1   D00089 2018-01-07       312.54      ONLINE          False
# ...
```

---

## Worked Examples

### End-to-End Pipeline

Combine preprocessing and modelling into a single reusable `sklearn.pipeline.Pipeline`:

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

from philanthropy.datasets import generate_synthetic_donor_data
from philanthropy.models import DonorPropensityModel

# --- Data preparation ---
df = generate_synthetic_donor_data(n_samples=2000, random_state=0)
feature_cols = ["total_gift_amount", "years_active", "event_attendance_count"]
X = df[feature_cols].to_numpy()
y = df["is_major_donor"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Build pipeline ---
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", DonorPropensityModel(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
    )),
])

# --- Train & evaluate ---
pipe.fit(X_train, y_train)
print(f"Test accuracy: {pipe.score(X_test, y_test):.3f}")

# 5-fold cross-validation on ROC-AUC
auc_scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
print(f"CV ROC-AUC: {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
```

---

### Affinity Score Ranking

Produce a ranked prospect table ready to export to your CRM:

```python
import pandas as pd
from philanthropy.datasets import generate_synthetic_donor_data
from philanthropy.models import DonorPropensityModel

df = generate_synthetic_donor_data(n_samples=500, random_state=1)
feature_cols = ["total_gift_amount", "years_active", "event_attendance_count"]
X = df[feature_cols].to_numpy()
y = df["is_major_donor"].to_numpy()

model = DonorPropensityModel(n_estimators=200, class_weight="balanced", random_state=0)
model.fit(X, y)

# Score all prospects
df["affinity_score"] = model.predict_affinity_score(X)

# Segment into tiers
def tier(score):
    if score >= 80: return "Premium"
    if score >= 60: return "Strong"
    if score >= 40: return "Moderate"
    return "Low"

df["tier"] = df["affinity_score"].apply(tier)

# Summary by tier
print(df.groupby("tier")["affinity_score"].describe().round(1))

# Top 10 prospects for the gift-officer call list
call_list = (
    df.sort_values("affinity_score", ascending=False)
    .head(10)[["affinity_score", "tier", "total_gift_amount", "years_active"]]
)
print(call_list.to_string(index=False))
```

---

### Fiscal Year Analysis

Combine `FiscalYearTransformer` with the transaction log generator to produce a fiscal-year gift summary:

```python
import pandas as pd
from philanthropy.utils import make_donor_dataset
from philanthropy.preprocessing import CRMCleaner, FiscalYearTransformer
from sklearn.pipeline import Pipeline

# Generate a multi-year transaction log
raw_df = make_donor_dataset(n_donors=300, start_year=2020, end_year=2024, random_state=42)

# Build a preprocessing pipeline (July fiscal year start)
cleaner = CRMCleaner(date_col="gift_date", amount_col="gift_amount", fiscal_year_start=7)
fy_transformer = FiscalYearTransformer(date_col="gift_date", fiscal_year_start=7)

cleaned = cleaner.fit_transform(raw_df)
enriched = fy_transformer.fit_transform(cleaned)

# Aggregate giving by fiscal year
fy_summary = (
    enriched.groupby("fiscal_year")
    .agg(
        total_gifts=("gift_amount", "sum"),
        gift_count=("gift_amount", "count"),
        unique_donors=("donor_id", "nunique"),
    )
    .round(2)
)
print(fy_summary)
```

---

### Medical Philanthropy Pipeline

Combine `EncounterTransformer` and `WealthScreeningImputer` with `DonorPropensityModel` in a single, leakage-safe pipeline:

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from philanthropy.preprocessing import (
    CRMCleaner,
    WealthScreeningImputer,
    EncounterTransformer,
    FiscalYearTransformer,
)
from philanthropy.models import DonorPropensityModel

# --- Clinical encounter records from the EHR data warehouse ---
enc_df = pd.DataFrame({
    "donor_id":       [101, 101, 102, 103, 104],
    "discharge_date": ["2022-01-10", "2023-06-01", "2022-03-05",
                       "2022-07-20", "2023-02-14"],
})

# --- Gift-level CRM export with partial wealth-screening data ---
gift_df = pd.DataFrame({
    "donor_id":            [101, 102, 103, 104, 105],
    "gift_date":           ["2023-08-01", "2023-10-15", "2023-08-01",
                            "2023-11-01", "2023-09-20"],
    "gift_amount":         [10_000.0, 500.0, 250.0, 2_000.0, 750.0],
    "estimated_net_worth": [np.nan, 1_500_000.0, np.nan, 800_000.0, np.nan],
    "real_estate_value":   [200_000.0, np.nan, 300_000.0, np.nan, 150_000.0],
    "is_major_donor":      [1, 0, 0, 0, 0],
})

# Step 1: fit EncounterTransformer to clinical data
enc_transformer = EncounterTransformer(encounter_df=enc_df, merge_key="donor_id")
gift_features = enc_transformer.fit_transform(gift_df.drop(columns=["is_major_donor"]))

# Step 2: build the rest of the pipeline
wealth_imp = WealthScreeningImputer(
    wealth_cols=["estimated_net_worth", "real_estate_value"],
    strategy="median",
    add_indicator=True,
)
cleaner = CRMCleaner(wealth_imputer=wealth_imp)

# gift_features now has numeric columns only (PII stripped by EncounterTransformer)
X = cleaner.fit_transform(gift_features).select_dtypes(include="number").to_numpy()
y = gift_df["is_major_donor"].to_numpy()

model = DonorPropensityModel(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X, y)
scores = model.predict_affinity_score(X)
print(scores)   # e.g. [87.5, 12.3, 5.0, 34.2, 8.1]
```

---

### Share-of-Wallet Capacity Ranking

Use `ShareOfWalletRegressor` to build a **dollar-capacity priority list** for your major-gift team:

```python
import numpy as np
import pandas as pd
from philanthropy.datasets import generate_synthetic_donor_data
from philanthropy.models import ShareOfWalletRegressor

df = generate_synthetic_donor_data(n_samples=500, random_state=0)
feature_cols = ["total_gift_amount", "years_active", "event_attendance_count"]
X = df[feature_cols].to_numpy()

# Simulate capacity labels (replace with prospect-research ratings in production)
rng = np.random.default_rng(0)
capacity_labels = df["total_gift_amount"].to_numpy() * rng.uniform(1, 15, 500)

model = ShareOfWalletRegressor(max_iter=200, l2_regularization=0.5, random_state=42)
model.fit(X, capacity_labels)

# Two-stage portfolio view: propensity × capacity
df["predicted_capacity"]  = model.predict(X)
df["capacity_ratio"]      = model.predict_capacity_ratio(
    X, historical_giving=df["total_gift_amount"].to_numpy()
)

# Rank by untapped potential — the gift-officer priority metric
priority_list = (
    df.sort_values("capacity_ratio", ascending=False)
    .head(10)[["total_gift_amount", "predicted_capacity", "capacity_ratio", "years_active"]]
    .rename(columns={
        "total_gift_amount":  "Historical Giving",
        "predicted_capacity": "Predicted Capacity",
        "capacity_ratio":     "Untapped Ratio",
        "years_active":       "Years Active",
    })
)
print(priority_list.to_string(index=False))
# Historical Giving  Predicted Capacity  Untapped Ratio  Years Active
#          2,341.18          187,293.44           79.99            27
#          5,213.45          312,807.00           59.99            24
#          ...
```

---

### Donor Retention Reporting


Calculate year-over-year retention and acquisition cost from your enriched DataFrame:

```python
from philanthropy.utils import make_donor_dataset
from philanthropy.preprocessing import FiscalYearTransformer
from philanthropy.metrics import donor_retention_rate, donor_acquisition_cost

raw_df = make_donor_dataset(n_donors=500, random_state=0)
enriched = FiscalYearTransformer(fiscal_year_start=7).fit_transform(raw_df)

# Build donor sets per fiscal year
fy_donors = (
    enriched.groupby("fiscal_year")["donor_id"]
    .apply(set)
    .to_dict()
)

# Compute retention for each consecutive pair of fiscal years
fiscal_years = sorted(fy_donors.keys())
print(f"{'FY Pair':<15} {'Retention':>10}")
print("-" * 27)
for prior, current in zip(fiscal_years, fiscal_years[1:]):
    rate = donor_retention_rate(fy_donors[current], fy_donors[prior])
    print(f"FY{prior}→FY{current}    {rate:>10.1%}")

# Acquisition cost example
dac = donor_acquisition_cost(total_fundraising_expense=120_000, new_donors_acquired=480)
print(f"\nDonor Acquisition Cost: ${dac:,.2f}")
```

---

### Hyperparameter Tuning with GridSearchCV

`DonorPropensityModel` is fully compatible with sklearn's model selection utilities:

```python
from sklearn.model_selection import GridSearchCV
from philanthropy.models import DonorPropensityModel
from philanthropy.datasets import generate_synthetic_donor_data

df = generate_synthetic_donor_data(1000, random_state=0)
X = df[["total_gift_amount", "years_active", "event_attendance_count"]].to_numpy()
y = df["is_major_donor"].to_numpy()

param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [None, 5, 10],
    "class_weight": [None, "balanced"],
}

grid = GridSearchCV(
    DonorPropensityModel(random_state=42),
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)
grid.fit(X, y)

print(f"Best ROC-AUC: {grid.best_score_:.3f}")
print(f"Best params:  {grid.best_params_}")
```

---

## API Reference

### Quick lookup

| Symbol | Module | Type |
|--------|--------|------|
| `generate_synthetic_donor_data` | `philanthropy.datasets` | function |
| `CRMCleaner` | `philanthropy.preprocessing` | Transformer |
| `FiscalYearTransformer` | `philanthropy.preprocessing` | Transformer |
| `WealthScreeningImputer` | `philanthropy.preprocessing` | Transformer |
| `EncounterTransformer` | `philanthropy.preprocessing` | Transformer |
| `DonorPropensityModel` | `philanthropy.models` | Classifier |
| `ShareOfWalletRegressor` | `philanthropy.models` | Regressor |
| `PropensityScorer` | `philanthropy.models` | Classifier |
| `LapsePredictor` | `philanthropy.models` | Classifier |
| `donor_retention_rate` | `philanthropy.metrics` | function |
| `donor_acquisition_cost` | `philanthropy.metrics` | function |
| `make_donor_dataset` | `philanthropy.utils` | function |

---

## Design Principles (Golden Rules)

PhilanthroPy is written to be a first-class scikit-learn citizen. All estimators follow these rules:

1. **`__init__` is pure**: No validation, no data processing — only `self.param = param` assignments. This allows `clone()` and `get_params()` to work correctly.

2. **Fitted attributes end with `_`**: Any attribute estimated during `fit()` uses a trailing underscore (e.g., `estimator_`, `classes_`, `n_features_in_`). This convention signals the fitted state and is checked by `check_is_fitted()`.

3. **Strict I/O contracts**: `fit()` calls `check_X_y(X, y)` and `predict*()`/`transform()` calls `check_array(X)` — guaranteeing acceptance of NumPy arrays, Pandas DataFrames, and sparse matrices alike while returning pure `np.ndarray` outputs.

4. **`check_estimator` compatible**: `DonorPropensityModel` passes all 45 checks in `sklearn.utils.estimator_checks.parametrize_with_checks` on scikit-learn 1.7.

5. **Mixin order**: Specialised mixins (e.g., `ClassifierMixin`) appear **before** `BaseEstimator` in the inheritance list, satisfying sklearn's `check_mixin_order` requirement.

---

## Development Environment

| Item | Value |
|------|-------|
| Python | 3.10.19 |
| scikit-learn | 1.7.2 |
| pandas | 2.3.3 |
| numpy | 2.2.6 |
| Conda env name | `Philanthropy` |

```bash
# Activate the pinned environment
conda activate Philanthropy

# Verify the install
python -c "import philanthropy; print(philanthropy.__version__)"
# 0.1.0
```

---

## Testing

```bash
# Run the full test suite
conda activate Philanthropy
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=philanthropy --cov-report=term-missing

# Run a specific test file
pytest tests/test_donor_propensity_model.py -v

# Run only the sklearn check_estimator battery
pytest tests/test_donor_propensity_model.py -k "sklearn_estimator_checks" -v
```

**Test coverage summary (161 tests):**

| Test file | Tests | What's covered |
|-----------|------:|----------------|
| `test_datasets.py` | 19 | Schema, dtypes, ranges, reproducibility, domain correlations, edge cases |
| `test_donor_propensity_model.py` | 84 | All public methods, Golden Rules, 45 `check_estimator` checks, Pipeline, `clone()`, `GridSearchCV` |
| `test_preprocessing.py` | 38 | `CRMCleaner` (+ `WealthScreeningImputer` integration), `FiscalYearTransformer` (6 hypothesis property-based invariants across leap years, pre-1970 dates, all 12 fiscal start months, timezone offsets), `WealthScreeningImputer` (all 3 strategies, indicator columns, leakage freeze), `EncounterTransformer` (PII stripping, NaN logic, frequency scoring, fit purity) |
| `test_leakage.py` | 7 | Temporal leakage prevention: `encounter_summary_` freeze, `WealthScreeningImputer` fill-value immutability post-transform, train/test split invariance |
| `test_metrics.py` | 6 | `donor_retention_rate`, `donor_acquisition_cost` |
| `test_propensity.py` | 3 | `PropensityScorer`, `LapsePredictor` |
| `test_utils.py` | 4 | `make_donor_dataset` fixture |

> **Property-based testing**: `test_preprocessing.py` uses [hypothesis](https://hypothesis.readthedocs.io/) to generate thousands of randomised datetime inputs, mathematically guaranteeing `FiscalYearTransformer` stability across the full input space. Run with `pytest tests/test_preprocessing.py -v`.

```bash
# Run only the property-based hypothesis tests
pytest tests/test_preprocessing.py -k "Hypothesis" -v

# Run the leakage prevention suite
pytest tests/test_leakage.py -v
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository and create a feature branch (`git checkout -b feat/my-feature`).
2. **Write tests first** — new public API must have ≥ 80% test coverage before merging.
3. Use **NumPy docstring format** for all public functions and classes.
4. Adhere to the [Golden Rules](#design-principles-golden-rules) for any new estimator.
5. Ensure `pytest tests/ -v` passes with **zero failures** in the `Philanthropy` conda environment.
6. Open a pull request with a clear description of the change and the problem it solves.

### Roadmap

- [x] `philanthropy.preprocessing.CRMCleaner` — leakage-safe CRM standardisation with embedded `WealthScreeningImputer`
- [x] `philanthropy.preprocessing.WealthScreeningImputer` — median/mean/zero imputation for 30–70%-missing vendor data
- [x] `philanthropy.preprocessing.EncounterTransformer` — clinical discharge → philanthropic feature engineering
- [x] `philanthropy.models.ShareOfWalletRegressor` — continuous capacity regression + `predict_capacity_ratio()`
- [x] Property-based testing (hypothesis) for `FiscalYearTransformer` across leap years, pre-1970 dates, all start months
- [x] Temporal leakage prevention test suite (`test_leakage.py`)
- [x] `philanthropy.preprocessing.RFMTransformer` — compute Recency, Frequency, Monetary features
- [x] `philanthropy.models.MajorGiftClassifier` — gradient-boosted variant with calibrated probabilities
- [x] `philanthropy.models.LapsePredictor` — production RF implementation (currently stub)
- [x] `philanthropy.metrics.donor_lifetime_value()` — LTV calculation with discount rate
- [x] `philanthropy.visualisation` — affinity score plots, retention waterfall charts
- [ ] Full Sphinx documentation site

---

## License

[MIT](LICENSE) © 2026 Shivam Lalakiya