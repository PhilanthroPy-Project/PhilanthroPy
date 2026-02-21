# PhilanthroPy 🎗️

> **A scikit-learn–compatible toolkit for predictive donor analytics in the nonprofit and medical philanthropy sector.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7%2B-orange.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/tests-120%20passing-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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
| CRM exports vary wildly by vendor (Raiser's Edge, Salesforce NPSP, Veeva) | `CRMCleaner` standardises raw exports before any modelling |
| Fiscal years start in July (or any month), not January | `FiscalYearTransformer` appends FY/FQ columns automatically |
| Major donors are <5 % of a prospect pool (severe class imbalance) | `DonorPropensityModel` exposes `class_weight` and outputs affinity scores on a 0–100 scale |
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
├── datasets/           # Synthetic data generators for prototyping & testing
│   └── _generator.py
├── preprocessing/      # CRM data cleaners and fiscal-year feature engineering
│   └── transformers.py
├── models/             # Scikit-learn–compatible donor propensity classifiers
│   ├── propensity.py   # PropensityScorer, LapsePredictor (base stubs)
│   └── _propensity.py  # DonorPropensityModel (production RF-backed model)
├── metrics/            # Fundraising-specific KPI functions
│   └── scoring.py
├── utils/              # Shared testing helpers and synthetic data utilities
│   └── testing.py
└── base.py             # Abstract base classes for all PhilanthroPy estimators
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

#### `CRMCleaner`

Standardises raw CRM export DataFrames. Validates fiscal year configuration at `.fit()` time and stores feature names for downstream compatibility.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `date_col` | `str` | `"gift_date"` | Name of the gift date column. |
| `amount_col` | `str` | `"gift_amount"` | Name of the gift amount column. |
| `fiscal_year_start` | `int` | `7` | Starting month of the fiscal year (1–12). |

**Fitted attributes:**

| Attribute | Description |
|-----------|-------------|
| `feature_names_in_` | Column names seen during `fit()`. |

```python
from philanthropy.preprocessing import CRMCleaner

cleaner = CRMCleaner(date_col="gift_date", amount_col="gift_amount", fiscal_year_start=7)
cleaned_df = cleaner.fit_transform(raw_df)
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
# 0  2023-07-01        2024             NaN
# 1  2023-06-30        2023             NaN
# 2  2024-01-15        2024             NaN
```

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

Identifies donors at risk of lapsing (i.e., not renewing their gift). Useful for retention-focused campaigns.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimator` | estimator \| None | `None` | Backend sklearn classifier. |
| `lapse_window_years` | `int` | `2` | Number of years of inactivity that defines a lapsed donor. |
| `threshold` | `float` | `0.5` | Decision threshold for `predict()`. |
| `fiscal_year_start` | `int` | `7` | Fiscal year starting month. |

```python
from philanthropy.models import LapsePredictor

predictor = LapsePredictor(lapse_window_years=3, threshold=0.3)
predictor.fit(X, y)
at_risk = predictor.predict(X)   # 1 = at risk of lapsing
```

---

### `philanthropy.metrics`

Pure functions — no fitting required. Take standard Python collections as input, return scalar floats. Safe to call in any reporting script.

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
| `DonorPropensityModel` | `philanthropy.models` | Classifier |
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

**Test coverage summary (120 tests):**

| Test file | Tests | What's covered |
|-----------|------:|----------------|
| `test_datasets.py` | 23 | Schema, dtypes, ranges, reproducibility, domain correlations, edge cases |
| `test_donor_propensity_model.py` | 97 | All public methods, Golden Rules, 45 `check_estimator` checks, Pipeline, `clone()`, `GridSearchCV` |
| `test_preprocessing.py` | 4 | `CRMCleaner`, `FiscalYearTransformer`, edge cases |
| `test_propensity.py` | 3 | `PropensityScorer`, `LapsePredictor` |
| `test_metrics.py` | 6 | `donor_retention_rate`, `donor_acquisition_cost` |
| `test_utils.py` | 4 | `make_donor_dataset` fixture |

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

- [ ] `philanthropy.preprocessing.RFMTransformer` — compute Recency, Frequency, Monetary features
- [ ] `philanthropy.models.MajorGiftClassifier` — gradient-boosted variant with calibrated probabilities
- [ ] `philanthropy.models.LapsePredictor` — production RF implementation (currently stub)
- [ ] `philanthropy.metrics.donor_lifetime_value()` — LTV calculation with discount rate
- [ ] `philanthropy.visualisation` — affinity score plots, retention waterfall charts
- [ ] Full Sphinx documentation site

---

## License

[MIT](LICENSE) © 2026 Shivam Lalakiya