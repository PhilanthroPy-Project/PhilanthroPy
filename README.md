<p align="center">
  <img src="docs/assets/logo.png" alt="PhilanthroPy logo" width="180"/>
</p>

<p align="center">
  <strong>PhilanthroPy: Code for a cause—predictive analytics for advancement teams.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/PhilanthroPy-Project/PhilanthroPy/main/pyproject.toml&query=$.project.version&label=version&color=blue" alt="version"/>
  <img src="https://img.shields.io/badge/python-3.9%2B-brightgreen" alt="python"/>
  <img src="https://img.shields.io/badge/sklearn-compatible-orange" alt="sklearn"/>
  <img src="https://img.shields.io/badge/docs-GitHub%20Pages-informational" alt="documentation"/>
  [![Tests](https://github.com/PhilanthroPy-Project/PhilanthroPy/actions/workflows/ci.yml/badge.svg)](https://github.com/PhilanthroPy-Project/PhilanthroPy/actions/workflows/ci.yml)
</p>

<p align="center">
  <strong><a href="https://PhilanthroPy-Project.github.io/PhilanthroPy/">🚀 View the Full Documentation Site</a></strong>
</p>

---

## What is PhilanthroPy?

PhilanthroPy is a production-ready Python library that slots directly into `sklearn.pipeline.Pipeline`. It covers the full predictive workflow for nonprofit and academic medical center (AMC) fundraising — from raw CRM cleaning and wealth imputation to major-gift propensity scoring, lapse prediction, and planned-giving intent.

---

## Installation

```bash
git clone https://github.com/PhilanthroPy-Project/PhilanthroPy.git
cd PhilanthroPy
pip install -e ".[dev]"
```

Or with Conda:
```bash
conda env create -f environment.yml && conda activate Philanthropy
pip install -e ".[dev]"
```

---

## Quick Start

```python
from philanthropy.datasets import generate_synthetic_donor_data
from philanthropy.models import DonorPropensityModel

df = generate_synthetic_donor_data(n_samples=500, random_state=42)
X = df[["total_gift_amount", "years_active", "event_attendance_count"]].to_numpy()
y = df["is_major_donor"].to_numpy()

model = DonorPropensityModel(n_estimators=200, random_state=0)
model.fit(X, y)
scores = model.predict_affinity_score(X)   # 0–100 affinity scale
```

---

## Feature Overview — v0.2.0

### 🧹 Preprocessing

| Transformer | Description |
|---|---|
| `CRMCleaner` | Standardise raw CRM exports — coerce types, strip whitespace, drop PII columns |
| `WealthScreeningImputer` | Leakage-safe wealth imputation (median / mean / zero) with optional missingness indicator columns; fill stats frozen at `fit()` time |
| `WealthPercentileTransformer` | Per-column wealth percentile rank (0–100); NaN-in → NaN-out |
| `FiscalYearTransformer` | Fiscal year & quarter features from gift dates; configurable fiscal year start month |
| `RFMTransformer` | Recency–Frequency–Monetary value feature engineering for donor segmentation |
| `EncounterTransformer` | Bridge hospital EHR encounter records with philanthropy CRM; computes `days_since_last_discharge` and `encounter_frequency_score`; snapshot at `fit()` prevents leakage |
| `GratefulPatientFeaturizer` ⭐ **NEW** | Clinical gravity score + AMC service-line capacity weights (cardiac 3.2×, oncology 2.9×, neuroscience 2.7×); outputs `clinical_gravity_score`, `distinct_service_lines`, `distinct_physicians`, `total_drg_weight` |
| `DischargeToSolicitationWindowTransformer` ⭐ **NEW** | Post-discharge solicitation window flags tracking recency; outputs `in_solicitation_window` (0/1), `window_position_score` (1.0 at midpoint, 0.0 at edges), and `discharge_recency_tier` (int 0-4); NaN → 0 |
| `PlannedGivingSignalTransformer` ⭐ **NEW** | Bequest / legacy-gift intent vector: `is_legacy_age`, `is_loyal_donor`, `inclination_score` (−1 sentinel for absent data), `composite_score` [0–3] |

### 🤖 Models

| Model | Description |
|---|---|
| `DonorPropensityModel` | Random Forest classifier with `predict_affinity_score()` returning a 0–100 scale |
| `MajorGiftClassifier` | Calibrated `HistGradientBoostingClassifier` — NaN-native, with `predict_affinity_score()` |
| `ShareOfWalletRegressor` | Estimates total giving capacity and untapped-potential ratio |
| `LapsePredictor` | Random Forest classifier for donor lapse; see full docs below |
| `MovesManagementClassifier` | Multi-class portfolio stage predictor |
| `PropensityScorer` | Lightweight logistic propensity baseline |
| `PlannedGivingIntentScorer` ⭐ **NEW** | Wraps GradientBoostingClassifier to predict bequest intent scores (0-100 scale) |

### 🔀 Model Selection

| Splitter | Description |
|---|---|
| `FiscalYearGroupedSplitter` ⭐ **NEW** | Walk-forward cross-validator preventing fiscal year data leakage |

### 📊 Metrics

| Function | Description |
|---|---|
| `donor_lifetime_value` | LTV with configurable discount rate |
| `retention_rate` | Period-over-period donor retention |
| `donor_acquisition_cost` | CAC from campaign spend data |

### 🗂 Datasets

| Function | Description |
|---|---|
| `generate_synthetic_donor_data` | Reproducible synthetic prospect pool — `n_samples`, `random_state` |

---

### LapsePredictor

Production Random Forest classifier for donor lapse risk. Predicts whether a donor will lapse within a configurable window.

| Parameter       | Type                 | Default | Description                |
|-----------------|----------------------|---------|----------------------------|
| n_estimators    | int                  | 100     | Number of trees            |
| max_depth       | int or None          | None    | Maximum tree depth         |
| min_samples_leaf| int                  | 1       | Min samples per leaf       |
| class_weight    | dict/"balanced"/None | None    | Class weighting            |
| lapse_window_years | int               | 2       | Prediction window in years |
| random_state    | int or None          | None    | Reproducibility seed       |

**Fitted attributes**

| Attribute   | Description                            |
|-------------|----------------------------------------|
| estimator_  | Trained RandomForestClassifier         |
| classes_    | Array of class labels                  |
| n_features_in_ | Feature count from fit              |

**Methods**

| Method               | Returns         | Description                               |
|----------------------|-----------------|-------------------------------------------|
| .fit(X, y)           | self            | Train on features and binary lapse label  |
| .predict(X)          | ndarray (n,)    | Binary predictions (1 = at-risk)          |
| .predict_proba(X)    | ndarray (n,2)   | Class probabilities                       |
| .predict_lapse_score(X) | ndarray (n,) | P(lapse) × 100, range [0.0, 100.0]        |

```python
from philanthropy.models import LapsePredictor

predictor = LapsePredictor(
    n_estimators=200,
    lapse_window_years=3,
    class_weight="balanced",
    random_state=42,
)
predictor.fit(X, y)
at_risk = predictor.predict(X)           # 1 = at risk of lapsing
scores  = predictor.predict_lapse_score(X)  # 0–100 lapse risk score
```

---

## sklearn Pipeline Example

```python
from sklearn.pipeline import Pipeline
from philanthropy.preprocessing import (
    FiscalYearTransformer, WealthScreeningImputer, DischargeToSolicitationWindowTransformer
)
from philanthropy.models import DonorPropensityModel

pipe = Pipeline([
    ("fy",      FiscalYearTransformer(date_col="gift_date")),
    ("wealth",  WealthScreeningImputer(wealth_cols=["estimated_net_worth"])),
    ("window",  DischargeToSolicitationWindowTransformer()),
    ("model",   DonorPropensityModel(n_estimators=200, random_state=0)),
])
pipe.fit(X_train, y_train)
scores = pipe.predict_proba(X_test)[:, 1]
```

---

## Medical Philanthropy Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from philanthropy.preprocessing import EncounterTransformer, CRMCleaner

# 1. EHR data extraction
encounter_features = EncounterTransformer(
    encounter_df=encounter_df,
    encounter_date_col="discharge_date",
    donor_id_col="mrn"
)

# 2. Use ColumnTransformer to merge EHR features with numeric CRM data
# This prevents the silent dropping of original CRM columns like estimated_net_worth
preprocessor = ColumnTransformer(
    transformers=[
        ("encounters", encounter_features, ["mrn", "gift_date"]),
        ("crm_nums", CRMCleaner(), ["estimated_net_worth", "real_estate_value"]),
    ],
    remainder="passthrough"
)

# 3. Fit pipeline on original dataset (gift_df)
gift_features = preprocessor.fit_transform(gift_df)
```

---

## Testing

| Test file                        | Tests | What's covered |
|----------------------------------|-------|----------------|
| test_datasets.py                 | 19    | (unchanged) |
| test_donor_propensity_model.py   | 84    | (unchanged) |
| test_preprocessing.py            | 35    | (unchanged) |
| test_leakage.py                  | 14    | WealthScreeningImputer API, fill-value freeze |
| test_metrics.py                  | 18    | donor_retention_rate, donor_acquisition_cost, donor_lifetime_value — edge cases |
| test_propensity.py               | 75    | PropensityScorer, LapsePredictor — full production coverage |
| test_utils.py                    | 4     | (unchanged) |
| test_share_of_wallet.py          | 25    | ShareOfWalletRegressor — capacity_floor, NaN inputs, predict_capacity_ratio |
| test_rfm_transformer.py          | 20    | RFMTransformer — recency/frequency/monetary, reference_date, leakage freeze |
| test_major_gift_classifier.py    | 20    | MajorGiftClassifier — calibrated proba, affinity score, check_estimator |
| test_visualisation.py            | 12    | plot_affinity_distribution headless, all public plot functions |

```bash
# Full suite
pytest tests/ -q

# sklearn check_estimator compliance
pytest tests/test_sklearn_compliance.py -q

# Property-based (Hypothesis)
pytest tests/test_properties.py tests/test_preprocessing_properties.py -q

# Leakage prevention
pytest tests/test_leakage.py -v
```

---

## Roadmap

### ✅ Completed

- `philanthropy.preprocessing.CRMCleaner` — leakage-safe CRM standardisation
- `philanthropy.preprocessing.WealthScreeningImputer` — median/mean/zero imputation
- `philanthropy.preprocessing.EncounterTransformer` — clinical discharge → feature engineering
- `philanthropy.preprocessing.RFMTransformer` — Recency, Frequency, Monetary features
- `philanthropy.models.ShareOfWalletRegressor` — capacity regression + predict_capacity_ratio()
- `philanthropy.models.MajorGiftClassifier` — gradient-boosted with calibrated probabilities
- `philanthropy.models.LapsePredictor` — production RF with predict_lapse_score()
- `philanthropy.metrics.donor_lifetime_value()` — NPV LTV with discount rate
- `philanthropy.visualisation` — affinity score plots, retention waterfall charts
- Property-based Hypothesis testing for FiscalYearTransformer
- Temporal leakage prevention test suite (test_leakage.py)
- GitHub Actions CI with Python 3.10/3.11 matrix

### 🔜 Next

- Full Sphinx documentation site (readthedocs.io deployment)
- PyPI package release (pip install philanthropy)
- `philanthropy.visualisation.plot_retention_waterfall()` — multi-year retention chart
- `philanthropy.visualisation.plot_capacity_heatmap()` — prospect pool heat map
- `philanthropy.preprocessing.CRMCleaner` — Salesforce NPSP and Veeva field-map presets
- `philanthropy.models.EnsemblePropensityModel` — stacked LapsePredictor + DonorPropensityModel

---

## Design Principles

- **Leakage-safe by design** — fill statistics, encounter summaries, and encounter snapshots are all frozen at `fit()` time; `transform()` is fully idempotent
- **sklearn-native** — all estimators pass `check_estimator`; support `set_output(transform="pandas")`, `clone()`, `get_params()` / `set_params()`
- **NaN-transparent** — wealth and clinical transformers declare `allow_nan = True`; no silent data loss
- **PII-aware** — `EncounterTransformer` auto-drops PII-like columns before returning features

---

## License

MIT License — see `LICENSE` for details.