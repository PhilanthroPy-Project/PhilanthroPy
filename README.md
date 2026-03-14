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
  <img src="https://img.shields.io/badge/tests-passing-success" alt="tests"/>
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
| `LapsePredictor` | Random Forest classifier for donor lapse; standard fit(X,y), `predict_lapse_score()` 0–100 |
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

## Design Principles

- **Leakage-safe by design** — fill statistics, encounter summaries, and encounter snapshots are all frozen at `fit()` time; `transform()` is fully idempotent
- **sklearn-native** — all estimators pass `check_estimator`; support `set_output(transform="pandas")`, `clone()`, `get_params()` / `set_params()`
- **NaN-transparent** — wealth and clinical transformers declare `allow_nan = True`; no silent data loss
- **PII-aware** — `EncounterTransformer` auto-drops PII-like columns before returning features

---

## License

MIT License — see `LICENSE` for details.