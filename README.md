<p align="center">
  <img src="docs/logo.png" alt="PhilanthroPy logo" width="180"/>
</p>

<p align="center">
  <strong>A scikit-learn–compatible toolkit for predictive donor analytics in the nonprofit and medical philanthropy sector.</strong>
</p>

## Overview

PhilanthroPy provides a production-ready Python library that slots directly into a `sklearn.pipeline.Pipeline`. It offers end-to-end support for predictive analytics workflows in philanthropy—from raw CRM data cleaning and imputation to major-gift propensity scoring and share-of-wallet estimation.

## Key Features

- **Robust Preprocessing**: Includes `CRMCleaner` for raw CRM standardisation and `WealthScreeningImputer` for leakage-safe wealth imputation.
- **Fiscal Calendar Engineering**: `FiscalYearTransformer` generates fiscal year/quarter features.
- **Clinical Data Integration**: `EncounterTransformer` natively bridges hospital clinical data with philanthropy CRM records.
- **Actionable Modelling**: Features `DonorPropensityModel` for 0–100 affinity scores and `ShareOfWalletRegressor` for total capacity estimates and untapped-potential ratios.
- **Fundraising KPIs**: Pure functions for tracking LTV, donor retention rate, and donor acquisition cost.

## Installation

You can install PhilanthroPy using the pinned Conda environment to replicate the development setup:

```bash
git clone https://github.com/PhilanthroPy-Project/PhilanthroPy.git
cd PhilanthroPy
conda env create -f environment.yml
conda activate Philanthropy
pip install -e ".[dev]"
```

Or via `pip`:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from philanthropy.datasets import generate_synthetic_donor_data
from philanthropy.models import DonorPropensityModel

# 1. Generate a synthetic prospect pool
df = generate_synthetic_donor_data(n_samples=500, random_state=42)
X = df[["total_gift_amount", "years_active", "event_attendance_count"]].to_numpy()
y = df["is_major_donor"].to_numpy()

# 2. Train the affinity model
model = DonorPropensityModel(n_estimators=200, class_weight="balanced", random_state=0)
model.fit(X, y)

# 3. Score prospects on a 0–100 scale
scores = model.predict_affinity_score(X)
df["affinity_score"] = scores
print(df.sort_values("affinity_score", ascending=False).head())
```

## Package Structure

- `philanthropy.preprocessing`: Transformers for CRM, Wealth, RFM, and Encounters.
- `philanthropy.models`: Scikit-learn regressors and classifiers tailored for advanced placement metrics.
- `philanthropy.metrics`: Key performance metrics (LTV, Retention, DAG).
- `philanthropy.datasets`: Generators for synthetic hospital or charity data.

## Documentation and Code Quality

PhilanthroPy enforces strict property-based tests via `hypothesis` and guarantees 100% compliance with scikit-learn's `check_estimator` utility, ensuring that everything integrates without a hitch into standard machine learning ecosystems.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.