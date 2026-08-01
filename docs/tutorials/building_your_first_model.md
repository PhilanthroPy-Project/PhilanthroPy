# Building Your First Model

Build a working machine learning pipeline with PhilanthroPy and scikit-learn. You start with raw donor data and finish with affinity scores a gift officer can act on.

## 1. Installation

Install PhilanthroPy with pip:

```bash
pip install philanthropy
```

## 2. Generating Synthetic Data

Start with synthetic donor data that stands in for a real CRM export.

```python
from philanthropy.datasets import generate_synthetic_donor_data

df = generate_synthetic_donor_data(n_samples=500, random_state=42)
X = df[["total_gift_amount", "years_active", "event_attendance_count"]].to_numpy()
y = df["is_major_donor"].to_numpy()
```

## 3. Creating a Basic Model

Pass the data straight into `DonorPropensityModel`.

```python
from philanthropy.models import DonorPropensityModel

model = DonorPropensityModel(n_estimators=200, random_state=0)
model.fit(X, y)
```

## 4. Predicting Affinity

The model returns an affinity score from 0-100 instead of a raw probability. That scale is easier for a gift officer to act on.

```python
scores = model.predict_affinity_score(X)
print(scores[:5])
```

## 5. Using Pipelines

PhilanthroPy components drop into scikit-learn pipelines as-is — but **route them with a `ColumnTransformer`, not in series.**

Each of these transformers consumes named columns and *replaces* every other column in its output. Chain them serially and `FiscalYearTransformer` hands its two-column `[fiscal_year, fiscal_quarter]` block to `WealthScreeningImputer`, which finds no wealth column and no-ops with a warning, and then to `DischargeToSolicitationWindowTransformer`, which now raises because `days_since_last_discharge` is gone. Before that guard existed the same mistake was silent: every feature came out `0.0` and the pipeline exited cleanly on a constant matrix.

A `ColumnTransformer` gives each transformer exactly its own columns and concatenates the results.

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from philanthropy.models import DonorPropensityModel
from philanthropy.preprocessing import (
    DischargeToSolicitationWindowTransformer,
    FiscalYearTransformer,
    WealthScreeningImputer,
)

rng = np.random.default_rng(0)
n = 400
raw = pd.DataFrame({
    "gift_date": pd.to_datetime("2020-01-01")
    + pd.to_timedelta(rng.integers(0, 2000, n), unit="D"),
    "estimated_net_worth": np.where(
        rng.random(n) < 0.3, np.nan, rng.lognormal(13, 1.5, n)
    ),
    "days_since_last_discharge": rng.integers(0, 800, n).astype(float),
})
labels = (
    (raw["days_since_last_discharge"].between(90, 365))
    & (raw["estimated_net_worth"].fillna(0) > 5e5)
).astype(int).to_numpy()

preprocessor = ColumnTransformer([
    ("fy", FiscalYearTransformer(date_col="gift_date"), ["gift_date"]),
    ("wealth", WealthScreeningImputer(wealth_cols=["estimated_net_worth"]),
     ["estimated_net_worth"]),
    ("window", DischargeToSolicitationWindowTransformer(),
     ["days_since_last_discharge"]),
])

pipe = Pipeline([
    ("features", preprocessor),
    ("model", DonorPropensityModel(n_estimators=200, random_state=0)),
])

pipe.fit(raw, labels)
scores = pipe.predict_proba(raw)[:, 1]
print(scores[:5].round(3))

# Regression guard: a constant score means the features never reached the
# model. The serial pipeline above would fail this.
assert len(set(scores.round(6))) > 1
```
