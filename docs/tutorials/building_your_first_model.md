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

PhilanthroPy components drop into scikit-learn pipelines as-is.

```python
from sklearn.pipeline import Pipeline
from philanthropy.preprocessing import FiscalYearTransformer, WealthScreeningImputer, DischargeToSolicitationWindowTransformer
from philanthropy.models import DonorPropensityModel

pipe = Pipeline([
    ("fy", FiscalYearTransformer(date_col="gift_date")),
    ("wealth", WealthScreeningImputer(wealth_cols=["estimated_net_worth"])),
    ("window", DischargeToSolicitationWindowTransformer()),
    ("model", DonorPropensityModel(n_estimators=200, random_state=0)),
])

# Assuming X_train, y_train, X_test are populated with appropriate data:
# pipe.fit(X_train, y_train)
# scores = pipe.predict_proba(X_test)[:, 1]
```
