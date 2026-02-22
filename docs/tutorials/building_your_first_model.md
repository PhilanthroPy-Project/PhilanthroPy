# Building Your First Model

This tutorial will guide you through building a basic machine learning pipeline with PhilanthroPy and scikit-learn.

## 1. Installation

You can install PhilanthroPy using pip:

```bash
pip install philanthropy
```

## 2. Generating Synthetic Data

Let's begin by generating some synthetic donor data. This simulates a real CRM export.

```python
from philanthropy.datasets import generate_synthetic_donor_data

df = generate_synthetic_donor_data(n_samples=500, random_state=42)
X = df[["total_gift_amount", "years_active", "event_attendance_count"]].to_numpy()
y = df["is_major_donor"].to_numpy()
```

## 3. Creating a Basic Model

We can pass this data directly into the `DonorPropensityModel`.

```python
from philanthropy.models import DonorPropensityModel

model = DonorPropensityModel(n_estimators=200, random_state=0)
model.fit(X, y)
```

## 4. Predicting Affinity

Instead of raw probabilities, the model provides an affinity score from 0-100, which is much more actionable for gift officers.

```python
scores = model.predict_affinity_score(X)
print(scores[:5])
```

## 5. Using Pipelines

PhilanthroPy components drop directly into scikit-learn pipelines.

```python
from sklearn.pipeline import Pipeline
from philanthropy.preprocessing import FiscalYearTransformer, WealthScreeningImputer, SolicitationWindowTransformer
from philanthropy.models import DonorPropensityModel

pipe = Pipeline([
    ("fy", FiscalYearTransformer(date_col="gift_date")),
    ("wealth", WealthScreeningImputer(wealth_cols=["estimated_net_worth"])),
    ("window", SolicitationWindowTransformer()),
    ("model", DonorPropensityModel(n_estimators=200, random_state=0)),
])

# Assuming X_train, y_train, X_test are populated with appropriate data:
# pipe.fit(X_train, y_train)
# scores = pipe.predict_proba(X_test)[:, 1]
```
