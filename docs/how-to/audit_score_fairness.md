# Audit score fairness

Wealth and capacity features (estimated net worth, real-estate value, geography) can stand in for protected characteristics. A model that never sees a protected attribute can still produce disparate outcomes through those proxies. `philanthropy.metrics` gives you the two diagnostics for checking.

!!! warning "A diagnostic, not a clearance"
    A passing ratio does not certify a model as non-discriminatory. The choice of protected groups and of decision threshold materially changes the result. Involve your equity and compliance stakeholders before acting on scores.

## Threshold first, then audit

Both functions take **binary decisions**, not continuous scores. The threshold is part of what you are auditing: the same model can pass at one cut-off and fail at another.

```python
import numpy as np
import pandas as pd

from philanthropy.datasets import generate_synthetic_donor_data
from philanthropy.metrics import disparate_impact_ratio, selection_rate_by_group
from philanthropy.models import DonorPropensityModel

rng = np.random.default_rng(0)
df = generate_synthetic_donor_data(n_samples=1200, random_state=42)
cols = ["total_gift_amount", "years_active", "event_attendance_count"]

model = DonorPropensityModel(n_estimators=50, random_state=0).fit(
    df[cols].to_numpy(), df["is_major_donor"].to_numpy()
)
scores = model.predict_affinity_score(df[cols].to_numpy())

# A protected attribute the model never saw. In production this is a real CRM
# field, joined in only for the audit, not a feature.
region = rng.choice(["north", "south", "east", "west"], size=len(df))

flagged = (scores >= 70).astype(int)
rates = selection_rate_by_group(flagged, region)
print({g: round(r, 3) for g, r in sorted(rates.items())})
```

`selection_rate_by_group` returns the fraction selected within each group. Read it before the ratio: a ratio of 0.5 means something very different at rates of 0.80/0.40 than at 0.002/0.001.

## The four-fifths rule

`disparate_impact_ratio` is `min(selection_rate) / max(selection_rate)`. The US EEOC's four-fifths rule flags anything below `0.8` as evidence of adverse impact warranting investigation.

```python
ratio = disparate_impact_ratio(flagged, region)
print(f"disparate impact ratio: {ratio:.3f}")

if ratio < 0.8:
    print("below the four-fifths threshold; investigate before acting")

assert 0.0 <= ratio <= 1.0
# The ratio is exactly min/max over the per-group selection rates.
assert ratio == min(rates.values()) / max(rates.values())
```

The EEOC's own worked example: 80 of 100 selected in one group and 40 of 100 in another gives 0.50.

```python
y_pred = [1] * 80 + [0] * 20 + [1] * 40 + [0] * 60
groups = ["A"] * 100 + ["B"] * 100
assert disparate_impact_ratio(y_pred, groups) == 0.5
assert selection_rate_by_group(y_pred, groups) == {"A": 0.8, "B": 0.4}
```

## Sweep the threshold

The ratio is a function of where you cut. Audit the cut-off you actually plan to use, and look at the curve around it.

```python
sweep = pd.DataFrame([
    {
        "threshold": t,
        "flagged": int((scores >= t).sum()),
        "ratio": round(disparate_impact_ratio((scores >= t).astype(int), region), 3),
    }
    for t in (50, 60, 70, 80, 90)
])
print(sweep.to_string(index=False))
```

A ratio that swings across the sweep is telling you the disparity is threshold-driven, not structural. One that is flat and low is telling you the model has learned a proxy.

## Missing group labels are refused, not imputed

Silently dropping donors with an unknown protected attribute biases the audit toward whoever your CRM records completely. Both functions raise instead.

```python
import pytest

with pytest.raises(ValueError, match="missing"):
    disparate_impact_ratio([1, 0, 1], [np.nan, 1.0, 1.0])

with pytest.raises(ValueError, match="missing"):
    selection_rate_by_group([1, 0, 1], [np.nan, 1.0, 1.0])
```

Decide what an unknown group means for your audit and encode it explicitly, as its own `"unknown"` category, or by scoping the audit to the recorded population and saying so.

## What to do with a failing ratio

Not "drop the feature". Removing `estimated_net_worth` does not remove the proxy if `real_estate_value` and zip code carry the same signal. Use `philanthropy.inspection.donor_feature_importance` to find which features drive the gap, then decide with your stakeholders whether the disparity reflects genuine capacity differences or an artifact of your data.

```python
from philanthropy.inspection import donor_feature_importance

importance = donor_feature_importance(
    model, df[cols].to_numpy(), df["is_major_donor"].to_numpy(),
    feature_names=cols, random_state=0,
)
print(importance)
```

See [Compliance considerations](../explanation/compliance_considerations.md) before this reaches production.
