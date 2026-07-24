# Build grateful patient features

Academic medical centers (AMCs) track hospital visits across different service lines. Those visits don't count equally for philanthropy — an oncology encounter correlates more strongly with giving than an urgent care visit. This guide turns encounter histories into model-ready features.

## Using `GratefulPatientFeaturizer`

`GratefulPatientFeaturizer` turns clinical encounter histories into predictive signals, drawn from encounter metadata alone — service line, attending physician, and dates. It drops identifier-like columns by name as defense-in-depth — **not** formal HIPAA de-identification. Review [Compliance Considerations](../explanation/compliance_considerations.md) before production use.

```python
import pandas as pd
from philanthropy.preprocessing import GratefulPatientFeaturizer

# Clinical history (no Patient Identifiers needed for output features)
encounters = pd.DataFrame({
    "donor_id": [1, 1, 2],
    "discharge_date": ["2022-01-01", "2023-06-15", "2022-09-30"],
    "service_line": ["cardiac", "cardiac", "oncology"],
    "attending_physician_id": ["P1", "P2", "P3"],
})

# Donors in our current dataset
X = pd.DataFrame({"donor_id": [1, 2, 3]})

gpf = GratefulPatientFeaturizer(encounter_df=encounters)
out = gpf.fit_transform(X)
print(out)
```

### Clinical gravity scores
`GratefulPatientFeaturizer` weights each encounter by service line. It applies illustrative default multipliers when `use_capacity_weights=True` (the default); to prioritize specific clinical areas, pass your own `capacity_weights` dictionary of `{service_line: multiplier}`.

## The solicitation window

Patients in a 6-to-24 month window post-discharge are often the warmest prospects. `DischargeToSolicitationWindowTransformer` scores each donor's proximity to that window:

```python
from philanthropy.preprocessing import DischargeToSolicitationWindowTransformer

window = DischargeToSolicitationWindowTransformer()
# X_out = window.fit_transform(X)
```
