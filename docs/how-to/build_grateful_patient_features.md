# Build Grateful Patient Features

Academic Medical Centers (AMCs) track hospital visits across different service lines. Not all hospital visits are equal for philanthropy (e.g., Oncology correlates more strongly with giving than Urgent Care).

## Using `GratefulPatientFeaturizer`

This HIPAA-safe transformer translates clinical encounter histories into powerful signals.

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

### Clinical Gravity Scores
The `GratefulPatientFeaturizer` automatically weights occurrences. You can provide a `service_line_weights` dictionary to prioritize specific clinical areas instead of using the defaults derived from AMC datasets.

## The Solicitation Window

Patients in a 6-to-24 month window post-discharge are often the warmest prospects. The `SolicitationWindowTransformer` measures proximity to this optimal sweet spot perfectly:

```python
from philanthropy.preprocessing import SolicitationWindowTransformer

window = SolicitationWindowTransformer()
# X_out = window.fit_transform(X)
```
