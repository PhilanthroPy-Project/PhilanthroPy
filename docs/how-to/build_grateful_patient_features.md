# Build grateful patient features

Academic medical centers (AMCs) track hospital visits across different service lines. Those visits don't count equally for philanthropy: an oncology encounter correlates more strongly with giving than an urgent care visit. This guide turns encounter histories into model-ready features.

## Using `GratefulPatientFeaturizer`

`GratefulPatientFeaturizer` turns clinical encounter histories into predictive signals, drawn from encounter metadata alone: service line, attending physician, and dates. It reads only those columns and returns four numeric aggregates, so no identifier from the encounter table reaches the model. That is a narrow read surface, **not** formal HIPAA de-identification; if you need identifier-like columns dropped from a wider encounter frame, use `EncounterTransformer` and its `pii_patterns`. Review [Compliance Considerations](../explanation/compliance_considerations.md) before production use.

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
print(pd.DataFrame(out, columns=gpf.get_feature_names_out()))
```

`transform` returns a plain `(n_samples, 4)` float array, never a DataFrame, with these columns in this order:

| Column | Meaning |
|---|---|
| `clinical_gravity_score` | Encounter count weighted by service-line multiplier. |
| `distinct_service_lines` | Number of distinct service lines the donor was seen in. |
| `distinct_physicians` | Number of distinct attending physicians. |
| `total_drg_weight` | Sum of `drg_weight_col`, or `0.0` when that column is unset. |

Donors with no encounters (donor 3 above) get `0.0` across all four.

### Clinical gravity scores
`GratefulPatientFeaturizer` weights each encounter by service line. It applies illustrative default multipliers when `use_capacity_weights=True` (the default); to prioritize specific clinical areas, pass your own `capacity_weights` dictionary of `{service_line: multiplier}`.

## The solicitation window

Patients in a 90-to-365 day window post-discharge are often the warmest prospects. `DischargeToSolicitationWindowTransformer` scores each donor's proximity to that window. It reads `days_since_discharge_col` by name and **raises** if a DataFrame does not carry it, so route it with a `ColumnTransformer` rather than chaining it behind a transformer that renames columns:

```python
from philanthropy.preprocessing import DischargeToSolicitationWindowTransformer

recency = pd.DataFrame({"days_since_last_discharge": [10.0, 200.0, 400.0]})

window = DischargeToSolicitationWindowTransformer()
scored = window.fit_transform(recency)
print(pd.DataFrame(scored, columns=window.get_feature_names_out()))
```

`in_solicitation_window` is 1 only for the 200-day row; `window_position_score` peaks at the window midpoint (227.5 days) and falls to 0 at either edge.
