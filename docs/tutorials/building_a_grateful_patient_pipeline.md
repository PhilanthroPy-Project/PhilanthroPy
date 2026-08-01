# Building a Grateful Patient Pipeline from EHR Export to Major Gift Score

Academic medical centers (AMCs) have a fundraising channel most nonprofits don't: grateful patients. Linking clinical encounters from an electronic health record — Epic, Cerner — to your CRM metrics gives a propensity model signals it can learn from.

This tutorial takes an EHR export, turns it into clinical features, and passes them to a major-gift model. You build the pipeline one step at a time.

Every code block below runs as written and is executed in CI. In production you would read the two frames from CSV or your warehouse; here they are inline so the tutorial is self-contained.

## Step 1: Loading the Data

You start with two tables and join them:

- **`encounter_df`**: Patient-level hospital discharge data, one row per encounter.
- **`donor_df`**: Gift transactions and CRM constituent metrics, one row per donor.

`mrn` is the merge key. It never reaches the model — `EncounterTransformer` uses it to join and then drops it as an identifier.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n_donors = 200
mrns = [f"MRN{i:05d}" for i in range(n_donors)]

# In production: encounter_df = pd.read_csv("ehr_encounters.csv")
encounter_df = pd.DataFrame({
    "mrn": rng.choice(mrns[:150], size=400),
    "discharge_date": pd.to_datetime("2021-01-01")
    + pd.to_timedelta(rng.integers(0, 900, 400), unit="D"),
    "service_line": rng.choice(
        ["cardiac", "oncology", "neuroscience", "general"], size=400
    ),
})

# In production: donor_df = pd.read_csv("donor_crm_export.csv")
donor_df = pd.DataFrame({
    "mrn": mrns,
    "gift_date": pd.to_datetime("2023-01-01")
    + pd.to_timedelta(rng.integers(0, 365, n_donors), unit="D"),
    "estimated_net_worth": np.where(
        rng.random(n_donors) < 0.3, np.nan, rng.lognormal(13, 1.5, n_donors)
    ),
    "real_estate_value": rng.lognormal(12, 1.2, n_donors),
})

encounters_per_donor = encounter_df["mrn"].value_counts()
donor_df["made_major_gift"] = (
    donor_df["mrn"].map(encounters_per_donor).fillna(0) >= 3
).astype(int)

print(donor_df.shape, encounter_df.shape, donor_df["made_major_gift"].mean())
```

## Step 2: The Grateful Patient Pipeline Structure

The pipeline has three parts: pull clinical encounters with `EncounterTransformer`, format CRM features with `CRMCleaner`, and pass both to a major-gift propensity scorer.

Route with a `ColumnTransformer`, not a serial `Pipeline`. Each transformer here consumes named columns and replaces them, so chaining them in series would hand the second one the first one's output under the wrong names — which produces a constant feature block and a model that trains on nothing while exiting 0.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from philanthropy.models import MajorGiftClassifier
from philanthropy.preprocessing import CRMCleaner, EncounterTransformer

# 1. Feature engineering: clinical encounters joined on mrn.
encounter_features = EncounterTransformer(
    encounter_df=encounter_df,
    discharge_col="discharge_date",
    merge_key="mrn",
)

# 2. Each branch gets exactly the columns it needs. remainder="drop" keeps the
#    identifier out of the matrix; the wealth columns survive because CRMCleaner
#    has its own branch rather than sitting downstream of the encounter step.
preprocessor = ColumnTransformer(
    transformers=[
        ("encounters", encounter_features, ["mrn", "gift_date"]),
        ("crm_clean", CRMCleaner(), ["estimated_net_worth", "real_estate_value"]),
    ],
    remainder="drop",
)

# 3. The complete flow. MajorGiftClassifier handles NaN natively, so the
#    missing net-worth values need no upstream imputation.
grateful_patient_pipeline = Pipeline([
    ("features", preprocessor),
    ("model", MajorGiftClassifier(max_iter=50, random_state=42)),
])

y_labels = donor_df["made_major_gift"].to_numpy()
grateful_patient_pipeline.fit(donor_df, y_labels)
```

## Step 3: Scoring a new campaign

Custom methods like `predict_affinity_score` are not proxied through an sklearn `Pipeline`, so score via the delegated `predict_proba` on the positive class.

```python
# In production: prospects_df = pd.read_csv("new_prospects.csv")
prospects_df = donor_df.drop(columns=["made_major_gift"]).head(10)

scores = grateful_patient_pipeline.predict_proba(prospects_df)[:, 1]
print(scores.round(3))

# A constant score means the features never reached the model — the exact
# failure the ColumnTransformer above prevents. Assert it, don't eyeball it.
assert scores.shape == (10,)
assert len(set(scores.round(6))) > 1, "degenerate pipeline: every score identical"
```

## Next Steps

Try other models — `DonorPropensityModel` gives you random forest classification — or evaluate your results with the PhilanthroPy metrics. To add service-line weighting on top of raw encounter counts, see [Build grateful patient features](../how-to/build_grateful_patient_features.md).
