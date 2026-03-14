# Building a Grateful Patient Pipeline from EHR Export to Major Gift Score

Academic Medical Centers (AMCs) have a unique fundraising channel: grateful patients. Connecting hospital clinical encounters (from Electronic Health Records like Epic or Cerner) with CRM metrics unlocks powerful predictive capabilities.

This guide walks through using PhilanthroPy to ingest an EHR export, generate clinical features, and pass them into a propensity model.

## Step 1: Loading the Data

You will typicaly join two main tables:
- **`encounter_df`**: Patient-level hospital discharge data.
- **`donor_df`**: Gift transactions and CRM constituent metrics.

```python
import pandas as pd
from philanthropy.preprocessing import EncounterTransformer, CRMCleaner

# 1. Provide your data
encounter_df = pd.read_csv("ehr_encounters.csv")
donor_df = pd.read_csv("donor_crm_export.csv")
```

## Step 2: The Grateful Patient Pipeline Structure

We configure multiple steps: Extracting clinical encounters using `EncounterTransformer`, formatting CRM features using `CRMCleaner`, and passing both to a major gift propensity scorer.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from philanthropy.models import MajorGiftClassifier

# 1. Feature Engineering: Combining clinical and numerical wealth data
# EncounterTransformer extracts clinical summaries from the EHR based on donor IDs.
encounter_features = EncounterTransformer(
    encounter_df=encounter_df,
    encounter_date_col='discharge_date',
    donor_id_col='mrn'
)

# 2. Setup the Pipeline
# Notice how we can combine the output with the raw CRM data 
# without dropping crucial features like estimated_net_worth.
preprocessor = ColumnTransformer(
    transformers=[
        ("encounters", encounter_features, ["mrn", "gift_date"]),
        ("crm_clean", CRMCleaner(), ["estimated_net_worth", "real_estate_value"]),
    ],
    remainder='drop'
)

# 3. Create the Complete Flow
grateful_patient_pipeline = Pipeline([
    ("features", preprocessor),
    ("model", MajorGiftClassifier(random_state=42))
])

# Fit on our data
y_labels = donor_df["made_major_gift"].to_numpy()
grateful_patient_pipeline.fit(donor_df, y_labels)

# Generate major gift scores for a new campaign
prospects_df = pd.read_csv("new_prospects.csv")
scores = grateful_patient_pipeline.predict_affinity_score(prospects_df)
print(scores)
```

## Next Steps

Experiment with different models like `DonorPropensityModel` for random forest classifications, or evaluate your results using PhilanthroPy metrics!
