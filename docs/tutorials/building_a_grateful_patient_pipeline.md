# Building a Grateful Patient Pipeline from EHR Export to Major Gift Score

Academic medical centers (AMCs) have a fundraising channel most nonprofits don't: grateful patients. Linking clinical encounters from an electronic health record — Epic, Cerner — to your CRM metrics gives a propensity model signals it can learn from.

This tutorial takes an EHR export, turns it into clinical features, and passes them to a major-gift model. You build the pipeline one step at a time.

## Step 1: Loading the Data

You start with two tables and join them:

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

The pipeline has three parts: pull clinical encounters with `EncounterTransformer`, format CRM features with `CRMCleaner`, and pass both to a major-gift propensity scorer.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from philanthropy.models import MajorGiftClassifier

# 1. Feature Engineering: Combining clinical and numerical wealth data
# EncounterTransformer extracts clinical summaries from the EHR based on donor IDs.
encounter_features = EncounterTransformer(
    encounter_df=encounter_df,
    discharge_col='discharge_date',
    merge_key='mrn'
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

# Generate major gift scores for a new campaign.
# Custom methods like predict_affinity_score are not proxied through an sklearn
# Pipeline, so score via the delegated predict_proba on the positive class.
prospects_df = pd.read_csv("new_prospects.csv")
scores = grateful_patient_pipeline.predict_proba(prospects_df)[:, 1]
print(scores)
```

## Next Steps

Try other models — `DonorPropensityModel` gives you random forest classification — or evaluate your results with the PhilanthroPy metrics.
