# Avoiding Temporal Data Leakage in Fundraising Models

One of the most common pitfalls in predicting major gifts or donor retention is **temporal data leakage** — using information from the future to predict an outcome in the past. In PhilanthroPy, transformers and models are designed to be "leakage-safe by design." 

This tutorial demonstrates how temporal leakage happens and how PhilanthroPy prevents it.

## The Problem: Naive Aggregation

Imagine creating a feature `total_lifetime_giving` and appending it to historical snapshots of donors. If you use the final, present-day `total_lifetime_giving` value when predicting whether a donor gave a major gift three years ago, you have trained your model on future information.

## The Solution: Fit-Time Snapshots

PhilanthroPy transformers compute and freeze aggregrations during `fit()`. When you call `transform()` on new or temporal data, it uses only the frozen statistics from `fit()`.

```python
from philanthropy.preprocessing import EncounterTransformer
import pandas as pd

# Assume we have an EHR encounter DataFrame
encounter_df = pd.DataFrame({
    'donor_id': [1, 1, 2],
    'encounter_date': ['2020-01-01', '2022-01-01', '2021-06-01'],
    'department': ['Cardiology', 'Oncology', 'Neurology']
})

donor_df_train = pd.DataFrame({
    'donor_id': [1, 2],
    'gift_date': ['2023-01-01', '2023-01-01']
})

# EncounterTransformer calculates statistics (like recency or frequency) 
# relative to the gift dates present AT FIT TIME.
transformer = EncounterTransformer(
    encounter_df=encounter_df, 
    encounter_date_col='encounter_date'
)

# Statistics are frozen here using the gift dates in the training set
transformer.fit(donor_df_train)

# When evaluating on a future test set, we do NOT reach into the future
donor_df_test = pd.DataFrame({
    'donor_id': [1],
    'gift_date': ['2024-01-01']  # Future gift date
})

# Evaluated strictly using knowledge available up to the training period boundary
features_test = transformer.transform(donor_df_test)
```

## Best Practices

1. **Split First**: Always split your data into training and test sets *before* passing them to a pipeline.
2. **Use Pipelines**: Encapsulate transformers inside a `sklearn.pipeline.Pipeline`.
3. **Use Temporal Splits**: For time-series data like fundraising, consider the `FiscalYearGroupedSplitter` (see the CV documentation) to ensure test folds strictly succeed training folds in time.
