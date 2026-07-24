# Avoiding temporal data leakage in fundraising models

Predicting major gifts or donor retention carries one common trap: **temporal data leakage** — using information from the future to predict an outcome in the past. PhilanthroPy's transformers and models are leakage-safe by design.

This tutorial shows how temporal leakage happens and how PhilanthroPy prevents it.

## The problem: naive aggregation

Say you build a feature `total_lifetime_giving` and attach it to historical snapshots of each donor. If you use the final, present-day `total_lifetime_giving` value to predict whether a donor gave a major gift three years ago, you have trained the model on the future.

## The solution: fit-time snapshots

PhilanthroPy transformers compute and freeze their aggregations during `fit()`. Call `transform()` on new or temporal data, and it uses only the statistics frozen at `fit()` time.

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
    discharge_col='encounter_date'
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

## Best practices

1. **Split first**: Split your data into training and test sets *before* passing them to a pipeline.
2. **Use pipelines**: Wrap your transformers inside a `sklearn.pipeline.Pipeline`.
3. **Use temporal splits**: For time-series data like fundraising, reach for `FiscalYearGroupedSplitter` (see the CV documentation) so test folds fall strictly after training folds in time.
