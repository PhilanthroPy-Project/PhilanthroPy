---
description: "How temporal leakage inflates donor model backtests, and how as-of feature timing in PhilanthroPy's transformers and models prevents it."
---

# Avoiding temporal data leakage in fundraising models

Predicting major gifts or donor retention carries one common trap: **temporal data leakage**, using information from the future to predict an outcome in the past. PhilanthroPy's transformers and models are leakage-safe by design.

This tutorial shows how temporal leakage happens and how PhilanthroPy prevents it.

## The problem: naive aggregation

Marianne Pelletier of Staupell Analytics Group describes the sharpest version of it, from her own modelling work:

> I once modeled new donors with the 100% correlated variable of their having a greater than 0 lifetime giving total!

The feature was the outcome. Validation looked perfect and the field did not, because a prospective new donor's lifetime giving is 0 as of the scoring date: cut to that date the feature goes constant rather than perfect. Her prescription is the rule this library enforces, roll-up variables that count up to the day before the event. She puts it as modelling baseball, where you build the stats that were true the day before the game and model on those. (Quoted with her permission.)

The same trap catches any feature built by aggregating a source table that runs past the outcome: a `total_lifetime_giving` attached to historical donor snapshots, a wealth-capacity field refreshed to today's value, a clinical encounter that had not happened yet.

## The cutoff: `as_of`

Every transformer that reads a dated source table takes an `as_of` date and removes the rows dated after it *before* any feature is computed. `RFMTransformer` rolls a gift log up to one row per donor, so it is where Pelletier's case lands:

```python
import pandas as pd
from philanthropy.preprocessing import RFMTransformer

gifts = pd.DataFrame({
    'donor_id': [1, 1, 1],
    'gift_date': ['2020-01-01', '2021-01-01', '2025-01-01'],
    'gift_amount': [100.0, 100.0, 50_000.0],
})

# Scoring as of 2022: the 2025 gift had not been given yet.
rfm = RFMTransformer(reference_date='2022-01-01', as_of='2022-01-01')
print(rfm.fit_transform(gifts))
#    donor_id  recency  frequency  monetary
# 0         1      365          2      200.0
```

`as_of` is inclusive, so to get Pelletier's day-before rule exactly, pass the day before the outcome you are predicting: `as_of=D` still rolls a gift made on `D` into `monetary`.

Leave `as_of` unset while the gift table runs past the reference date and the transformer warns rather than quietly aggregating the future. Without the cutoff the same donor rolls up to `frequency=3`, `monetary=50200.0` and a *negative* recency of -1096 days: the model is being told about a gift from three years after the date it is scoring.

`EncounterTransformer` and `GratefulPatientFeaturizer` take the same `as_of` argument for clinical encounter tables. `EncounterRecencyTransformer` takes it too, with one difference: it emits one row per input row, so dropping rows would break it inside a `Pipeline`. It blanks post-cutoff encounters to `NaT` instead, and they come out as a missing encounter rather than a future one.

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
4. **Set `as_of`**: Pass the end of your training window to every transformer that reads a dated table: `RFMTransformer`, `EncounterTransformer`, `GratefulPatientFeaturizer`, `EncounterRecencyTransformer`. Cross-validation cannot catch a leak that happens inside one donor's own features.
