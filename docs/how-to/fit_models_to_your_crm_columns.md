# Fit models to your CRM columns

Every CRM export looks a little different: columns come in a different order, regions and appeal codes are categories rather than numbers, some records count for more than others, and your team knows things the data cannot teach a model ("more years of giving never makes a donor *more* likely to lapse"). Four options on the gradient-boosting models handle these:

| Option | What it does | Models |
|---|---|---|
| Column names | Point at a column by name, not position | `AskAmountRecommender` (`last_gift_idx`, `avg_gift_idx`), and dict keys in `monotonic_cst` |
| `categorical_features` | Split on category codes directly, no one-hot step | `AskAmountRecommender`, `ShareOfWalletRegressor`, `LapsePredictor`, `MovesManagementClassifier`, `PlannedGivingIntentScorer` |
| `sample_weight` in `fit` | Count some donors more than others | the same five |
| `monotonic_cst` | Force the score to only rise (or only fall) with a feature | `AskAmountRecommender`, `ShareOfWalletRegressor`, `LapsePredictor`, `PlannedGivingIntentScorer`, `MajorGiftClassifier` |

`LapsePredictor` uses these only with `backend="hist_gradient_boosting"`; its default random forest has no categorical splits or constraints.

All four default to `None`, which keeps the behaviour unchanged.

## A donor table with a category column

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 1500
X = pd.DataFrame({
    "last_gift": rng.lognormal(5, 0.8, n).round(2),
    "avg_gift": rng.lognormal(5, 0.6, n).round(2),
    "years_giving": rng.integers(1, 25, n).astype(float),
    "region": rng.choice(["Midwest", "Northeast", "South", "West"], n),
})
next_gift = np.maximum(X["last_gift"], X["avg_gift"]) * rng.lognormal(0.1, 0.3, n)
lapsed = (rng.random(n) < 1 / (1 + np.exp(0.15 * X["years_giving"] - 1))).astype(int)
```

## Category columns: encode as codes, then name them

The models need numbers, so turn each category into an integer code. Pass the column to `categorical_features` and the model splits on groups of codes ("Midwest or South") instead of treating the codes as an ordered scale. Codes must be non-negative integers below 255, which is plenty for regions, constituencies or appeal codes.

Keep the code mapping you fit with and reuse it when you score new donors, so "South" is the same code both times.

```python
regions = pd.CategoricalDtype(["Midwest", "Northeast", "South", "West"])
X["region"] = X["region"].astype(regions).cat.codes.astype(float)
```

## Column names instead of positions

`AskAmountRecommender` in relative mode needs to know which columns hold the last and average gift. Name them. The model looks the names up in the DataFrame you fit on, so a CRM export that reorders its columns next quarter still points at the right data.

```python
from philanthropy.models import AskAmountRecommender

ask = AskAmountRecommender(
    target_mode="relative",
    last_gift_idx="last_gift",
    avg_gift_idx="avg_gift",
    categorical_features=["region"],
    random_state=0,
).fit(X, next_gift)
print(ask.last_gift_idx_, ask.avg_gift_idx_)  # 0 1

reordered = X[["region", "years_giving", "avg_gift", "last_gift"]]
ask_reordered = AskAmountRecommender(
    target_mode="relative",
    last_gift_idx="last_gift",
    avg_gift_idx="avg_gift",
    categorical_features=["region"],
    random_state=0,
).fit(reordered, next_gift)
print(ask_reordered.last_gift_idx_, ask_reordered.avg_gift_idx_)  # 3 2

assert (ask.last_gift_idx_, ask_reordered.last_gift_idx_) == (0, 3)
```

A name needs a DataFrame: with a NumPy array there are no names to look up, and `fit` raises. Score with the same column order you fit with; sklearn checks the names and refuses a mismatch rather than reading the wrong column.

## Weights: count some donors more

Pass `sample_weight` to `fit` when rows should not count equally: you oversampled major-gift prospects and need to undo it, you want recent years to count more, or the file is a weighted survey sample.

```python
from philanthropy.models import LapsePredictor

weights = np.where(X["years_giving"] < 5, 2.0, 1.0)  # newer donors count double
lapse = LapsePredictor(
    backend="hist_gradient_boosting",
    categorical_features=["region"],
    random_state=0,
).fit(X, lapsed, sample_weight=weights)
```

A weight of 0 drops a donor from training without removing the row.

`PlannedGivingIntentScorer` also takes `class_weight="balanced"` now, for when bequest intent is too rare for the model to learn from unweighted. Like `MajorGiftClassifier`, it turns the scores into a ranking rather than a probability.

## Constraints: build in what your team already knows

A model trained on a few thousand donors can learn noise: a lapse score that dips at 11 years of giving and rises again at 12. Gift officers notice, and stop trusting the list. `monotonic_cst` rules that out. `1` means the score never falls as the feature rises, `-1` that it never rises, `0` (or leaving the feature out) no constraint. Key the dict by column name.

```python
constrained = LapsePredictor(
    backend="hist_gradient_boosting",
    categorical_features=["region"],
    monotonic_cst={"years_giving": -1},
    random_state=0,
).fit(X, lapsed)

# One donor, every value of years_giving from 1 to 25.
grid = pd.concat([X.iloc[[0]]] * 25, ignore_index=True)
grid["years_giving"] = np.arange(1, 26, dtype=float)
scores = constrained.predict_lapse_score(grid)
print(scores.round(1))

assert (np.diff(scores) <= 1e-9).all()  # never rises with tenure
```

Constrain only what you are sure of. A constraint the data contradicts costs accuracy, so compare the holdout score with and without it.

`AskAmountRecommender` enforces constraints only with `loss="squared_error"`. With its default `"absolute_error"` loss the backend recomputes each leaf after splitting, which can break the constraint, so `fit` raises instead of silently ignoring it.

```python
import pytest

with pytest.raises(ValueError, match="squared_error"):
    AskAmountRecommender(monotonic_cst={"avg_gift": 1}).fit(X, next_gift)

ask_constrained = AskAmountRecommender(
    loss="squared_error", monotonic_cst={"avg_gift": 1}, random_state=0
).fit(X, next_gift)
```

`MovesManagementClassifier` has no `monotonic_cst`: it predicts several stages at once, and sklearn does not support constraints for more than two classes.
