# Recommend ask amounts

A gift officer walks into a meeting with three numbers, not one: a conservative ask they expect to be accepted, a target, and a stretch. `AskAmountRecommender` predicts the base ask from giving history and capacity, and `ask_ladder` expands it into those rungs.

## Fit the recommender

`AskAmountRecommender` is a regressor: `y` is the dollar amount you consider the right ask for each historical prospect — most teams use the largest gift the donor actually made, or the ask that closed.

```python
import numpy as np
import pandas as pd

from philanthropy.models import AskAmountRecommender

rng = np.random.default_rng(0)
n = 300
X = pd.DataFrame({
    "largest_prior_gift": rng.lognormal(7, 1.0, n),
    "total_gift_amount": rng.lognormal(8, 1.2, n),
    "years_active": rng.integers(1, 30, n).astype(float),
    "estimated_net_worth": rng.lognormal(13, 1.5, n),
})
# The ask that actually closed, historically.
y = X["largest_prior_gift"] * rng.uniform(1.0, 1.8, n)

model = AskAmountRecommender(max_iter=100, random_state=0).fit(X, y)

base_ask = model.predict(X.head(5))
print(base_ask.round(0))
```

## Expand into a ladder

`ask_ladder` multiplies the base ask by each entry of `multipliers` and returns one row per prospect, one column per rung.

```python
ladder = model.ask_ladder(X.head(5))
print(pd.DataFrame(ladder.round(0), columns=["conservative", "target", "stretch"]))

assert ladder.shape == (5, 3)
# Rungs are strictly ascending for every prospect.
assert (np.diff(ladder, axis=1) > 0).all()
```

The default `multipliers=(1.0, 1.5, 2.5)` is a heuristic, not a benchmark. Override it with your own campaign's escalation policy — values must be positive, and passing them ascending is what makes the columns read as a ladder.

```python
aggressive = model.ask_ladder(X.head(5), multipliers=(1.0, 2.0, 4.0, 8.0))
print(aggressive.shape)
assert aggressive.shape == (5, 4)
```

!!! note "`ask_ladder`, not `predict_ask_array`"
    The method was called `predict_ask_array` before 0.6.0, deprecated in 0.6.0, and **removed in 0.7.0**. The `predict_` prefix is reserved for methods returning one value per row; this one returns a `(n, 3)` dollar matrix.

## Sequencing the portfolio

Pair the ladder with `MovesManagementClassifier.action_priority`, which says *where in the lifecycle* each donor is, so you know whether to make the ask at all.

```python
from philanthropy.models import MovesManagementClassifier

stages = np.asarray(["IDENTIFY", "QUALIFY", "CULTIVATE", "SOLICIT"] * (n // 4))
moves = MovesManagementClassifier(max_iter=50, random_state=0).fit(X, stages)

priority = moves.action_priority(X.head(5))
plan = pd.DataFrame({
    "stage": priority["stage"],
    "confidence": priority["confidence"].round(2),
    "target_ask": ladder[:, 1].round(0),
})
print(plan)
print(priority["portfolio_summary"])
```

`action_priority` returns a dict, not an array — `stage` and `confidence` are per-donor, `portfolio_summary` counts donors per stage across the whole batch. Solicit the `SOLICIT` rows at the target rung; the `CULTIVATE` rows are not ready for a number yet.

!!! note "`action_priority`, not `predict_action_priority`"
    Same rename, same reason: it returns a dict, so it never satisfied the `predict_*` contract. The old name was removed in 0.7.0.
