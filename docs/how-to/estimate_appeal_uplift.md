# Estimate appeal uplift

A propensity model ranks who is *likely to give*. That is not the same as who gives *because you asked*. Some donors would have given anyway; soliciting them costs money and goodwill for nothing. A few are "sleeping dogs": the appeal actively suppresses their giving.

`UpliftTLearner` estimates the causal lift of the appeal itself: `P(give | solicited) − P(give | not solicited)`.

!!! warning "Tier 3: Experimental"
    `UpliftTLearner` lives in `philanthropy.experimental`. Its `fit(X, y, treatment)` signature breaks the sklearn `fit(X, y)` contract, so it is not `check_estimator` compliant, it cannot go into a `Pipeline`, and its API may change without a deprecation cycle. Do not build production infrastructure on it.

## You need a randomised holdout

Uplift modelling is causal inference, not curve fitting. It requires a `treatment` column recording who *actually received* the appeal, and that assignment must have been random. If your team solicited whoever looked most promising, the treated and control arms differ systematically and the "uplift" you measure is selection bias.

Hold out a random control group in the appeal itself. There is no way to recover this after the fact.

```python
import numpy as np
import pandas as pd

from philanthropy.experimental import UpliftTLearner

rng = np.random.default_rng(0)
n = 1200

X = pd.DataFrame({
    "years_active": rng.integers(1, 30, n).astype(float),
    "total_gift_amount": rng.lognormal(6, 1.0, n),
    "event_attendance_count": rng.integers(0, 12, n).astype(float),
})

# Randomised: a coin flip decides who received the appeal.
treatment = rng.integers(0, 2, n)

# Simulated outcome. Loyal donors give regardless (low uplift); newer donors
# respond to being asked (high uplift).
baseline = 0.15 + 0.02 * np.minimum(X["years_active"], 20)
lift = np.where(X["years_active"] < 5, 0.25, 0.02)
p_give = baseline + treatment * lift
y = (rng.random(n) < p_give).astype(int)

model = UpliftTLearner(n_estimators=100, random_state=0)
model.fit(X.to_numpy(), y, treatment)
```

Both arms must be non-empty and `treatment` must be binary `{0, 1}`; anything else raises.

```python
import pytest

with pytest.raises(ValueError, match="Both arms must be present"):
    UpliftTLearner(n_estimators=10, random_state=0).fit(
        X.to_numpy(), y, np.ones(n, dtype=int)
    )

with pytest.raises(ValueError, match="binary"):
    UpliftTLearner(n_estimators=10, random_state=0).fit(
        X.to_numpy(), y, rng.integers(0, 3, n)
    )
```

## Read the score on its own scale

`predict_uplift_score` returns a value in **[−1, 1]**, not the 0–100 scale the propensity models use. Zero means the appeal makes no difference.

```python
uplift = model.predict_uplift_score(X.to_numpy())
print(f"range: [{uplift.min():.3f}, {uplift.max():.3f}]  mean: {uplift.mean():.3f}")

assert uplift.shape == (n,)
assert ((uplift >= -1.0) & (uplift <= 1.0)).all()
```

Three bands matter operationally:

| Score | Segment | Action |
|---|---|---|
| Clearly positive | **Persuadables** | Solicit. This is where the appeal budget belongs. |
| Near zero | **Sure things / lost causes** | Skip. They give anyway, or never. |
| Negative | **Sleeping dogs** | Suppress. Asking reduces their giving. |

```python
segments = pd.cut(
    uplift, bins=[-1.01, -0.02, 0.02, 1.01],
    labels=["sleeping dogs", "no effect", "persuadable"],
)
print(pd.Series(segments).value_counts())

# The simulated lift was concentrated in newer donors; recover that.
by_tenure = pd.DataFrame({"uplift": uplift, "new": X["years_active"] < 5})
print(by_tenure.groupby("new")["uplift"].mean().round(3))
assert by_tenure.loc[by_tenure["new"], "uplift"].mean() > \
       by_tenure.loc[~by_tenure["new"], "uplift"].mean()
```

`predict` is a convenience wrapper, `(predict_uplift_score(X) > 0).astype(int)`, marking donors worth soliciting.

```python
solicit = model.predict(X.to_numpy())
print(f"{solicit.sum()} of {n} donors worth soliciting")
np.testing.assert_array_equal(solicit, (uplift > 0).astype(int))
```

## Uplift ranking is not propensity ranking

The point of the exercise: the two orderings disagree, and acting on propensity alone wastes the appeal on people who were going to give regardless.

```python
from philanthropy.models import DonorPropensityModel

propensity = DonorPropensityModel(n_estimators=50, random_state=0).fit(X.to_numpy(), y)
affinity = propensity.predict_affinity_score(X.to_numpy())

top_by_uplift = set(np.argsort(uplift)[-200:])
top_by_affinity = set(np.argsort(affinity)[-200:])
overlap = len(top_by_uplift & top_by_affinity)
print(f"top-200 overlap: {overlap} of 200")

assert overlap < 200, "identical rankings would mean uplift adds nothing"
```

Budget against the uplift ranking; use the propensity score for sizing the ask, not for deciding whom to ask.
