# Score matching-gift eligibility

Corporate matching gifts are the cheapest revenue in fundraising: the donor has already given, and the employer doubles it for the cost of a form. `MatchingGiftFeaturizer` turns an employer name and a gift amount into three model-ready features, so "does this donor's employer match?" becomes a column rather than a manual lookup.

## Build the features

The featurizer takes a **pandas DataFrame**; it raises `TypeError` on an ndarray, because it needs the two columns by name. Employer names are matched case-insensitively after stripping.

```python
import numpy as np
import pandas as pd

from philanthropy.preprocessing import MatchingGiftFeaturizer

donors = pd.DataFrame({
    "employer": ["Boeing", "  microsoft ", "Acme Diner", "", None, "BOEING"],
    "gift_amount": [500.0, 1000.0, 250.0, 100.0, 750.0, "not a number"],
})

# Your organisation's match registry. Keys are normalised for you.
match_ratios = {"Boeing": 1.0, "Microsoft": 2.0, "Delta Air Lines": 0.5}

featurizer = MatchingGiftFeaturizer(match_ratios=match_ratios)
out = featurizer.fit_transform(donors)

print(pd.DataFrame(out, columns=featurizer.get_feature_names_out()))
```

Three columns, always in this order:

| Column | Meaning |
|---|---|
| `has_employer` | `1.0` when the employer cell is a non-empty string, else `0.0`. |
| `match_ratio` | The registry lookup for the normalised employer; `0.0` when unknown. |
| `potential_matched_amount` | `gift_amount` (non-numeric or missing → `0`) × `match_ratio`. |

```python
names = list(featurizer.get_feature_names_out())
frame = pd.DataFrame(out, columns=names)

# "  microsoft " matches "Microsoft"; case and whitespace are normalised.
assert frame.loc[1, "match_ratio"] == 2.0
assert frame.loc[1, "potential_matched_amount"] == 2000.0

# An unknown employer is present but unmatched: has_employer 1, ratio 0.
assert frame.loc[2, "has_employer"] == 1.0
assert frame.loc[2, "match_ratio"] == 0.0

# Empty string and None both read as "no employer on file".
assert frame.loc[3, "has_employer"] == 0.0
assert frame.loc[4, "has_employer"] == 0.0

# A non-numeric gift amount is coerced to 0, not propagated as NaN.
assert frame.loc[5, "match_ratio"] == 1.0
assert frame.loc[5, "potential_matched_amount"] == 0.0
```

`has_employer` is a separate column from `match_ratio` on purpose. "No employer recorded" and "employer recorded but does not match" are different states, and the first is a data-quality signal your CRM team can act on.

## The registry is frozen at fit time

`match_ratios_` is a normalised snapshot taken in `fit`. Mutating the dict you passed in afterwards does not change transform output, the same leakage-safety contract every other transformer in the library follows.

```python
match_ratios["Acme Diner"] = 3.0            # caller edits their own dict
after = featurizer.transform(donors)
np.testing.assert_array_equal(after, out)   # frozen: Acme is still unmatched

refit = MatchingGiftFeaturizer(match_ratios=match_ratios).fit_transform(donors)
assert refit[2, 1] == 3.0                   # a refit picks the new ratio up
```

## Route it in a pipeline

Like the other named-column transformers, give it its own `ColumnTransformer` branch rather than chaining it in series.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from philanthropy.models import DonorPropensityModel

rng = np.random.default_rng(0)
n = 300
prospects = pd.DataFrame({
    "employer": rng.choice(["Boeing", "Microsoft", "Acme Diner", ""], size=n),
    "gift_amount": rng.lognormal(6, 1.0, n),
    "years_active": rng.integers(1, 30, n).astype(float),
})
labels = (
    (prospects["employer"].isin(["Boeing", "Microsoft"]))
    & (prospects["gift_amount"] > 400)
).astype(int).to_numpy()

pipe = Pipeline([
    ("features", ColumnTransformer([
        ("match", MatchingGiftFeaturizer(match_ratios={"Boeing": 1.0, "Microsoft": 2.0}),
         ["employer", "gift_amount"]),
        ("passthrough", "passthrough", ["years_active"]),
    ])),
    ("model", DonorPropensityModel(n_estimators=50, random_state=0)),
])
pipe.fit(prospects, labels)

scores = pipe.predict_proba(prospects)[:, 1]
print(scores[:5].round(3))
assert len(set(scores.round(6))) > 1
```

## Sizing the opportunity

`potential_matched_amount` is a per-donor dollar figure, so summing it gives the unrealised match revenue sitting in your file, usually the number that justifies the campaign.

```python
matched = MatchingGiftFeaturizer(
    match_ratios={"Boeing": 1.0, "Microsoft": 2.0}
).fit_transform(prospects[["employer", "gift_amount"]])

unrealised = matched[:, 2].sum()
eligible = int(matched[:, 1].sum() > 0) and int((matched[:, 1] > 0).sum())
print(f"{eligible} donors at match-eligible employers, ${unrealised:,.0f} unrealised")
```
