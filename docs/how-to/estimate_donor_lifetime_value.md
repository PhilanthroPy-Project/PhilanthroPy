# Estimate donor lifetime value from a gift export

Most of PhilanthroPy's models need a label: who upgraded, who lapsed, who made a major gift. Two models need nothing but the gift history every CRM can export (donor, date, amount):

* `BetaGeoModel` (the BG/NBD "buy till you die" model) estimates how likely each donor is to still be giving, and how many gifts they will make over the next year or any other horizon.
* `GammaGammaModel` estimates each donor's typical future gift, pulling donors with few gifts toward the file's average.

Multiply the two and you have each donor's expected giving over the horizon, per donor rather than one file-wide average.

!!! note "Tier 2: Beta"
    Both models reproduce the published CDNOW parameters, but they have not yet been checked against a real gift file's holdout year. Run the holdout check below on your own data before you rely on them.

## Shape the export into a gift table

You need one row per gift with three columns: `donor_id`, `gift_date` and `gift_amount`. A CRM export has its own column names, so rename them. Filter out what does not count as a gift first (failed or refunded payments, pledges, soft credits): the model treats every remaining row as money received.

The example simulates a CiviCRM-style contribution export for 2,000 donors, 2019 to 2024.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
start = pd.Timestamp("2019-01-01")

rows = []
for donor in range(2000):
    rate = rng.gamma(1.5, 1.0)        # gifts per year while still active
    p_drop = rng.beta(1.0, 4.0)       # chance of dropping out after each gift
    mean_gift = rng.gamma(3.0, 40.0)  # this donor's typical gift
    t = rng.uniform(0, 2)             # first gift in 2019 or 2020
    while t < 6:                      # through the end of 2024
        rows.append({
            "Contact ID": f"D{donor:04d}",
            "Contribution Date": start + pd.Timedelta(days=365.25 * t),
            "Total Amount": round(rng.gamma(4.0, mean_gift / 4.0), 2),
        })
        if rng.random() < p_drop:
            break
        t += rng.exponential(1 / rate)
export = pd.DataFrame(rows)

gifts = export.rename(columns={
    "Contact ID": "donor_id",
    "Contribution Date": "gift_date",
    "Total Amount": "gift_amount",
})
```

## Summarise each donor

`repeat_giving_summary` turns the gift table into the four numbers both models read. Hold out the last year so you can check the forecast against what actually happened: `as_of="2023-12-31"` ignores every gift after that date.

```python
from philanthropy.models import repeat_giving_summary

summary = repeat_giving_summary(gifts, as_of="2023-12-31")
print(summary.head())
```

| Column | Meaning |
|---|---|
| `frequency` | Repeat gifts: distinct gift days minus one. Two gifts on the same day count once. |
| `recency` | Years from the donor's first gift to their last. |
| `T` | Years from the first gift to `as_of`. |
| `monetary_value` | Mean of the repeat gifts. The first gift is left out, as the model expects; 0 for a one-time donor. |

Times are in years by default, so a `horizon` of 1 later means "the next year". Pass `time_unit_days=7` to work in weeks.

## Who is still giving, and how often

```python
from philanthropy.models import BetaGeoModel

rfm = summary[["frequency", "recency", "T"]]
bg = BetaGeoModel().fit(rfm)

summary["p_active"] = bg.predict_alive_score(rfm)
summary["gifts_next_year"] = bg.predict_gifts_forecast(rfm, horizon=1.0)
print(summary[["frequency", "recency", "T", "p_active", "gifts_next_year"]].head())

assert summary["p_active"].between(0, 1).all()
```

Read `p_active` with care:

* A donor who gave often and then went quiet scores low, because the model expected another gift by now.
* A donor whose only gift so far was the first one always scores 1. In this model a donor can only drop out right after a repeat gift, so it has no evidence either way. Use `gifts_next_year`, which stays low for them, to rank.

## How much each gift will be

`GammaGammaModel` assumes a donor's gift size does not depend on how often they give. Check that on your repeat donors before you use it. A correlation near zero is fine; a strong one means the product below double counts.

```python
from philanthropy.models import GammaGammaModel

repeat = summary["frequency"] > 0
corr = summary.loc[repeat, ["frequency", "monetary_value"]].corr().iloc[0, 1]
print(f"frequency vs gift size correlation: {corr:.2f}")
assert abs(corr) < 0.3

size_inputs = summary[["frequency", "monetary_value"]]
gg = GammaGammaModel().fit(size_inputs)
summary["gift_size"] = gg.predict_gift_size_forecast(size_inputs)
```

A donor with many repeat gifts is predicted close to their own average gift. A donor with one or two is pulled toward the file's average, and a one-time donor gets the file's average.

## Expected giving next year

```python
summary["value_next_year"] = summary["gifts_next_year"] * summary["gift_size"]
print(summary.sort_values("value_next_year", ascending=False)
      [["frequency", "p_active", "gifts_next_year", "gift_size", "value_next_year"]]
      .head().round(2))
```

For a multi-year value, forecast over a longer `horizon` and discount it the way your finance office does. `philanthropy.metrics.donor_lifetime_value` does the discounting for a single average donor.

## Check the forecast on the year you held out

Count each donor's actual gift days in 2024 and compare them with the forecast made from data up to the end of 2023.

```python
actual = (
    gifts[(gifts["gift_date"] > "2023-12-31") & (gifts["gift_date"] <= "2024-12-31")]
    .assign(day=lambda d: d["gift_date"].dt.normalize())
    .groupby("donor_id")["day"].nunique()
    .reindex(summary.index, fill_value=0)
)

predicted_total = summary["gifts_next_year"].sum()
print(f"gifts in 2024  predicted: {predicted_total:.0f}  actual: {actual.sum()}")
assert abs(predicted_total - actual.sum()) / actual.sum() < 0.15

check = pd.DataFrame({
    "predicted": summary["gifts_next_year"],
    "actual": actual,
    "repeat_gifts_so_far": summary["frequency"].clip(upper=6),
})
print(check.groupby("repeat_gifts_so_far").mean().round(2))
```

Two things to look at:

* **The total.** It should be close. A large miss usually means the export still contains rows that are not gifts, or the giving pattern changed (a new monthly giving programme, a campaign year).
* **The table by past repeat gifts.** Predicted and actual should rise together. If the model is right on average but wrong for one group, plan around that group.

On real files, expect the totals to be less tight than on this simulated one. The model knows nothing about appeals, campaigns or seasons, only each donor's past rhythm.
