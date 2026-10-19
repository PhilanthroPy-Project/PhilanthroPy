# How sure are we?

How often does the range around a suggested amount hold the gift the donor actually gives?

**Use the model.** `GiftIntervalCalibrator` puts a range around each suggested amount, and you choose
how often it should hold the real gift: 80, 90 or 95 times in 100. On all four real files it held the
gift within 3 points of the level asked for. The ranges are honest, but they are wide: they tell you
how much a single donor's next gift can swing, not where it will land.

=== "KDD Cup 1998 (real donor file)"

    We took the donors who gave to the 1997 mailing, built the suggested amount on 55% of them, set the
    ranges on another 15%, and checked them on the held-out 30%: 1,453 donors who gave.

    Asked to hold 90 out of 100 gifts, the range held 90 out of 100. The typical 90% range was
    19 dollars wide.

    ![Range asked for vs. how often it held the actual gift, KDD Cup 1998](../assets/results/interval_kdd98.png#only-light)
    ![Range asked for vs. how often it held the actual gift, KDD Cup 1998](../assets/results/interval_kdd98-dark.png#only-dark)

=== "DonorsChoose (real donor file)"

    We pretended it was the start of each of the last 4 fiscal years in turn: the suggested amount
    learned from every year before the one just ended, the range was set on the year just ended, and we
    checked it on the coming year's donors who gave again.

    Asked to hold 90 out of 100 gifts, the range held 89 out of 100 on average, and 89 out of
    100 in the weakest year. The typical 90% range was 323 dollars wide.

    ![Range asked for vs. how often it held the actual gift, DonorsChoose](../assets/results/interval_donorschoose.png#only-light)
    ![Range asked for vs. how often it held the actual gift, DonorsChoose](../assets/results/interval_donorschoose-dark.png#only-dark)

=== "PSID (household survey)"

    The same check over the last 4 survey waves, one at a time, on households that gave again.

    Asked to hold 90 out of 100 gifts, the range held 89 out of 100 on average, and 87 out of
    100 in the weakest wave. The typical 90% range was 4,830 dollars wide: household giving in this
    survey swings by thousands of dollars from one wave to the next.

    ![Range asked for vs. how often it held the actual gift, PSID](../assets/results/interval_psid.png#only-light)
    ![Range asked for vs. how often it held the actual gift, PSID](../assets/results/interval_psid-dark.png#only-dark)

    Only aggregates are shown here: no household IDs, no single-household examples.

=== "Karlan and List (real donor file)"

    Donors who gave to the 2005 letter, with the same 55/15/30 split as the KDD Cup 1998 tab: 310 donors
    who gave in the held-out 30%.

    Asked to hold 90 out of 100 gifts, the range held 89 out of 100. The typical 90% range was 69
    dollars wide.

    ![Range asked for vs. how often it held the actual gift, Karlan and List](../assets/results/interval_karlan_list.png#only-light)
    ![Range asked for vs. how often it held the actual gift, Karlan and List](../assets/results/interval_karlan_list-dark.png#only-dark)

    Data CC BY 4.0, copyright American Economic Association 2007 (openICPSR 113224).

=== "Sample data (checks the code runs, not that the model works)"

    On our generated donors, asked to hold 90 out of 100 gifts, the range held 91 out of 100 across
    5 draws.

    ![Range asked for vs. how often it held the actual gift, sample data](../assets/results/interval_synthetic.png#only-light)
    ![Range asked for vs. how often it held the actual gift, sample data](../assets/results/interval_synthetic-dark.png#only-dark)

## What this means for your file

- Give 10,000 donors a 90% range and, on these files, between about 8,880 and 8,960 of their next gifts
  land inside it.
- A range that holds the gift that often has to be wide. Use it to set expectations (a gift well
  below the range is unusual; one at the bottom of it is not), not as the ask itself.
- Your file is not one of these. Set the ranges on your own recent donors, never on the donors you
  built the suggested amount from, and check how often they held before you rely on them.

## Try it on your own donors

```python
import numpy as np
from philanthropy.models import AskAmountRecommender, GiftIntervalCalibrator

# Replace with your own file: older donors to fit on, recent donors the model
# never saw to set the ranges on; random arrays here only show the API runs.
rng = np.random.default_rng(0)
X_fit, X_cal, X_new = rng.random((300, 4)), rng.random((100, 4)), rng.random((20, 4))
y_fit, y_cal = rng.gamma(2.0, 50.0, 300), rng.gamma(2.0, 50.0, 100)

ask = AskAmountRecommender(random_state=0).fit(X_fit, y_fit)
ranges = GiftIntervalCalibrator(ask, alpha=0.10).fit(X_cal, y_cal)  # 90% ranges
interval = ranges.predict_gift_interval(X_new)
interval.lower, interval.upper
```

??? note "How we tested"

    The suggested amount is the same `AskAmountRecommender` as on the
    [What will this donor give next?](ask.md) page, with default settings. `GiftIntervalCalibrator`
    with its default score (the absolute miss, so every donor gets the same width) was set on donors
    the model never trained on and checked on later ones it never saw. A level counts as on target when
    the range held the gift no more than 3 points less often than asked. On the two walk-forward files
    the suggested amount here learns from one fewer year than on the ask page, because the year just
    ended is used to set the ranges.

??? note "Numbers for analysts"

    | File | Asked for | Held the gift | Median width |
    |---|---|---|---|
    | KDD Cup 1998 | 80% | 79 of 100 | $11 |
    | KDD Cup 1998 | 90% | 90 of 100 | $19 |
    | KDD Cup 1998 | 95% | 94 of 100 | $24 |
    | DonorsChoose | 80% | 79 of 100 | $172 |
    | DonorsChoose | 90% | 89 of 100 | $323 |
    | DonorsChoose | 95% | 95 of 100 | $599 |
    | PSID household survey | 80% | 78 of 100 | $3,159 |
    | PSID household survey | 90% | 89 of 100 | $4,830 |
    | PSID household survey | 95% | 94 of 100 | $7,157 |
    | Karlan and List | 80% | 81 of 100 | $49 |
    | Karlan and List | 90% | 89 of 100 | $69 |
    | Karlan and List | 95% | 95 of 100 | $83 |
