# Run the check on your own donor score

Most donor scores are checked by pretending it is an earlier date, scoring everyone, and seeing what they did next. The common shortcut is to build the inputs for that check from today's database instead of the database as it stood on the pretend date. Lifetime giving calculated today already contains the gifts the score is supposed to predict, so the score looks better than it is.

On a public donor file (KDD Cup 1998, 95,412 donors), that shortcut moved the ROC-AUC for one test mailing from 0.53 to 0.85 with the same three-input model on the same donors. A coin flip is 0.50. Averaged over all three test mailings it was 0.48 against 0.86; the full write-up is [Real-Data Replication](../explanation/real_data_replication.md). That leak was severe: two running totals saw the whole future. In a real shop the leak is usually partial, so your gap is probably smaller, but the headline number alone cannot tell you how big it is.

This page is the check from the Apra Connections article [*The Check Your Donor Score Has Probably Never Had*](https://connections.aprahome.org/Articles/the-check-your-donor-score-has-probably-never-had). It works on any score, in-house or vendor. You need a CRM query and a spreadsheet, SQL or a few lines of Python; the library is optional.

## The five steps

1. **Pick a cutoff date** at least 12 months ago and write it down. For a vendor score or a saved snapshot, the cutoff must fall after the score was built, otherwise the score has already seen the year you are grading. A score less than a year old cannot be checked this way yet.
2. **Rebuild every input as of the cutoff.** Use only records dated on or before it: gifts by gift date, events by event date, contact reports by contact date. For screening, use the file you had then, or leave the field out. Leave out anyone added to the database after the cutoff. List every undated field as a suspect. This is the step almost everyone skips.
3. **Score the rebuilt table.** For an in-house model, retrain it on history from before the cutoff only. For a vendor score, ask the vendor to run the version they had at the cutoff on the file from step 2. If your CRM keeps old score snapshots, use the one dated before the cutoff.
4. **Pull what happened next.** For the same people, record what they did in the 12 months after the cutoff, measured the way the score claims to predict: gave, gave above a set amount, or moved to a major gift.
5. **Score it again the shortcut way.** Repeat step 3 on the same people with inputs pulled from today's database (for a saved snapshot, use today's score). For each version, sort by score, take the top 10%, and divide their rate of the outcome by the rate for everyone in the file. That ratio is how many times better than random the score is.

If the two ratios are close, nothing leaked, and the as-of ratio is how well the score actually works. If today's version looks much better, swap one input at a time back to its as-of value and rerun. The input that closes the gap is your leak.

## Step 2 in Python: gift roll-ups as of the cutoff

If your gift history is a table of one row per gift, `RFMTransformer` with `as_of` drops every gift dated after the cutoff before it computes recency, frequency and monetary totals. `as_of` is inclusive.

```python
import pandas as pd
from philanthropy.preprocessing import RFMTransformer

gifts = pd.DataFrame({
    'donor_id': [1, 1, 1],
    'gift_date': ['2020-01-01', '2021-01-01', '2025-01-01'],
    'gift_amount': [100.0, 100.0, 50_000.0],
})

rfm = RFMTransformer(reference_date='2022-01-01', as_of='2022-01-01')
print(rfm.fit_transform(gifts))
#    donor_id  recency  frequency  monetary
# 0         1      365          2      200.0
```

Without `as_of`, the same donor rolls up to `frequency=3` and `monetary=50200.0`, and the transformer warns that the table runs past the reference date. To read a CRM export straight into a gift table, see [Get your CRM export in](get_your_crm_export_in.md).

## Step 5 in Python: the two ratios

Put one row per donor in a table with the outcome from step 4 and both versions of the score. The data below is a stand-in; replace it with yours.

```python
import numpy as np
import pandas as pd


def top_decile_lift(score, outcome, top=0.10):
    """Outcome rate among the top `top` share by score, divided by the overall rate."""
    df = pd.DataFrame({"score": score, "outcome": outcome})
    top_rate = df.nlargest(round(len(df) * top), "score")["outcome"].mean()
    return top_rate / df["outcome"].mean()


# Stand-in for your file: one row per donor on file at the cutoff.
rng = np.random.default_rng(0)
gave = rng.random(5_000) < 0.05            # step 4: gave in the 12 months after
signal = rng.normal(size=5_000) + gave     # what was knowable at the cutoff
check = pd.DataFrame({
    "gave": gave,
    "score_as_of": signal,                 # step 3: inputs dated on or before the cutoff
    "score_today": signal + 3 * gave,      # step 5: today's totals already hold that year's gifts
})

for col in ("score_as_of", "score_today"):
    print(f"{col}: {top_decile_lift(check[col], check['gave']):.1f}x random")
# score_as_of: 3.5x random
# score_today: 9.8x random
```

Here the shortcut makes the score look almost three times as good as it is. On your file, the gap between the two lines is the number to report.

## A related check: is a model worth it at all?

The steps above grade a score you already have. A different question is whether a model beats a simple rule (months since last gift, most recent gift first) on your own file. `philanthropy check` answers that one: it builds every feature as of each fiscal year, trains on all labelled years but the last, and compares the model with the best of a few fixed rules on the last year.

```bash
philanthropy check --task lapse --source npsp --data opportunities.csv
```

It does not take your existing score as input, so it is not a substitute for the five steps. See [Use the CLI](use_the_cli.md#check-the-model-against-the-simple-rule-on-your-own-file) for the options and output.

## Tell us what you found

If you run the check, we would like to hear what the gap was, even if it was zero. Open an [adopter issue](https://github.com/PhilanthroPy-Project/PhilanthroPy/issues/new?template=adopter.yml) or a [discussion](https://github.com/PhilanthroPy-Project/PhilanthroPy/discussions). You choose whether your organization is named.
