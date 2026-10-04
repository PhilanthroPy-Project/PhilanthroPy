# Who to mail

Is it worth mailing this donor at all?

**Use the model (tested on one organisation so far).** We skipped 9,920 of 28,624 letters and still
raised $1,393 more. On a real donor file, mailing only the donors the model expects to be worth the
postage beats mailing everyone, in dollars raised and in letters saved. Both files below come from
one charity's 1997 mailing.

=== "KDD Cup 1998 (real donor file)"

    We pretended it was 1 June 1997 and asked: for each donor, is a $0.68 mailing worth sending, or
    should we skip it?

    ![Net revenue against how many donors are mailed, most to least likely to respond, KDD Cup 1998](../assets/results/who_to_mail.png#only-light)
    ![Net revenue against how many donors are mailed, most to least likely to respond, KDD Cup 1998](../assets/results/who_to_mail-dark.png#only-dark)

    The chart ranks the 28,624 donors held out for this test from most to least likely to respond,
    and tracks net revenue as you mail further down that list. The model's own stopping point,
    mailing 18,704 of them, is marked: past that point, the next donor's expected gift no longer
    covers the $0.68 mailing cost, so mailing further loses money. That stopping point brings in
    $4,542 after costs, against $3,149 for mailing everyone.

    Beats mailing everyone. Skipping the donors least likely to respond raised more money net, not
    less, even though fewer pieces went out. The curve also answers "what if our mailing costs more
    (or less) than $0.68?": a higher cost per piece shifts the stopping point left, without needing
    a new chart.

=== "cup98VAL (same charity, a file the model never saw)"

    The KDD Cup 1998 tab scores a random 30% slice of the same file the model was fit on. KDD Cup
    1998 also released a second, entirely separate file for exactly this purpose: `cup98VAL`, 96,367
    more donors, with its answer key (`valtargt`) withheld until after the original competition and
    never touched during fitting or the 55/15/30 split. Scored on that file with the same $0.68
    rule: mailing the 62,703 of 96,367 donors the model expected to be worth it brought in $13,764
    after costs, against $10,560 for mailing everyone. Same shape as the first chart, on a file the
    model has never seen in any capacity.

    ![Net revenue against how many donors are mailed, most to least likely to respond, cup98VAL](../assets/results/who_to_mail_cup98val.png#only-light)
    ![Net revenue against how many donors are mailed, most to least likely to respond, cup98VAL](../assets/results/who_to_mail_cup98val-dark.png#only-dark)

## What this means for your file

- Scaled to a file of 10,000 donors like the KDD Cup 1998 one, the model skips about 3,470 letters
  and still brings in about $490 more after costs than mailing everyone.
- Gifts in that file are small (a few dollars to a few hundred), so read the dollar figures as a
  direction, not a forecast for a major-gift program.
- The answer depends on your cost per piece: at a higher cost the model mails fewer donors.

## Try it on your own donors

```python
import numpy as np
from philanthropy.models import AskAmountRecommender, DonorPropensityModel

# Replace with your own file: one row per donor, whether they gave to the last
# mailing, and what they gave; random arrays here only show the API runs.
rng = np.random.default_rng(0)
X_train, X_new = rng.random((400, 4)), rng.random((100, 4))
responded = rng.integers(0, 2, 400)
amount = rng.gamma(2.0, 20.0, 400)

response = DonorPropensityModel(random_state=0).fit(X_train, responded)
gave = responded == 1
ask = AskAmountRecommender(random_state=0).fit(X_train[gave], amount[gave])
expected_gift = response.predict_proba(X_new)[:, 1] * ask.predict(X_new)
mail = expected_gift > 0.68  # your cost per piece
```

??? note "How we tested"

    Real file: [KDD Cup 1998](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html), a 1990s
    direct-mail history, with a 55/15/30 split and the separate `cup98VAL` file as a second check.
    Results on your own file, and at your own mailing cost, will differ.

    Not testable on DonorsChoose, PSID or Karlan and List. None of these files records a
    per-contact mailing cost, so cost-aware selection has no cost side to weigh.

??? note "Numbers for analysts"

    "Mail if expected gift beats the cost" multiplies a response model's score by a suggested gift
    amount, so there is no single ranking to attribute to one feature list. The drivers below are
    for the response half of that decision only; see [What will this donor give next?](ask.md)
    for the amount half's own drivers.

    --8<-- "results/_features/who_to_mail__who_to_mail_kdd98.md"

    --8<-- "results/_features/who_to_mail__who_to_mail_cup98val.md"
