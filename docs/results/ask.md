# What will this donor give next?

**Use the simple rule.** On no real file does the model forecast the next gift reliably better
than a simple rule from the donor's own history: it loses on some files and is about the same on
the rest. The library gives you the usual rule, the larger of the donor's last gift and average
gift, stretched and rounded to a ladder, as one function call (`suggest_ask`, below).

This page tests a forecast: how much a donor will give next time. A suggested ask is a policy
choice built on that forecast (how far to stretch, how to round), and no file here can say which
ask works best, because the ask itself changes the gift.

=== "KDD Cup 1998 (real donor file)"

    Featured on a real donor file: [KDD Cup 1998](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html).
    We pretended it was 1 June 1997 and asked the model to suggest an amount for
    each donor who went on to give.

    Of every 100 suggested amounts, 65 landed within 25% of what the donor
    actually gave. The best of the simple rules we compare against here, the
    higher of what they gave last time or their average gift, landed within 25%
    for 67 out of every 100.

    ![How close is the suggested ask: model vs. best simple rule](../assets/results/ask_kdd98.png#only-light)
    ![How close is the suggested ask: model vs. best simple rule](../assets/results/ask_kdd98-dark.png#only-dark)

    Does not beat the simple rule; use the rule instead. On this file, the
    simple rule is a better guess than the model's suggestion.

=== "DonorsChoose (real donor file)"

    On [DonorsChoose Open Data](https://www.icpsr.umich.edu/web/ICPSR/studies/37898) (ICPSR 37898,
    doi:10.3886/ICPSR37898.v1), we asked the model to suggest a donor's next fiscal-year total giving,
    given they give again (this file's gift dates are month-resolution, so a true next-single-gift
    amount is not well defined, matching the synthetic ask benchmark's own yearly target instead). A
    random 10% sample of citizen donors, fiscal years 2015-2018 tested in turn, training on every
    earlier year each time.

    Of every 100 suggested amounts, 30 landed within 25% of what the donor actually gave the
    following year. The best of the 3 simple rules checked here (last period's total alone, the
    winner in all 4 test years) landed within 25% for 32 out of every 100.

    ![How close is the suggested ask: model vs. best simple rule, DonorsChoose](../assets/results/ask_donorschoose.png#only-light)
    ![How close is the suggested ask: model vs. best simple rule, DonorsChoose](../assets/results/ask_donorschoose-dark.png#only-dark)

    Does not beat the simple rule here either; use the rule instead. Adding this year's giving
    trend vs. last year's as its own feature made no real difference (30 out of 100 within 25% with
    it, versus 30 without).

    Aggregates only; no donor-level figures are shown here.

=== "PSID (household survey)"

    The [Panel Study of Income Dynamics](https://psidonline.isr.umich.edu/) (PSID) is a long-running
    survey of US households, interviewed every two years (one "wave" every two years). Each wave asks
    the household head how much the household gave to all charities over the past year, so these are
    self-reported household totals across every charity, not one organization's donor file. We asked
    the model to suggest a household's total giving in the next wave, given it gives again. We
    tested the 2015, 2017, 2019 and 2021 waves in turn, training on every earlier wave back to 2001
    each time: between 2,360 and 2,881 households per test wave.

    Of every 100 suggested amounts, 29 landed within 25% of what the household actually gave in the
    next wave. The best of the 3 simple rules checked here (last wave's total alone, the winner in
    all 4 test waves) landed within 25% for 28 out of every 100. The model was a little ahead in
    2015 and 2017 and a shade behind in 2019 and 2021. On average it was off by $1,878,
    against $2,166 for the rule, and by less in every test wave.

    ![How close is the suggested ask: model vs. best simple rule, PSID](../assets/results/ask_psid.png#only-light)
    ![How close is the suggested ask: model vs. best simple rule, PSID](../assets/results/ask_psid-dark.png#only-dark)

    About the same as the rule here. It lands within 25% no more often than last wave's total
    does, though it is off by less on average. Adding the giving trend between waves as its own feature
    made no real difference (29 out of 100 within 25% with it, versus 29 without).

    Only aggregates are shown here: no household IDs, no single-household examples. PSID data are not
    redistributed with this project; to rerun these numbers, download your own extract from the PSID
    Data Center.

    Panel Study of Income Dynamics, public use dataset. Produced and distributed by the Survey
    Research Center, Institute for Social Research, University of Michigan, Ann Arbor, MI.

=== "Karlan and List (real donor file)"

    In the [Karlan and List](datasets.md#karlan-and-list-matching-grant-experiment) experiment, about
    50,000 past donors to one US charity got one fundraising letter in 2005. We asked the model to
    predict how much each donor who gave would give, from what the charity knew before mailing. The
    file has no last or average gift, so the rules are the donor's largest past gift and the typical
    gift in the training donors. We tested on a held-out 30% of the donors: 310 who gave.

    Of every 100 predicted amounts, 45 landed within 25% of the actual gift. The best rule, the
    donor's largest past gift, landed within 25% for 41 out of every 100. On average the model was
    off by $17, against $23 for the rule.

    ![How close is the suggested ask: model vs. best simple rule, Karlan and List](../assets/results/ask_karlan_list.png#only-light)
    ![How close is the suggested ask: model vs. best simple rule, Karlan and List](../assets/results/ask_karlan_list-dark.png#only-dark)

    About the same as the rule here. It is not reliably closer within 25%, though it is off by less
    on average. This file has no last gift, which is the rule that is hardest to beat on the other
    files, so read this tab as a weaker test than those.

    Aggregates only; no donor-level figures are shown here.

    Karlan, D. and List, J. A. (2007), "Does Price Matter in Charitable Giving? Evidence from a
    Large-Scale Natural Field Experiment", *American Economic Review* 97(5): 1774-1793. Data from
    openICPSR 113224 (doi:10.3886/E113224V1), CC BY 4.0, copyright American Economic Association 2007.

=== "Sample data (checks the code runs, not that the model works)"

    Our sample (synthetic) donor panel is a code check only and is not counted in the verdict
    (about 23 in 100 within 25%, versus about 24 in 100 for the best of last gift, max(last gift,
    average gift), and the median training gift).

## What this means for your file

- Of 10,000 suggested amounts on a file like KDD Cup 1998, the simple rule lands within 25% of the
  next gift for about 6,740 donors and the model for about 6,500.
- If you can only work the top of your list, the donors the model ranks highest bring in about as
  much as the ones the rule ranks highest (table below).
- Your file is not one of these. If you fit `AskAmountRecommender`, let it check itself against
  the rule (below) and fall back to the rule when it does not win.

Each list is ranked by its own forecast (the rule's list by the rule's own amount). Of every $100
the donors gave next time, the top 10% of each ranking brought in:

| File | Model's top 10% | Rule's top 10% | |
|---|---|---|---|
| KDD Cup 1998 | $22 of every $100 | $22 | About the same |
| DonorsChoose | $64 of every $100 | $64 | About the same |
| PSID household survey | $43 of every $100 | $42 | Beats the rule, in every test wave |
| Karlan and List | $27 of every $100 | $28 | About the same |

This does not change the bottom line above. It does show one place the model adds something: on
PSID its ranking finds the biggest givers slightly better than last wave's total does.

## Try it on your own donors

```python
from philanthropy.models import suggest_ask

suggest_ask(last_gift=[100, 40], avg_gift=[80, 55])  # array([125., 75.])
```

It takes the larger amount, adds 10% and rounds up to the next $25 (`stretch=` and `round_to=`
change both). If you do fit `AskAmountRecommender`, give it the columns for last and average gift
(`last_gift_idx`, `avg_gift_idx`): it then checks itself against the rule on a holdout and sets
`beats_rule_`, so you can fall back to the rule when it is `False`.

??? note "How we tested"

    Real files: [KDD Cup 1998](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html),
    DonorsChoose Open Data, the PSID household survey and the Karlan and List experiment. Each tab
    says what was predicted and on which donors. The model never beats the rule on any file, and is
    only about even with it on PSID and Karlan and List. Results on your own data will differ.

??? note "Numbers for analysts"

    --8<-- "results/_features/ask__ask_kdd98.md"

    --8<-- "results/_features/ask__ask_synthetic.md"

    --8<-- "results/_features/ask__ask_donorschoose.md"

    --8<-- "results/_features/ask__ask_psid.md"

    --8<-- "results/_features/ask__ask_karlan_list.md"
