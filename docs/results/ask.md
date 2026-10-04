# Suggested ask

Featured on a real donor file: [KDD Cup 1998](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html).
We pretended it was 1 June 1997 and asked the model to suggest an amount for
each donor who went on to give.

Of every 100 suggested amounts, 65 landed within 25% of what the donor
actually gave. The best of the simple rules we compare against here, the
higher of what they gave last time or their average gift, landed within 25%
for 67 out of every 100.

![How close is the suggested ask: model vs. best simple rule](../assets/results/ask_kdd98.png#only-light)
![How close is the suggested ask: model vs. best simple rule](../assets/results/ask_kdd98-dark.png#only-dark)

**Does not beat the simple rule; use the rule instead.** On this file, the
simple rule is a better guess than the model's suggestion.

## DonorsChoose

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

**Does not beat the simple rule here either; use the rule instead.** Adding this year's giving
trend vs. last year's as its own feature made no real difference (30 out of 100 within 25% with
it, versus 30 without).

Aggregates only; no donor-level figures are shown here.

## PSID household survey

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
2015 and 2017 and a shade behind in 2019 and 2021. When it missed, it missed by less: off by
$1,878 on average, against $2,166 for the rule, and smaller in every test wave.

![How close is the suggested ask: model vs. best simple rule, PSID](../assets/results/ask_psid.png#only-light)
![How close is the suggested ask: model vs. best simple rule, PSID](../assets/results/ask_psid-dark.png#only-dark)

**About the same as the rule here.** It lands within 25% no more often than last wave's total
does, though its misses are smaller. Adding the giving trend between waves as its own feature
made no real difference (29 out of 100 within 25% with it, versus 29 without).

Only aggregates are shown here: no household IDs, no single-household examples. PSID data are not
redistributed with this project; to rerun these numbers, download your own extract from the PSID
Data Center.

Panel Study of Income Dynamics, public use dataset. Produced and distributed by the Survey
Research Center, Institute for Social Research, University of Michigan, Ann Arbor, MI.

## Which data

Real files: [KDD Cup 1998](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html),
DonorsChoose Open Data and the PSID household survey. Our sample (synthetic) donor panel is a code
check only and is not counted in the verdict (about 23 in 100 within 25%, versus about 24 in 100
for the best of last gift, max(last gift, average gift), and the median training gift). It never
beats the rule on any file, and is only about even with it on PSID; use the simple rule. Results
on your own data will differ.

## What the model looks at

--8<-- "results/_features/ask__ask_kdd98.md"

--8<-- "results/_features/ask__ask_synthetic.md"

--8<-- "results/_features/ask__ask_donorschoose.md"

--8<-- "results/_features/ask__ask_psid.md"
