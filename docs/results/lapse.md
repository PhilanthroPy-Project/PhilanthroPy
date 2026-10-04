# Lapse

**Use the model.** On PSID household survey data, where most households keep giving, it beats
the simple rule at picking who will lapse. On KDD Cup 1998 and DonorsChoose, where nearly every
donor lapses, a lapse list is about the same as the rule or a little behind it, so use its
retention list there instead: who is most likely to keep giving, which beats the rule on both
files.

Featured on a real donor file: [KDD Cup 1998](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html).
We pretended it was 1 June 1997, the date of that program's own held-out
mailing, and checked who gave nothing to it.

Almost everyone in this file was already about to lapse: about 95 out of
every 100 donors gave nothing to the next mailing, whether you pick with a
model or not. Of our top 10% of picks, 96 out of every 100 lapsed. The best
of the simple rules we compare against here, years since the donor's last
gift, found 97 out of every 100. Picking at random also finds about 95 out
of every 100.

![Model vs. best simple rule, top 1/5/10% of picks, KDD Cup 1998](../assets/results/lapse_kdd98.png#only-light)
![Model vs. best simple rule, top 1/5/10% of picks, KDD Cup 1998](../assets/results/lapse_kdd98-dark.png#only-dark)

**About the same as random picks.** When almost every donor is a lapse risk,
telling them apart barely matters; do not expect a lapse model to sharpen your
list much on a file shaped like this one.

## The useful list here is the opposite one

On a file this lopsided, the question worth asking is not "who is about to lapse" (nearly
everyone), it is "who is the model most confident will *not* lapse". Take the donors the model
ranks least likely to lapse: at the top 5% of that ranking, 8 out of every 100 gave again,
against 6 out of every 100 for the same rule inverted (donors with the fewest years since their
last gift, i.e. the ones the rule itself would call safest). At the top 10% the model is
also ahead (6.5 vs 5.6); at the top 1% the two are too close to call (9 vs 9). Picking at random finds about 5 out of every 100 at any size.

![Retained donors in the model's least-likely-to-lapse 10%, vs. the same rule, KDD Cup 1998](../assets/results/lapse_kdd98_retention.png#only-light)
![Retained donors in the model's least-likely-to-lapse 10%, vs. the same rule, KDD Cup 1998](../assets/results/lapse_kdd98_retention-dark.png#only-dark)

**Slightly ahead of the rule for this retention read**, most clearly at the top 5%; treat the
margin as small. On a file where nearly everyone lapses, use the model this way, as a retention
list, not as a lapse list.

## DonorsChoose

[DonorsChoose Open Data](https://www.icpsr.umich.edu/web/ICPSR/studies/37898) (ICPSR 37898,
doi:10.3886/ICPSR37898.v1) has the same lopsided shape as KDD Cup 1998, only more so: of a random
10% sample of individual ("citizen") donors, about 84 out of every 100 gave nothing in the
following fiscal year (fiscal years 2015-2018 tested in turn, training on every earlier year each
time). Most donors here give exactly once, so "about to lapse" describes nearly everyone, not a
distinguishable group.

Of the model's top 10% of picks, 88 out of every 100 lapsed. The best simple rule found 90 out of
every 100. **Loses here, by a small margin**, same shape as KDD Cup 1998: when the base rate is
this high, there is little room for any ranking to add much.

![Model vs. best simple rule, top 1/5/10% of picks, DonorsChoose](../assets/results/lapse_donorschoose.png#only-light)
![Model vs. best simple rule, top 1/5/10% of picks, DonorsChoose](../assets/results/lapse_donorschoose-dark.png#only-dark)

### The useful list here is the same inverted read

Flip the question the same way as on KDD98: take the donors the model is most confident will
*not* lapse (the bottom decile by lapse score). At the top 10% of that ranking, 50 out of every
100 gave again, against 45 out of every 100 for the same rule inverted. At the top 1%, 83 out of
100 against 74 out of 100 for the rule. Picking at random against the ~16 out of 100 overall
retention rate finds about 16 out of every 100 at any size.

![Retained donors in the model's least-likely-to-lapse 10%, vs. the same rule, DonorsChoose](../assets/results/lapse_donorschoose_retention.png#only-light)
![Retained donors in the model's least-likely-to-lapse 10%, vs. the same rule, DonorsChoose](../assets/results/lapse_donorschoose_retention-dark.png#only-dark)

**Beats the rule here**, in every test year, and furthest ahead at the top of the list. As on
KDD98, use this as a
retention list on a file shaped this way, not as a lapse list. Aggregates only; no donor-level
figures are shown here.

Adding this year's giving trend vs. last year's as its own feature made no real difference to
either read (top 10% lapse hit rate: 88 out of 100 with it, 88 without).

## PSID household survey

The [Panel Study of Income Dynamics](https://psidonline.isr.umich.edu/) (PSID) is a long-running
survey of US households, interviewed every two years (one "wave" every two years). Each wave asks
the household head how much the household gave to all charities over the past year, so these are
self-reported household totals across every charity, not one organization's donor file. Here a
household "lapsed" if it gave something in one wave and reported giving nothing in the next. We
tested the 2015, 2017, 2019 and 2021 waves in turn, training on every earlier wave back to 2001
each time: between 3,109 and 3,803 households per test wave. Unlike the two donor files above,
lapsing is a minority outcome here: about 26 out of every 100 households lapsed.

Of the model's top 10% of picks, 59 out of every 100 lapsed. The best simple rule (the smallest
giving this wave first, the best of the 3 rules checked here in all 4 test waves) found 54 out of
every 100. The model was ahead in every test wave. At the top 1% it found 67 against 56.

![Model vs. best simple rule, top 1/5/10% of picks, PSID](../assets/results/lapse_psid.png#only-light)
![Model vs. best simple rule, top 1/5/10% of picks, PSID](../assets/results/lapse_psid-dark.png#only-dark)

**Beats the rule here.** On a file where lapsing is the exception, not the norm, the model's
list of likely lapsers is clearly better than ranking by this wave's giving.

### The retention read

Flipped the same way as above: about 74 out of every 100 households gave again. Of the 10% the
model ranks least likely to lapse, 95 out of every 100 gave again, against 92 for the best rule
inverted (households with the longest giving runs in three test waves, the biggest givers in
2015). At the top 5% the two are level (94 vs 94), and at the top 1% too close to call (94 vs
95).

![Retained households in the model's least-likely-to-lapse 10%, vs. the same rule, PSID](../assets/results/lapse_psid_retention.png#only-light)
![Retained households in the model's least-likely-to-lapse 10%, vs. the same rule, PSID](../assets/results/lapse_psid_retention-dark.png#only-dark)

**Beats the rule for this retention read at the top 10%, by a small margin**, and level with it
higher up the list. When most households already keep giving, both find a group that almost all
gives again; on a file shaped like this, the lapse list above is the more useful one.

Adding the giving trend between waves as its own feature made no real difference to either read
(top 10% lapse hit rate: 61 out of 100 with it, 59 without).

Only aggregates are shown here: no household IDs, no single-household examples. PSID data are not
redistributed with this project; to rerun these numbers, download your own extract from the PSID
Data Center.

Panel Study of Income Dynamics, public use dataset. Produced and distributed by the Survey
Research Center, Institute for Social Research, University of Michigan, Ann Arbor, MI.

## Which data

Real files: [KDD Cup 1998](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html) and
DonorsChoose Open Data, where lapsing is close to universal (95 and 84 out of 100 respectively),
and the PSID household survey, where it is a minority outcome (about 26 out of 100) and the model
beats the rule. On our sample (synthetic) donor panel, which only checks that the code runs, the
simple rule "years since the donor's last gift" is ahead of the model at every pick size. Results on your own file will differ from all four.

## What the model looks at

--8<-- "results/_features/lapse__lapse_kdd98.md"

--8<-- "results/_features/lapse__lapse_donorschoose.md"

--8<-- "results/_features/lapse__lapse_psid.md"
