# Lapse

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
last gift, i.e. the ones the rule itself would call safest). At the top 1% and top 10% the two
are close enough to call about the same (9 vs 9, and 6 vs 6), though the model's own number is a
shade ahead at every size checked. Picking at random finds about 5 out of every 100 at any size.

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

**About the same as the rule overall, ahead at the top of the list.** As on KDD98, use this as a
retention list on a file shaped this way, not as a lapse list. Aggregates only; no donor-level
figures are shown here.

Adding this year's giving trend vs. last year's as its own feature made no real difference to
either read (top 10% lapse hit rate: 88 out of 100 with it, 88 without).

## Which data

Real files: [KDD Cup 1998](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html) and
DonorsChoose Open Data, where lapsing is close to universal (95 and 84 out of 100 respectively).
On our sample (synthetic) donor panel, where lapsing is a genuine minority outcome, the simple
rule "years since the donor's last gift" beats the model at every pick size we checked; use that
rule instead there. Results on your own file will differ from all three.

## What the model looks at

--8<-- "results/_features/lapse__lapse_kdd98.md"

--8<-- "results/_features/lapse__lapse_donorschoose.md"
