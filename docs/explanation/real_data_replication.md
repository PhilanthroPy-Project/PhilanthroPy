# Real-Data Replication: KDD Cup 1998

Everything else this project measures is synthetic. This page is not.

The two leakage experiments in [Benchmarks](benchmarks.md) were re-run, unchanged
in structure, on a real donor file: [KDD Cup 1998](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html),
95,412 donors with a 24-mailing direct-mail history, reshaped into a 22-period
donor-period panel. The headline: **building features over the whole export
instead of as of each decision point inflates walk-forward ROC-AUC by +0.376**,
roughly three times the synthetic effect, and it is the largest single number
this repository has measured.

## The two questions

Both experiments hold everything constant except one choice.

**A. Does the CV splitter matter?** Random `StratifiedKFold` against walk-forward
`FiscalYearGroupedSplitter`, each compared to a genuinely held-out final period.

**B. Does feature construction matter?** The same walk-forward CV, run once on
features computed *as of* each period and once on the same aggregates computed
over the whole file including future periods. That second one is the mistake this
library exists to prevent: build features once over the full history, then split.

## Synthetic and real, side by side

Five seeds each, mean ROC-AUC. The synthetic column is
`scripts/leakage_experiment.py`; the real column is
`scripts/real_data_leakage_experiment.py`.

### A. Splitter choice

| Evaluation | Synthetic panel | KDD Cup 1998 |
|---|---:|---:|
| True future, final period genuinely held out | 0.639 | 0.541 |
| Walk-forward `FiscalYearGroupedSplitter` | 0.625 (**-0.014**) | 0.482 (**-0.059**) |
| Random `StratifiedKFold` | 0.608 (**-0.030**) | 0.648 (**+0.107**) |

### B. Feature timing

| Walk-forward CV, features built... | Synthetic panel | KDD Cup 1998 |
|---|---:|---:|
| as of each period | 0.625 | 0.482 |
| over the whole export, including future periods | 0.750 | 0.858 |
| **inflation** | **+0.126** | **+0.376** |

Same model, same splitter, same label in every column. The only thing that
changes between the two rows of table B is *when* the aggregate was computed.

## What the real data changed, including where it disagreed

The script carries a prediction recorded in its own docstring **before it was
run**: real leakage would be *smaller* than the synthetic figures, because real
giving habits persist without the synthetic generator's manufactured drift
working against them, so a classifier should already capture more of that
persistence from as-of history alone. The predicted range was +0.03 to +0.08.

**That prediction was wrong**, by a factor of roughly five, in the direction that
strengthens this library's argument rather than weakening it. It is reported here
rather than quietly re-run because a prediction that only survives when it is
convenient is not a prediction.

Three things came out of the disagreement:

**Leakage is worse on real data, not better.** A real donor's lifetime total
repeats identically across all 22 of their period-rows, which is a stronger and
more identity-revealing signal for a leaky feature to exploit than the synthetic
panel's softer persistence. +0.376 against +0.126.

**Splitter choice matters more here, and in the opposite direction.** Random
`StratifiedKFold` *overstates* the true future by +0.107 AUC, an order of
magnitude past the synthetic 0.014-0.030, and reverses the synthetic run's
finding, where the random split understated it. The familiar claim that a random
split flatters your backtest reproduces on this real file and did not reproduce
on the synthetic one.

**Walk-forward CV undershoots the true future here (0.482 against 0.541), where
in the synthetic run it slightly overshot.** Real promotion response rates swing
sharply by campaign type, 8% to 22% across the historical mailings, rather than
drifting smoothly the way the synthetic generator's drift term does. The three
most-recent periods that walk-forward evaluates on are therefore not uniformly
easier or harder than the single held-out final period. That is a property of
this donor file, not a defect in the splitter.

Put together: correct feature timing is worth about six times what correct
splitter choice is worth on this file (+0.376 against +0.107), and a correct
splitter does not recover a single point of the feature-timing loss.

## How the panel was built

KDD Cup 1998 ships one row per donor with 24 direct-mail promotions,
`ADATE_2`..`ADATE_24` (mail date), and, for promotions 3-24, `RAMNT_3`..`RAMNT_24`
(amount given, absent if no gift). Promotion 2 is the held-out 97NK mailing the
original competition scores (`TARGET_B` / `TARGET_D`), which is why it has no
`RAMNT_2`.

Column index does not track calendar order for an individual donor's history,
because donors are not mailed on identical schedules. It tracks the *campaign's*
mail date exactly, because every recipient of a given campaign was mailed on the
same date: `ADATE_2` is 9706 (June 1997) on every row. So promotions 3-24, oldest
to newest, are exactly the reverse of their column index, for every donor, with
no per-row date parsing.

That gives 22 chronological periods per donor. Period *p*'s label is "did this
donor give at period *p+1*", mirroring "gave in the following year" in the
synthetic panel. For the last historical period, period *p+1* **is** the 97NK
mailing, so its label is `TARGET_B` itself rather than something derived.

Two caveats worth stating plainly. A donor not mailed a given campaign
contributes no gift that period, which is indistinguishable in this file from
being mailed and not responding; both record as zero. And real data has no seed
to regenerate, so the five seeds vary only the classifier's and the splitters'
own randomness, not the panel.

## Reproducing it

```bash
python scripts/real_data_leakage_experiment.py
```

The script calls
[`fetch_kdd98_donors`](../reference/datasets.md), which downloads `cup98lrn.zip`
(~36 MB) from the UCI mirror on first use and caches it under `~/philanthropy_data`
(override with the `PHILANTHROPY_DATA` environment variable). The download is
SHA-256 checked against the file served at that URL.

**This is the one function in the package that touches the network, and it only
does so when you call it.** Everything else is offline by construction and
`tests/test_no_network.py` enforces that with socket poisoning plus an import
scan. Nothing about your own donors, gifts, or environment is ever transmitted.
If your environment has no outbound access, pass `download_if_missing=False` and
place the archive in the cache directory yourself.

Expect the full run to take several minutes: five seeds times four evaluations
over a 95,412 x 22 panel, with `RandomForestClassifier(n_estimators=200)`.

The script's output and an environment lock are archived on Zenodo at
[10.5281/zenodo.22050649](https://doi.org/10.5281/zenodo.22050649), so the
numbers on this page can be checked without re-downloading anything.

## What this does and does not establish

It establishes that the leakage mechanism this library is designed around is
real, is larger on a real donor file than on a synthetic one, and is not fixed by
choosing a better splitter.

It does not establish an accuracy number for your program. One real file is one
real file, and this one is a 1997 direct-mail acquisition list, not a
major-gift pipeline. No estimator's *accuracy* benchmark in this project is
validated on real data; the *leakage mechanism* now is.

For how to run the equivalent check on your own giving history, see
[Validating on your own data](benchmarks.md#validating-on-your-own-data).
