# Model Validation & Benchmarks

Every model here ships with a reproducible benchmark number. This page shows you
those numbers, and why you should **not** trust them for your own program until
you re-validate on your own data.

!!! warning "The model-benchmark numbers below are synthetic"
    The table below is measured on `generate_synthetic_donor_data`, a
    reproducible but **artificial** donor pool. It says nothing about how a model
    will perform on your CRM. **Re-run the evaluation on your own labelled giving
    history before trusting any score in production.**

    The leakage experiment further down this page is different: it is also run
    on a real donor file, [KDD Cup 1998](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html),
    via `scripts/real_data_leakage_experiment.py`. No estimator's *accuracy*
    numbers above are validated on real data; the *leakage mechanism* now is.
    The one other real dataset that ships, `load_ciob_fundraising`, has no
    donor rows, no gift amounts and no labels, and no estimator is fitted on it.

## What was wrong with these numbers before 0.7.0, and what fixed it

The generator used to run the domain's causal arrow backwards. It drew the label
from a logistic model of `years_active` and `event_attendance_count`, and *then*
drew `total_gift_amount` conditional on that label. The feature carrying most of
the signal was generated from the answer.

That was measurable, and it is the reason these numbers moved:

| Evaluation | ROC-AUC | Accuracy |
|---|---:|---:|
| Bayes-optimal over the causal features, old generator | | 0.768 |
| `DonorPropensityModel` including `total_gift_amount`, old generator | **0.935** | **0.880** |
| Bayes-optimal given latent capacity, new generator | | 0.806 |
| `DonorPropensityModel` including `total_gift_amount`, new generator | 0.814 | 0.759 |

The second row **beat the Bayes rate of the generator's own process** by roughly
19 AUC points. No model can legitimately do that; it is the signature of a
feature derived from the target. In a real shop the causation runs the other way,
and using cumulative lifetime giving to predict "is a major donor" is the classic
fundraising leakage this library exists to prevent, so the reference dataset was
teaching the anti-pattern.

A latent **giving capacity** now drives everything: it is a confounder causing
both the giving history and the label. `total_gift_amount` is a noisy realisation
of capacity, and the label is a soft $25,000 threshold on it. `last_gift_date`
follows engagement rather than the label, which was a second target-derived
feature. The fourth row sits **below** the ceiling in the third, which is the
correct relationship and the whole point.

The numbers got worse and more trustworthy. The base rate also fell from 0.687 to
0.378, which is still far above a real major-donor rate of a few percent.

## Reproducing this table

Run the benchmark yourself. It lives in the repo as a committed, dependency-free
script:

```bash
python scripts/benchmark_models.py
```

For each of **five seeds** (42–46) the script builds a 4,000-row synthetic pool,
takes a stratified 75/25 train/test split, and fits every applicable binary
classifier in `philanthropy.models` on the documented feature set
(`total_gift_amount`, `years_active`, `event_attendance_count`) against the
`is_major_donor` label. It prints precision / recall / F1 / ROC-AUC on the
held-out test set using `sklearn.metrics`.

Five seeds rather than one on purpose. A single three-decimal score reads as a
claim about the method when it is mostly a claim about the split; the spread
below is the honest resolution of these numbers.

## Results

Synthetic pool: 4,000 rows, positive rate 0.378; test split 1,000 rows.
**Each cell is the mean across five seeds, with the min–max range in
parentheses.**

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| `PropensityScorer` (baseline) | 0.000 (0.000–0.000) | 0.000 (0.000–0.000) | 0.000 (0.000–0.000) | 0.500 (0.500–0.500) |
| `DonorPropensityModel` | 0.668 (0.625–0.699) | 0.609 (0.592–0.632) | 0.636 (0.625–0.645) | 0.810 (0.802–0.815) |
| `MajorGiftClassifier` | 0.731 (0.687–0.765) | 0.559 (0.536–0.585) | 0.633 (0.606–0.656) | 0.827 (0.817–0.832) |
| `LapsePredictor` | 0.668 (0.625–0.699) | 0.609 (0.592–0.632) | 0.636 (0.625–0.645) | 0.810 (0.802–0.815) |
| `PlannedGivingIntentScorer` | 0.731 (0.717–0.740) | 0.605 (0.576–0.636) | 0.662 (0.643–0.683) | 0.840 (0.833–0.844) |

*(Measured with scikit-learn 1.7.2 on the synthetic dataset; your numbers will
differ.)*

The ranges are narrow, roughly ±0.01 on F1 and ±0.01 on ROC-AUC, so the models
are genuinely close to each other on this task. Any comparison between them that
turns on the third decimal is reading noise.

## How to read this

- **`PropensityScorer` is the floor.** It is a constant-probability baseline
  (P=0.5), so its ROC-AUC of 0.500 means exactly "no better than chance." Every
  real model has to beat it, and here they all do. The margin is now bounded by
  something real: the Bayes ceiling of 0.806 accuracy given latent capacity. A
  model reporting much above that on this data would indicate a bug, not skill.
- **The baseline's precision/recall are 0.000 by construction, not by failure.**
  Since 0.6.0 its threshold comparison is strict (`proba > threshold`), so a
  constant 0.5 score falls below the default 0.5 threshold and it predicts the
  negative class for everyone. scikit-learn requires
  `argmax(predict_proba) == predict`, and `argmax` of a tied `[0.5, 0.5]` row is
  index 0. Either choice is arbitrary for a constant scorer; only the ROC-AUC of
  0.500 carries information.
- **`LapsePredictor` and `DonorPropensityModel` report identical numbers** on
  this task. Both wrap a default `RandomForestClassifier` with the same
  `random_state` and features, so the match is expected, not a bug.
  `LapsePredictor` is purpose-built for a lapse label, not `is_major_donor`; it
  appears here only because its estimator is applicable.
- **ROC-AUC is the most transferable metric** across base rates. Precision and
  recall depend on the 0.5 decision threshold and this pool's 0.378 positive
  rate, which is still far higher than a real major-donor base rate of a few
  percent. Expect precision to fall sharply on realistically imbalanced data.

## Where the leakage actually is

The library's central claim is that it is leakage-safe by construction. That
claim was never quantified, which was the largest hole in the argument. It is now
measured by a committed script:

```bash
python scripts/leakage_experiment.py
```

A seeded donor-year panel, 3,000 donors over 6 panel years, with a stable
per-donor propensity (real donors have habits, and that persistence is what any
leakage must exploit) and a sector-wide drift that makes later years genuinely
harder. Label: did the donor give in the following year. Five seeds, mean with
min-max.

**Does the choice of CV split matter?** Less than the folklore says.

| Evaluation | ROC-AUC | Error vs the true future |
|---|---:|---:|
| True future, final year genuinely held out | 0.639 (0.621-0.653) | |
| Walk-forward `FiscalYearGroupedSplitter` | 0.625 (0.620-0.636) | **-0.014** |
| Random `StratifiedKFold` | 0.608 (0.601-0.616) | -0.030 |

Both CV runs exclude the final panel year, which is the year the target column
scores. That exclusion matters: "train on everything before the final year, score
the final year" is exactly what a walk-forward splitter's last fold does, so
leaving the year in would put the estimand inside one estimator and not the
other, and walk-forward would win by construction. An earlier version of this
script did that and reported walk-forward as three times more accurate.

Walk-forward CV estimates the future about twice as accurately, which is a real
result and an argument for the splitter. But note the direction: the
random split **understated** the future here, it did not flatter it. The common
claim that a random split inflates your backtest did not reproduce, in this or in
two other configurations tried before this one, including a static per-donor
label. Do not repeat that claim on the strength of this repository.

**Does the choice of feature construction matter?** Enormously.

| Walk-forward CV, features built... | ROC-AUC |
|---|---:|
| as of each panel year | 0.625 (0.620-0.636) |
| over the whole export, including future years | 0.750 (0.745-0.757) |
| | **+0.126 AUC of pure inflation** |

Same model, same splitter, same label. The only difference is whether the
aggregate features were computed as of the decision point or once over the full
history. Building features first and splitting afterwards inflates the score by
0.126 AUC and no choice of splitter recovers it.

That is the failure mode this library is built around, and it is why the fitted
statistics are frozen in `fit` (see
[Design principles](design_principles.md)) and why `EncounterTransformer` and
`GratefulPatientFeaturizer` take an `as_of` cutoff. A correct splitter is
worth about 0.016 AUC of accuracy in your estimate; correct feature timing is
worth 0.126, roughly eight times more.

These are synthetic numbers on a generator whose persistence and drift I chose.
They establish the mechanism and its rough magnitude, not a value to quote for
your program.

## Real-data replication: KDD Cup 1998

The two experiments above were re-run, unchanged in structure, on a real donor
file: [KDD Cup 1998](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html),
95,412 donors with a 24-mailing direct-mail history, reshaped into a
22-period donor-period panel (`philanthropy.datasets.fetch_kdd98_donors`).
Label: did the donor give at the following mailing; the final period's label
is the dataset's own held-out target. Five seeds, mean with min-max.

```bash
python scripts/real_data_leakage_experiment.py
```

**Prediction, recorded in the script before it was run:** real leakage would
be *smaller* than the synthetic figures below, because real giving habits
persist without the synthetic generator's manufactured drift working against
them. **That prediction was wrong**, in the direction that strengthens this
library's argument rather than weakening it:

| Evaluation | ROC-AUC | Error vs the true future |
|---|---:|---:|
| True future, final period genuinely held out | 0.541 | |
| Walk-forward `FiscalYearGroupedSplitter` | 0.482 | **-0.059** |
| Random `StratifiedKFold` | 0.648 | **+0.107** |

| Walk-forward CV, features built... | ROC-AUC |
|---|---:|
| as of each period | 0.482 |
| over the whole file, including future periods | 0.858 |
| | **+0.376 AUC of pure inflation** |

Whole-history feature construction inflates real-data ROC-AUC by **+0.376
AUC**, roughly three times the synthetic **+0.126**. A real donor's lifetime
total repeats identically across all 22 of their period-rows, which is a
stronger, more identity-revealing signal for a leaky feature to exploit than
the synthetic panel's softer persistence. Splitter choice also matters more
here than in the synthetic run: random `StratifiedKFold` overstates the true
future by **+0.107 AUC**, an order of magnitude past the synthetic 0.014-0.030,
reversing the direction the synthetic run found (there, random split *understated*
the future).

One more honest discrepancy, reported rather than smoothed: walk-forward CV
(0.482) undershoots the true-future baseline (0.541) here, where in the
synthetic run it slightly overshot. Real promotion response rates swing
sharply by campaign type (8%-22% across the historical mailings) rather than
drifting smoothly the way the synthetic generator's drift term does, so the
three most-recent periods walk-forward evaluates on are not uniformly easier
or harder than the single held-out final period. That is a property of this
donor file, not a bug in the splitter.

These numbers, unlike the ones above, are on real donor data. They still
establish a mechanism and its magnitude on one real file, not a value to
quote for your program.

The script's output and environment lock are archived separately on Zenodo:
[10.5281/zenodo.22050649](https://doi.org/10.5281/zenodo.22050649).

## Validating on your own data

1. Assemble a labelled historical dataset (features + a binary outcome you can
   observe, e.g. "made a major gift in the following year").
2. Split **temporally**, not randomly (train on earlier years, test on later
   ones) using `FiscalYearGroupedSplitter` to avoid leakage across fiscal
   boundaries.
3. Report ROC-AUC plus precision/recall **at the threshold you will actually
   act on**, and calibrate that threshold to your team's capacity.
4. Re-check periodically: donor behaviour and your data pipeline both drift.
