# Model Validation & Benchmarks

Every model here ships with a reproducible benchmark number. This page shows you
those numbers, and why you should **not** trust them for your own program until
you re-validate on your own data.

!!! danger "These numbers are synthetic, and inflated by a flaw in the generator"
    The table below is measured on `generate_synthetic_donor_data`, a reproducible
    but **artificial** donor pool. It says nothing about how a model will perform
    on your CRM, and it is optimistic for a specific, measurable reason given
    below: the generator draws the gift amount *from* the label, so the strongest
    feature is a noisy readout of the answer key. **Re-run the evaluation on your
    own labelled giving history before trusting any score in production.**

    There is no validation on real donor data anywhere in this repository. The one
    real dataset that ships, `load_ciob_fundraising`, has no donor rows, no gift
    amounts and no labels, and no estimator is ever fitted on it. Every number on
    this page, in the README, and in the paper is synthetic.

## Why these numbers are inflated

The synthetic generator runs the domain's causal arrow backwards. It draws the
label from a logistic model of `years_active` and `event_attendance_count`, and
*then* draws `total_gift_amount` conditional on that label
(`mu=9.5` if major, `7.5` otherwise). So the feature that carries most of the
signal is generated from the answer.

That is measurable. Integrating the generator's own noise term out of its
logistic model gives the Bayes-optimal error over the two **causal** features:

| Evaluation | ROC-AUC | Accuracy |
|---|---:|---:|
| Bayes-optimal, causal features only | n/a | **0.768** |
| `DonorPropensityModel`, causal features only | 0.748 | 0.720 |
| `DonorPropensityModel`, **including** `total_gift_amount` | **0.935** | **0.880** |

Held-out, 4,000 rows, 75/25 stratified split, base rate 0.687.

The middle row is the honest ceiling and the model sits just under it, which is
the correct result. The bottom row *beats the Bayes rate of the causal process*
by roughly 19 AUC points. A model cannot legitimately do that; it is the
signature of a feature derived from the target.

In a real shop the causation runs the other way, and using cumulative lifetime
giving to predict "is a major donor" is the classic fundraising leakage this
library otherwise exists to prevent. So read the table below as an
implementation smoke test, not as evidence that the methods work.

The generator is not fixed here, because changing the data-generating process
moves every published number and is a design decision about what the reference
dataset should represent. It is tracked separately.

!!! note "A correction"
    A previous version of this page distrusted its own numbers for the wrong
    reason. It said the synthetic data was "cleanly separable by construction".
    It is not: the label is a Bernoulli draw with a real noise term, and the
    irreducible error over the causal features is 23.2%, not zero. The problem
    is the leaked feature, not clean separability.

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

Synthetic pool: 4,000 rows, positive rate 0.678; test split 1,000 rows.
**Each cell is the mean across five seeds, with the min–max range in
parentheses.**

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| `PropensityScorer` (baseline) | 0.000 (0.000–0.000) | 0.000 (0.000–0.000) | 0.000 (0.000–0.000) | 0.500 (0.500–0.500) |
| `DonorPropensityModel` | 0.891 (0.882–0.904) | 0.936 (0.922–0.943) | 0.913 (0.902–0.923) | 0.922 (0.906–0.936) |
| `MajorGiftClassifier` | 0.873 (0.866–0.896) | 0.967 (0.962–0.971) | 0.918 (0.912–0.931) | 0.927 (0.917–0.936) |
| `LapsePredictor` | 0.891 (0.882–0.904) | 0.936 (0.922–0.943) | 0.913 (0.902–0.923) | 0.922 (0.906–0.936) |
| `PlannedGivingIntentScorer` | 0.888 (0.882–0.904) | 0.955 (0.951–0.958) | 0.920 (0.916–0.930) | 0.935 (0.922–0.945) |

*(Measured with scikit-learn 1.7.2 on the synthetic dataset; your numbers will
differ.)*

The ranges are narrow (roughly ±0.01 on F1 and ±0.015 on ROC-AUC), so the
models are genuinely close to each other on this task. Any comparison between
them that turns on the third decimal is reading noise.

## How to read this

- **`PropensityScorer` is the floor.** It is a constant-probability baseline
  (P=0.5), so its ROC-AUC of 0.500 means exactly "no better than chance." Every
  real model has to beat it. Here they all do, by a wide margin, but the
  synthetic data's separability inflates that margin.
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
  recall depend on the 0.5 decision threshold and this pool's 0.677 positive
  rate, far higher than a real major-donor base rate, which is typically a few
  percent. Expect precision to fall sharply on realistically imbalanced data.

## Validating on your own data

1. Assemble a labelled historical dataset (features + a binary outcome you can
   observe, e.g. "made a major gift in the following year").
2. Split **temporally**, not randomly: train on earlier years, test on later
   ones, using `FiscalYearGroupedSplitter` to avoid leakage across fiscal
   boundaries.
3. Report ROC-AUC plus precision/recall **at the threshold you will actually
   act on**, and calibrate that threshold to your team's capacity.
4. Re-check periodically: donor behaviour and your data pipeline both drift.
