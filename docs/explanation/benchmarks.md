# Model Validation & Benchmarks

Every model here ships with a reproducible benchmark number. This page shows you
those numbers, and why you should **not** trust them for your own program until
you re-validate on your own data.

!!! warning "These numbers are synthetic"
    The table below is measured on `generate_synthetic_donor_data`, a
    reproducible but **artificial** donor pool. It says nothing about how a model
    will perform on your CRM. **Re-run the evaluation on your own labelled giving
    history before trusting any score in production.**

    There is no validation on real donor data anywhere in this repository. The
    one real dataset that ships, `load_ciob_fundraising`, has no donor rows, no
    gift amounts and no labels, and no estimator is ever fitted on it. Every
    number on this page, in the README, and in the paper is synthetic.

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

## Validating on your own data

1. Assemble a labelled historical dataset (features + a binary outcome you can
   observe, e.g. "made a major gift in the following year").
2. Split **temporally**, not randomly (train on earlier years, test on later
   ones) using `FiscalYearGroupedSplitter` to avoid leakage across fiscal
   boundaries.
3. Report ROC-AUC plus precision/recall **at the threshold you will actually
   act on**, and calibrate that threshold to your team's capacity.
4. Re-check periodically: donor behaviour and your data pipeline both drift.
