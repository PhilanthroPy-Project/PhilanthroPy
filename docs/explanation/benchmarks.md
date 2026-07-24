# Model Validation & Benchmarks

This page answers "how good is this, actually?" with reproducible numbers — and,
just as importantly, explains why you should **not** trust them for your program
without re-validating on your own data.

!!! warning "These numbers are on synthetic data"
    The table below is measured on `generate_synthetic_donor_data`, a reproducible
    but **artificial** donor pool. Synthetic features are cleanly separable by
    construction, so these scores are optimistic and say nothing about how a model
    will perform on your CRM. **Re-run the evaluation on your own labelled giving
    history before trusting any score in production.**

## Reproducing this table

The benchmark is a committed, dependency-free script:

```bash
python scripts/benchmark_models.py
```

It builds a 4,000-row synthetic pool (`random_state=42`), takes a stratified
75/25 train/test split, fits every applicable binary classifier in
`philanthropy.models` on the documented feature set
(`total_gift_amount`, `years_active`, `event_attendance_count`) against the
`is_major_donor` label, and prints precision / recall / F1 / ROC-AUC on the
held-out test set using `sklearn.metrics`.

## Results

Synthetic pool: 4,000 rows, positive rate 0.677; test split 1,000 rows.

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| `PropensityScorer` (baseline) | 0.677 | 1.000 | 0.807 | 0.500 |
| `DonorPropensityModel` | 0.895 | 0.935 | 0.915 | 0.931 |
| `MajorGiftClassifier` | 0.869 | 0.966 | 0.915 | 0.932 |
| `LapsePredictor` | 0.895 | 0.935 | 0.915 | 0.931 |
| `PlannedGivingIntentScorer` | 0.883 | 0.951 | 0.916 | 0.941 |

*(Measured with scikit-learn 1.7.2 on the synthetic dataset; your numbers will
differ.)*

## How to read this

- **`PropensityScorer` is the floor.** It is a constant-probability baseline
  (P=0.5): a ROC-AUC of 0.500 is exactly "no better than chance." Every real
  model must beat it — here they all do, by a wide margin, but that margin is
  inflated by the synthetic data's separability.
- **`LapsePredictor` and `DonorPropensityModel` report identical numbers** on
  this task because both wrap a default `RandomForestClassifier` with the same
  `random_state` and features. That is expected, not a bug — `LapsePredictor` is
  purpose-built for a lapse label, not `is_major_donor`; it appears here only
  because its estimator is applicable.
- **ROC-AUC is the most transferable metric** across base rates; precision and
  recall depend on the 0.5 decision threshold and this pool's 0.677 positive
  rate, which is far higher than a real major-donor base rate (typically a few
  percent). Expect precision to fall sharply on realistically imbalanced data.

## Validating on your own data

1. Assemble a labelled historical dataset (features + a binary outcome you can
   observe, e.g. "made a major gift in the following year").
2. Split **temporally**, not randomly — train on earlier years, test on later
   ones — using `FiscalYearGroupedSplitter` to avoid leakage across fiscal
   boundaries.
3. Report ROC-AUC plus precision/recall **at the threshold you will actually
   act on**, and calibrate that threshold to your team's capacity.
4. Re-check periodically: donor behaviour and your data pipeline both drift.
