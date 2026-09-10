# Use the CLI

Installing the package puts a `philanthropy` executable on your PATH. It is a CSV-in / CSV-out interface for analysts who are not primarily Python engineers: turn a raw CRM gift export into a feature table, train a model from it, score a prospect list, and report holdout metrics, no Python file to write.

```bash
philanthropy --version
philanthropy --help
```

Four subcommands: `features`, `train`, `score`, `validate`.

## Build the feature table from a CRM gift export

`train` wants one row per donor with columns like `total_gift_amount` and `years_active`. A CRM gift export is one row per *gift*, so `features` does the roll-up.

```bash
philanthropy features --source raisers_edge --data gifts.csv --out features.csv
```

`--source` accepts `raisers_edge` (Blackbaud Raiser's Edge and RE NXT) or `civicrm`. `--data` takes a single CSV or a directory of them, walked recursively, which is the shape of a folder of monthly exports. Omit `--out` and the CSV goes to stdout.

The output has one row per donor and these columns:

`contact_id`, `constituent_email`, `first_name`, `last_name`, `total_gift_amount`, `gift_count`, `largest_gift_amount`, `first_gift_date`, `last_gift_date`, `years_active`, `recency_days`, `distinct_financial_types`

Header spelling is normalised for you, so a desktop Export (`Constituent ID`, `Gift Date`, `Gift Amount`, `Gift Type`) and a SKY API pull (`constituent_id`, `date`, `amount`, `type`) both work.

!!! warning "A pledge is not a payment, and `features` knows the difference"

    In Raiser's Edge a pledge and the money paid against it are **separate gift records**, and a recurring gift row is a template rather than a sum ever received. Adding up the amount column double-counts every committed dollar. `features` drops the commitment rows (`Pledge`, `Matching Gift Pledge`, `Recurring Gift`) and the ledger corrections, and keeps the payments (`Pay-Cash`, `PledgePayment`, `RecurringGiftPayment`, ...). Export the **Gift Type** field or it cannot do this, and it will warn you. The excluded set is the `exclude_gift_types` parameter of `philanthropy.ingest.raisers_edge_gifts_to_features` if your site spells its types differently. For CiviCRM the equivalent traps are test-mode rows and non-`Completed` contributions, and they are dropped the same way.

!!! note "`features` does not invent a label"

    `train` needs a `--target` column and a gift export contains no such column. Deciding who counts as a major donor, who lapsed, or who is a planned-giving prospect is yours to define; `features` gets you the predictors, not the answer key.


## Train a model from a labelled CSV

`train` needs the label column, the feature columns, and an output path. Point it at the `features.csv` from the previous step once you have added your own label column to it. It writes a **bundle** (the fitted model plus the feature list, the target name, and the library versions) via `philanthropy.utils.save_model`.

```bash
philanthropy train \
  --data gifts.csv \
  --target is_major_donor \
  --features total_gift_amount,years_active,event_attendance_count \
  --model DonorPropensityModel \
  --out model.joblib
```

`--model` accepts `DonorPropensityModel` (the default), `MajorGiftClassifier`, `LapsePredictor`, or `PlannedGivingIntentScorer`. `--random-state` defaults to `0`, so a rerun on the same CSV gives the same model.

## Score a prospect list

`score` reuses the feature list stored in the bundle, so you do not repeat `--features`. Omit `--out` and the CSV goes to stdout, which pipes.

```bash
philanthropy score --model model.joblib --data prospects.csv --out scores.csv

# or straight to a pipe
philanthropy score --model model.joblib --data prospects.csv | head -20
```

The output is the input CSV plus one `score` column: `predict_affinity_score` when the model has one, otherwise `predict_proba[:, 1]`.

!!! warning "Scored CSVs are neutralised against formula injection"
    Donor-controlled string fields arrive from third-party webhooks. A cell like `=cmd|'/c calc'!A1` executes when an analyst opens the file in Excel or Google Sheets. Any text cell starting with `=`, `+`, `-`, `@`, tab or CR is prefixed with an apostrophe on the way out. Numeric columns are untouched.

## Report holdout metrics

`validate` prints precision, recall, F1 and ROC-AUC. The target comes from the bundle unless you override it.

```bash
philanthropy validate --model model.joblib --data holdout.csv
philanthropy validate --model model.joblib --data holdout.csv --target is_major_donor
```

## End to end, in Python

The same three steps, so this page is executable. `main()` takes the argument list a shell would pass.

```python
import pandas as pd
from philanthropy.cli import main
from philanthropy.datasets import generate_synthetic_donor_data

FEATURES = "total_gift_amount,years_active,event_attendance_count"
generate_synthetic_donor_data(n_samples=300, random_state=1).to_csv("gifts.csv", index=False)

main(["train", "--data", "gifts.csv", "--target", "is_major_donor",
      "--features", FEATURES, "--out", "model.joblib"])

main(["score", "--model", "model.joblib", "--data", "gifts.csv", "--out", "scores.csv"])
scored = pd.read_csv("scores.csv")
print(scored[["total_gift_amount", "score"]].head())

main(["validate", "--model", "model.joblib", "--data", "gifts.csv"])

assert "score" in scored.columns
assert len(scored) == 300
```

## Bundles are pickles

!!! danger "Only load model files you trust"
    `--model` unpickles the file, which executes arbitrary code. Never point `score` or `validate` at a bundle from an untrusted source.

The artifacts are the same objects `save_model` / `load_model` produce, so anything the CLI trains loads in Python and vice versa. See [Save and load models](save_and_load_models.md).
