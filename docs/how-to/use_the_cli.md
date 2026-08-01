# Use the CLI

Installing the package puts a `philanthropy` executable on your PATH. It is a CSV-in / CSV-out interface for analysts who are not primarily Python engineers: train a model from a labelled export, score a prospect list, and report holdout metrics — no Python file to write.

```bash
philanthropy --version
philanthropy --help
```

Three subcommands: `train`, `score`, `validate`.

## Train a model from a labelled CSV

`train` needs the label column, the feature columns, and an output path. It writes a **bundle** — the fitted model plus the feature list, the target name, and the library versions — via `philanthropy.utils.save_model`.

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
