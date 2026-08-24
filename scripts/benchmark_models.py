"""Reproducible benchmark of PhilanthroPy binary classifiers.

Builds a synthetic labelled donor pool with
``philanthropy.datasets.generate_synthetic_donor_data``, does a stratified
train/test split, fits every applicable binary classifier in
``philanthropy.models``, and prints precision / recall / f1 / roc_auc on the
held-out test set.

``MovesManagementClassifier`` is intentionally excluded: it is a multi-class
moves-management stage predictor, not a binary classifier for the
``is_major_donor`` label.

Run from the repo root:

    python scripts/benchmark_models.py
"""

from __future__ import annotations

import os
import sys

# Import the philanthropy package co-located with this script (the repo under
# test), ahead of any editable install pointing elsewhere on the machine.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from philanthropy.datasets import generate_synthetic_donor_data
from philanthropy.models import (
    DonorPropensityModel,
    LapsePredictor,
    MajorGiftClassifier,
    PlannedGivingIntentScorer,
    PropensityScorer,
)

# Five seeds, not one: a three-decimal score from a single split reads as a
# claim about the method when it is mostly a claim about the split.
SEEDS = (42, 43, 44, 45, 46)
N_SAMPLES = 4000
FEATURES = ["total_gift_amount", "years_active", "event_attendance_count"]
LABEL = "is_major_donor"


def build_models(seed):
    """Return {name: estimator} for every applicable binary classifier."""
    return {
        "PropensityScorer (baseline)": PropensityScorer(),
        "DonorPropensityModel": DonorPropensityModel(random_state=seed),
        "MajorGiftClassifier": MajorGiftClassifier(random_state=seed),
        "LapsePredictor": LapsePredictor(random_state=seed),
        "PlannedGivingIntentScorer": PlannedGivingIntentScorer(random_state=seed),
    }


METRICS = ("precision", "recall", "f1", "roc_auc")


def run_one_seed(seed):
    """Fit every model on one seed's split; return {name: {metric: value}}."""
    df = generate_synthetic_donor_data(n_samples=N_SAMPLES, random_state=seed)
    X = df[FEATURES].to_numpy()
    y = df[LABEL].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    results = {}
    for name, model in build_models(seed).items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            results[name] = {
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }
        except Exception as exc:  # report, never fabricate
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return results, y.mean(), len(y_test)


def main() -> None:
    per_seed = {}
    positive_rate = test_n = None
    for seed in SEEDS:
        per_seed[seed], positive_rate, test_n = run_one_seed(seed)

    print(
        f"Synthetic donor pool: {N_SAMPLES} rows, "
        f"positive rate {positive_rate:.3f} (major donors), "
        f"features={FEATURES}"
    )
    print(f"Stratified 75/25 split, test n={test_n}, seeds={list(SEEDS)}")
    print("Each cell is mean (min-max) across the seeds.\n")

    header = f"{'model':<30}" + "".join(f"{m:>22}" for m in METRICS)
    print(header)
    print("-" * len(header))

    for name in build_models(SEEDS[0]):
        row = f"{name:<30}"
        for metric in METRICS:
            values = [
                per_seed[s][name][metric]
                for s in SEEDS
                if "error" not in per_seed[s][name]
            ]
            if not values:
                row += f"{'ERROR':>22}"
                continue
            mean = sum(values) / len(values)
            row += f"{mean:>10.3f} ({min(values):.3f}-{max(values):.3f})"
        print(row)

    for seed in SEEDS:
        for name, res in per_seed[seed].items():
            if "error" in res:
                print(f"\n{name} (seed {seed}) ERROR: {res['error']}")


if __name__ == "__main__":
    main()
