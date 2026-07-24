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

RANDOM_STATE = 42
N_SAMPLES = 4000
FEATURES = ["total_gift_amount", "years_active", "event_attendance_count"]
LABEL = "is_major_donor"


def build_models():
    """Return {name: estimator} for every applicable binary classifier."""
    return {
        "PropensityScorer (baseline)": PropensityScorer(),
        "DonorPropensityModel": DonorPropensityModel(random_state=RANDOM_STATE),
        "MajorGiftClassifier": MajorGiftClassifier(random_state=RANDOM_STATE),
        "LapsePredictor": LapsePredictor(random_state=RANDOM_STATE),
        "PlannedGivingIntentScorer": PlannedGivingIntentScorer(
            random_state=RANDOM_STATE
        ),
    }


def main() -> None:
    df = generate_synthetic_donor_data(n_samples=N_SAMPLES, random_state=RANDOM_STATE)
    X = df[FEATURES].to_numpy()
    y = df[LABEL].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    print(
        f"Synthetic donor pool: {N_SAMPLES} rows, "
        f"positive rate {y.mean():.3f} (major donors), "
        f"features={FEATURES}, random_state={RANDOM_STATE}"
    )
    print(
        f"Train/test split: {len(y_train)}/{len(y_test)} "
        f"(test positive rate {y_test.mean():.3f})\n"
    )

    header = (
        f"{'model':<30}{'precision':>11}{'recall':>11}"
        f"{'f1':>11}{'roc_auc':>11}"
    )
    print(header)
    print("-" * len(header))

    for name, model in build_models().items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba)
            print(
                f"{name:<30}{precision:>11.3f}{recall:>11.3f}"
                f"{f1:>11.3f}{auc:>11.3f}"
            )
        except Exception as exc:  # report, never fabricate
            print(f"{name:<30}  ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
