"""Method reference — the donor-analytics pipeline from Lalakiya (2025).

Reproduces the *method* described in:

    S. A. Lalakiya, "AI for Advancement: Predictive Donor Analytics and
    Fundraising Intelligence at Scale," 2025 IEEE 11th Int. Conf. on Computing,
    Engineering and Design (ICCED). DOI: 10.1109/ICCED68324.2025.11325064.

Pipeline: RFM feature engineering -> a panel of classifiers scored by CV ->
permutation importance -> feature correlation. This mirrors the paper's
Tables III/IV/VI/IX and Fig. 6, using PhilanthroPy's own ``RFMTransformer`` and
``donor_feature_importance``.

Scope (read this): a reference implementation of the *method* on reproducible
SYNTHETIC donor data. It is **not** a reproduction of the paper's reported
metrics, and it is **not** run on the paper's cited dataset. (That dataset — NYC
CIOB "Official Fundraising by City Agencies", exposed here as
``load_ciob_fundraising`` — is an agency<->nonprofit affiliation registry with
no donor-level giving, so it cannot produce RFM features or an engagement
target.) Every number below depends only on the synthetic generator seeded here.

Boosting stand-in: scikit-learn's ``HistGradientBoosting`` replaces the paper's
XGBoost — PhilanthroPy depends only on the scikit-learn stack.

Run it:

    python examples/method_reference_lalakiya2025.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from philanthropy.inspection import donor_feature_importance
from philanthropy.preprocessing import RFMTransformer

RANDOM_STATE = 0


def _z(values) -> np.ndarray:
    """Standardise to zero mean / unit variance."""
    arr = np.asarray(values, dtype=float)
    return (arr - arr.mean()) / (arr.std() + 1e-9)


def _synthetic_gift_log(n_donors: int = 700, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """A synthetic gift-transaction log: one row per gift.

    Columns match what ``RFMTransformer`` expects (``donor_id``, ``gift_date``,
    ``gift_amount``). Donors vary in how often and how recently they give, so the
    aggregated RFM features carry a real signal.
    """
    rng = np.random.default_rng(seed)
    ref = pd.Timestamp("2025-01-01")
    donor_ids, dates, amounts = [], [], []
    for donor in range(n_donors):
        n_gifts = int(rng.integers(1, 25))          # frequency varies by donor
        recency_bias = rng.beta(1.5, 3.0)           # some donors give recently
        for _ in range(n_gifts):
            days_ago = int(rng.integers(0, 365 * 5) * (0.4 + recency_bias))
            donor_ids.append(donor)
            dates.append(ref - pd.Timedelta(days=days_ago))
            amounts.append(round(float(rng.lognormal(6.5, 1.1)), 2))
    return pd.DataFrame(
        {"donor_id": donor_ids, "gift_date": dates, "gift_amount": amounts}
    )


def _engagement_label(rfm: pd.DataFrame, seed: int = RANDOM_STATE) -> np.ndarray:
    """Noisy latent engagement target derived from RFM (cf. paper Table IX).

    Engagement rises with Frequency and Monetary and falls with Recency. Gaussian
    noise keeps the label non-trivial so the classifier panel has real signal to
    learn rather than a closed-form rule.
    """
    rng = np.random.default_rng(seed + 1)
    z = (
        0.9 * _z(rfm["frequency"])
        + 1.0 * _z(np.log1p(rfm["monetary"]))
        - 0.8 * _z(rfm["recency"])
        + rng.normal(0, 0.8, len(rfm))
    )
    return (z > np.median(z)).astype(int)


def main() -> None:
    # 1) RFM feature engineering — the paper's core preprocessing step.
    gifts = _synthetic_gift_log()
    rfm = RFMTransformer().fit_transform(gifts)
    X = rfm[["recency", "frequency", "monetary"]]
    y = _engagement_label(rfm)

    # 2) Classifier panel scored by 5-fold CV (paper Tables III/IV/VI analog).
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=150, random_state=RANDOM_STATE
        ),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "GaussianNB": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, est in models.items():
        scores = cross_validate(est, X, y, cv=cv, scoring=["accuracy", "roc_auc"])
        rows.append(
            (
                name,
                scores["test_accuracy"].mean(),
                scores["test_accuracy"].std(),
                scores["test_roc_auc"].mean(),
            )
        )
    table = pd.DataFrame(
        rows, columns=["model", "accuracy", "acc_std", "auc"]
    ).sort_values("auc", ascending=False, ignore_index=True)

    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    print("Model comparison (5-fold CV on synthetic RFM features):")
    print(table.to_string(index=False, float_format=fmt))

    # 3) Permutation importance via the library helper (paper Fig. 6 / SHAP analog).
    best = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE).fit(X, y)
    importance = donor_feature_importance(best, X, y, random_state=RANDOM_STATE)
    print("\nFeature importance (permutation — SHAP analog):")
    print(importance.to_string(index=False, float_format=fmt))

    # 4) RFM<->engagement correlation (paper Table IX analog).
    corr = X.assign(engagement=y).corr()
    print("\nRFM-Engagement correlation matrix:")
    print(corr.to_string(float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
