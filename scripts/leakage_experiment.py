"""Quantify where temporal leakage actually comes from in donor analytics.

Two experiments on a seeded synthetic donor-year panel, five seeds each.

**A. Does the choice of CV split matter?** Random ``StratifiedKFold`` versus
walk-forward ``FiscalYearGroupedSplitter``, each compared against a genuinely
held-out final year. The received wisdom is that a random split flatters you.

**B. Does the choice of feature construction matter?** The same walk-forward CV,
run once on features computed **as of** each panel year and once on the same
aggregates computed over the **whole** export including future years. This is
the mistake the library exists to prevent: build features once over the full
history, then split.

Run from the repo root:

    python scripts/leakage_experiment.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from philanthropy.datasets import make_donor_panel
from philanthropy.model_selection import FiscalYearGroupedSplitter

SEEDS = [42, 43, 44, 45, 46]
N_DONORS = 3000
YEARS = list(range(2018, 2025))
FEATURES = ["total", "n", "recent"]


def _panel(seed):
    """Aggregate the shipped donor panel two ways: as-of, and whole-history.

    The panel itself comes from ``philanthropy.datasets.make_donor_panel``, so
    this experiment and the tutorials run on the same generator rather than on
    a private copy of it. Its persistence is deliberate: real donors have
    habits, and that is what any leakage has to exploit. Its drift makes later
    years genuinely harder, so a CV scheme cannot look good just by ignoring
    time.

    What happens *here* is the actual subject: the same aggregates computed
    once as of each panel year, and once over the whole export.
    """
    gifts = make_donor_panel(
        n_donors=N_DONORS,
        n_years=len(YEARS),
        start_fiscal_year=YEARS[0],
        random_state=seed,
    )["gifts"]

    # Dense donor x year amounts. At most one gift per donor-year, which the
    # generator guarantees, so this assignment cannot silently drop a gift.
    amount = np.zeros((N_DONORS, len(YEARS)))
    amount[gifts["donor_id"], gifts["fiscal_year"] - YEARS[0]] = gifts["gift_amount"]
    gave = amount > 0

    as_of, whole = [], []
    for j, year in enumerate(YEARS[:-1]):
        label = gave[:, j + 1].astype(int)   # gave in the FOLLOWING year
        common = dict(donor=np.arange(N_DONORS), fy=year, recent=amount[:, j], y=label)
        as_of.append(pd.DataFrame(
            dict(total=amount[:, : j + 1].sum(1), n=gave[:, : j + 1].sum(1), **common)
        ))
        whole.append(pd.DataFrame(
            dict(total=amount.sum(1), n=gave.sum(1), **common)
        ))
    return pd.concat(as_of, ignore_index=True), pd.concat(whole, ignore_index=True)


def _clf():
    return RandomForestClassifier(n_estimators=200, random_state=0)


def _cv(df, splitter, use_groups):
    """Cross-validate on the panel EXCLUDING its final year.

    Excluding it matters. The target below is "train on everything before the
    final year, score the final year", which is precisely what the last fold of
    a walk-forward splitter does. Leaving the final year in the CV data would put
    the estimand inside one estimator and not the other, and walk-forward would
    win by construction rather than on merit.
    """
    df = df[df["fy"] < df["fy"].max()]
    X, y, fy = df[FEATURES].to_numpy(), df["y"].to_numpy(), df["fy"].to_numpy()
    kwargs = {"groups": fy} if use_groups else {}
    return cross_val_score(_clf(), X, y, cv=splitter, scoring="roc_auc", **kwargs).mean()


def _true_future(df):
    """Fit on every year but the last, score the last. The honest target."""
    X, y, fy = df[FEATURES].to_numpy(), df["y"].to_numpy(), df["fy"].to_numpy()
    train, test = fy < fy.max(), fy == fy.max()
    model = _clf().fit(X[train], y[train])
    return roc_auc_score(y[test], model.predict_proba(X[test])[:, 1])


def _fmt(values):
    return f"{np.mean(values):.3f} ({min(values):.3f}-{max(values):.3f})"


def main():
    random_cv, walk_cv, truth, whole_cv = [], [], [], []
    for seed in SEEDS:
        as_of, whole = _panel(seed)
        truth.append(_true_future(as_of))
        random_cv.append(_cv(as_of, StratifiedKFold(5, shuffle=True, random_state=0), False))
        walk_cv.append(_cv(as_of, FiscalYearGroupedSplitter(n_splits=3), True))
        whole_cv.append(_cv(whole, FiscalYearGroupedSplitter(n_splits=3), True))

    print(f"Donor-year panel: {N_DONORS} donors x {len(YEARS) - 1} panel years, "
          f"label = gave in the following year.")
    print(f"Seeds {SEEDS}. Each cell is mean (min-max) ROC-AUC.\n")

    print("A. Does the CV split matter? (as-of features throughout)")
    print(f"  true future, held-out final year : {_fmt(truth)}")
    print(f"  walk-forward fiscal-year CV      : {_fmt(walk_cv)}"
          f"   error {np.mean(walk_cv) - np.mean(truth):+.3f}")
    print(f"  random StratifiedKFold CV        : {_fmt(random_cv)}"
          f"   error {np.mean(random_cv) - np.mean(truth):+.3f}")

    print("\nB. Does feature construction matter? (walk-forward CV throughout)")
    print(f"  features built as of each year   : {_fmt(walk_cv)}")
    print(f"  same features over whole history : {_fmt(whole_cv)}")
    print(f"  inflation from whole-history      : "
          f"{np.mean(whole_cv) - np.mean(walk_cv):+.3f} AUC")

    print("\nRead this as: the split matters little once features are built as of "
          "\nthe decision point, and feature construction matters a lot. "
          "\nBoth numbers are synthetic; see docs/explanation/benchmarks.md.")


if __name__ == "__main__":
    main()
