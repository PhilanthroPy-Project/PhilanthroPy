"""Real-data replication of ``scripts/leakage_experiment.py``.

That script runs two experiments on a *synthetic* donor-year panel: does the
choice of CV splitter matter (A), and does building features as-of each
period versus over whole history matter (B). Both are the argument this
library exists to make, and both were only ever demonstrated on data this
package's own author generated. This script re-runs the identical two
experiments on a real donor file, KDD Cup 1998 (see
:func:`philanthropy.datasets.fetch_kdd98_donors`), because a claim that only
holds on data built to hold it is not evidence.

## From wide promotion history to a donor-period panel

KDD Cup 1998 ships one row per donor with 24 direct-mail promotions,
``ADATE_2``..``ADATE_24`` (mail date), and, for promotions 3-24,
``RAMNT_3``..``RAMNT_24`` (amount given, absent if no gift). Promotion 2 is
the held-out 97NK mailing being scored by the original competition
(``TARGET_B``/``TARGET_D``); it has no ``RAMNT_2`` column for that reason.
Column index does not track calendar order for an individual donor's mail
history (donors are not mailed on identical schedules), but it tracks the
*campaign's* mail date exactly, because every recipient of a given campaign
was mailed on the same date: ``ADATE_2`` is 9706 (June 1997) for every row.
So promotions 3-24, oldest to newest, are exactly the reverse of their column
index -- 24 is oldest, 3 is newest, immediately followed by the 2/97NK
target -- for every donor, with no per-row date parsing required.

That gives 22 chronological periods per donor. Period *p*'s label is
"did this donor give at period *p+1*", mirroring "gave in the following
year" in the synthetic panel; for the last historical period, period *p+1*
**is** the 97NK mailing, so its label is ``TARGET_B`` itself, not something
derived. A donor not mailed a given campaign contributes no gift that period,
which is indistinguishable here from being mailed and not responding; both
are recorded as zero. Two feature sets are built exactly as in the synthetic
script: **as-of** (cumulative amount/count through period *p*, inclusive)
and **whole-history** (the same aggregates over all 22 periods, including
ones after *p*, which is the leak this library exists to prevent). Real data
has no seed to regenerate, so `SEEDS` here vary only the classifier's and the
splitters' own randomness, not the panel itself.

## Pre-registered prediction, recorded before this script was run

The synthetic script measures **A** (splitter choice) as a 0.014-0.030 AUC
gap and **B** (whole-history leakage) as +0.126 AUC of inflation. Real donor
giving should be *more* autocorrelated than the synthetic drift model (habits
persist; there is no manufactured drift working against them), so leakage
should still inflate the score, but a fixed-effect classifier already
captures more of that persistence from as-of history alone here than in the
synthetic panel. Prediction: **B is positive but smaller than +0.126**, in
the neighbourhood of +0.03 to +0.08 AUC; **A stays small**, comparable to the
synthetic 0.01-0.03. If the real numbers land outside that range, that
disagreement is reported below as the finding, not adjusted away.

Run from the repo root:

    python scripts/real_data_leakage_experiment.py
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

from philanthropy.datasets import fetch_kdd98_donors
from philanthropy.model_selection import FiscalYearGroupedSplitter

SEEDS = [42, 43, 44, 45, 46]
FEATURES = ["total", "n", "recent"]
# Oldest to newest promotion index: 24 is the earliest campaign, 3 the latest
# before the held-out 97NK/TARGET_B mailing.
HISTORICAL_PROMOTIONS = list(range(24, 2, -1))


def _panel():
    """The real donor-period panel, built once (the data has no seed)."""
    df = fetch_kdd98_donors()
    ramnt = df[[f"RAMNT_{i}" for i in HISTORICAL_PROMOTIONS]].fillna(0.0).to_numpy()
    gave = ramnt > 0
    target_b = df["TARGET_B"].to_numpy().astype(int)
    donor = df["CONTROLN"].to_numpy()

    whole_total, whole_n = ramnt.sum(1), gave.sum(1)
    n_periods = len(HISTORICAL_PROMOTIONS)

    as_of, whole = [], []
    cum_total = np.zeros(len(df))
    cum_n = np.zeros(len(df))
    for p in range(n_periods):
        recent = ramnt[:, p]
        as_of_total, as_of_n = cum_total + recent, cum_n + gave[:, p]
        label = gave[:, p + 1].astype(int) if p < n_periods - 1 else target_b
        common = dict(donor=donor, fy=p, recent=recent, y=label)
        as_of.append(pd.DataFrame(dict(total=as_of_total, n=as_of_n, **common)))
        whole.append(pd.DataFrame(dict(total=whole_total, n=whole_n, **common)))
        cum_total, cum_n = as_of_total, as_of_n
    return pd.concat(as_of, ignore_index=True), pd.concat(whole, ignore_index=True)


def _clf(seed):
    return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)


def _cv(df, splitter, use_groups, seed):
    """Cross-validate on the panel EXCLUDING its final period.

    Excluding it matters, for the same reason as in the synthetic script: the
    target is "train on everything before the final period, score the final
    period", which is exactly what the last fold of a walk-forward splitter
    does. Leaving the final period in would let walk-forward win by
    construction rather than on merit.
    """
    df = df[df["fy"] < df["fy"].max()]
    X, y, fy = df[FEATURES].to_numpy(), df["y"].to_numpy(), df["fy"].to_numpy()
    kwargs = {"groups": fy} if use_groups else {}
    return cross_val_score(
        _clf(seed), X, y, cv=splitter, scoring="roc_auc", **kwargs
    ).mean()


def _true_future(df, seed):
    """Fit on every period but the last, score the last. The honest target."""
    X, y, fy = df[FEATURES].to_numpy(), df["y"].to_numpy(), df["fy"].to_numpy()
    train, test = fy < fy.max(), fy == fy.max()
    model = _clf(seed).fit(X[train], y[train])
    return roc_auc_score(y[test], model.predict_proba(X[test])[:, 1])


def _fmt(values):
    return f"{np.mean(values):.3f} ({min(values):.3f}-{max(values):.3f})"


def main():
    as_of, whole = _panel()
    n_periods = as_of["fy"].nunique()

    random_cv, walk_cv, truth, whole_cv = [], [], [], []
    for seed in SEEDS:
        truth.append(_true_future(as_of, seed))
        random_cv.append(
            _cv(as_of, StratifiedKFold(5, shuffle=True, random_state=seed), False, seed)
        )
        walk_cv.append(_cv(as_of, FiscalYearGroupedSplitter(n_splits=3), True, seed))
        whole_cv.append(_cv(whole, FiscalYearGroupedSplitter(n_splits=3), True, seed))

    print(f"KDD Cup 1998 donor-period panel: {as_of['donor'].nunique()} donors x "
          f"{n_periods} promotion periods, label = gave at the following period "
          f"(final period = TARGET_B).")
    print(f"Seeds {SEEDS}. Each cell is mean (min-max) ROC-AUC.\n")

    print("A. Does the CV split matter? (as-of features throughout)")
    print(f"  true future, held-out final period : {_fmt(truth)}")
    print(f"  walk-forward fiscal-year CV        : {_fmt(walk_cv)}"
          f"   error {np.mean(walk_cv) - np.mean(truth):+.3f}")
    print(f"  random StratifiedKFold CV          : {_fmt(random_cv)}"
          f"   error {np.mean(random_cv) - np.mean(truth):+.3f}")

    print("\nB. Does feature construction matter? (walk-forward CV throughout)")
    print(f"  features built as of each period   : {_fmt(walk_cv)}")
    print(f"  same features over whole history   : {_fmt(whole_cv)}")
    print(f"  inflation from whole-history        : "
          f"{np.mean(whole_cv) - np.mean(walk_cv):+.3f} AUC")

    print("\nCompare against the pre-registered prediction and the synthetic run "
          "\nin this script's own docstring and in docs/explanation/benchmarks.md. "
          "\nReport whichever number actually came out, including a smaller or "
          "\nnegative effect; that disagreement is the finding.")


if __name__ == "__main__":
    main()
