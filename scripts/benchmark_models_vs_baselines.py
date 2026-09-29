"""
scripts/benchmark_models_vs_baselines.py
=========================================
Does each model beat the simple domain rule a fundraising shop already uses,
on data it was not fit on?

That is a different, and lower, bar than the per-model accuracy table in
``scripts/benchmark_models.py`` / ``docs/explanation/benchmarks.md``: this
script pairs every estimator with the naive rule it is meant to replace
("rank by last year's total", "predict whatever they gave last time", "mail
everyone"), evaluates both on a held-out, walk-forward split, and reports
whether the model actually earns its complexity. Two datasets: a synthetic
multi-year donor panel (:func:`~philanthropy.datasets.make_donor_panel`, at
least five seeds, mean and min-max range) and, opt-in, the real KDD Cup 1998
direct-mail file (:func:`~philanthropy.datasets.fetch_kdd98_donors`, one
seed, because it is one real file with no second draw to average over).

Run:

    python scripts/benchmark_models_vs_baselines.py                     # both datasets
    python scripts/benchmark_models_vs_baselines.py --skip-kdd98         # synthetic only, no download
    python scripts/benchmark_models_vs_baselines.py --fast --skip-kdd98  # one-seed smoke run, <20s
    python scripts/benchmark_models_vs_baselines.py --out results/bench  # also writes bench.json / bench.csv
    python scripts/benchmark_models_vs_baselines.py --with-cup98val      # also scores on cup98VAL (opt-in, +~37MB)

The KDD Cup 1998 section downloads ``cup98lrn.zip`` (~36 MB) to
``~/philanthropy_data`` on first use (see ``fetch_kdd98_donors``); pass
``--skip-kdd98`` to stay offline entirely. ``--with-cup98val`` additionally
scores cost-aware selection on KDD98's own held-out validation file
(``cup98VAL.zip`` + ``valtargt.txt``, another ~37 MB, see
``fetch_kdd98_val_donors``), reported next to the random-split number rather
than replacing it; off by default and has no effect with
``--skip-kdd98``.

## What is measured

For every classifier: top-1%/5%/10% hit rate (and its lift over the
baseline), ROC-AUC, average precision, and a decile calibration gap (mean
absolute difference between predicted and actual rate across probability
deciles, lower is better). For every amount regressor: mean absolute error
and the share of predictions within 25% of the true amount. For
``PlannedGivingIntentScorer`` and ``GiftIntervalCalibrator``, which have no
matching label or baseline rule in the synthetic generator, the check is
narrower: does the classifier beat chance, and does the calibrated interval
attain roughly its requested coverage.

## Verdict rule

Every row with a baseline gets one of three verdicts, from a single ratio:
``ratio = model / baseline`` for a metric where higher is better (hit rate,
ROC-AUC, net revenue), or ``ratio = baseline / model`` where lower is better
(MAE, the calibration gap). ``ratio >= 1.15`` is ``"wins"``, ``1.00 <= ratio
< 1.15`` is ``"modest"``, ``ratio < 1.00`` is ``"loses"``. A row with no
baseline (average precision) uses ``"n/a"``; the two coverage checks use a
coverage-specific rule instead: attained coverage within 3 points of the
requested level is ``"wins"``, within 6 points is ``"modest"``, more than 6
points short is ``"loses"``.

## Reading the KDD Cup 1998 numbers against the reference run

A prior full run (seed 42, no subsampling) measured: upgrade model top-1%
hit rate 17.4% vs a 7.8% baseline; ``DonorPropensityModel`` ROC-AUC 0.508;
``MajorGiftClassifier`` ROC-AUC 0.589; ``LapsePredictor`` ROC-AUC 0.556 vs a
0.503 baseline; ``AskAmountRecommender`` MAE $4.54 vs $3.89 for "last gift";
cost-aware net revenue $4,240 vs $3,149 for mailing everyone. This script
does not force its own numbers to match: other work may have touched the
estimators it calls since, and a widening gap without a matching nearby code
change is worth investigating rather than silently accepting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from philanthropy.datasets import fetch_kdd98_donors, fetch_kdd98_val_donors, make_donor_panel
from philanthropy.ingest import build_upgrade_snapshots
from philanthropy.ingest._upgrade_snapshots import _fy_end
from philanthropy.metrics import fundraising_roi
from philanthropy.model_selection import FiscalYearGroupedSplitter
from philanthropy.models import (
    AskAmountRecommender,
    DonorPropensityModel,
    GiftIntervalCalibrator,
    LapsePredictor,
    MajorGiftClassifier,
    PlannedGivingIntentScorer,
)
from philanthropy.preprocessing import (
    CRMCleaner,
    FiscalYearTransformer,
    RFMTransformer,
    WealthScreeningImputer,
)

DEFAULT_SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46)
KDD_SEED = 42
KDD_AS_OF = "1997-06-01"
WIN_RATIO = 1.15
COVERAGE_TOL = 0.03
N_BOOTSTRAP = 1000
BOOTSTRAP_METRICS = ("top10pct_hit_rate", "roc_auc")
KDD_SPLIT = "55/15/30 stratified (train/validation/test)"
SYNTHETIC_SPLIT = "walk-forward (train on fiscal years < T, test on T)"

CSV_FIELDS = ("dataset", "model", "metric", "value", "baseline", "lo", "hi", "n_seeds", "verdict", "note")


# --------------------------------------------------------------------------- #
# E.11a rule 3: fixed baseline rule sets, committed before any run.
# Every classifier row is scored against the *best* of its model's rules
# (the toughest bar available, not a single arbitrary pick); every ask row
# against the *lowest-MAE* of its rules. This module never re-picks a rule
# set after seeing results on a given dataset.
# --------------------------------------------------------------------------- #
def _rfm_cell_score(recency: np.ndarray, frequency: np.ndarray, monetary: np.ndarray) -> np.ndarray:
    """Recency+frequency+monetary quintile sum: the segmentation most CRMs
    implement natively. ``recency`` is periods/months *since* the last gift
    (lower is better), the others are cumulative counts/amounts (higher is
    better); all three are rank-percentiled first so they combine on one
    scale regardless of units."""

    def pct_rank(values: np.ndarray, ascending_is_better: bool) -> np.ndarray:
        r = pd.Series(values).rank(pct=True, method="average")
        return (r if ascending_is_better else 1.0 - r).to_numpy()

    return (
        pct_rank(-np.asarray(recency), True)
        + pct_rank(np.asarray(frequency), True)
        + pct_rank(np.asarray(monetary), True)
    )


def _best_classifier_baseline(
    y_true: np.ndarray, rules: Dict[str, np.ndarray], metric_fn
) -> Tuple[str, np.ndarray, float]:
    """The toughest (highest-scoring) rule in a fixed set, by ``metric_fn``."""
    scored = {name: metric_fn(y_true, score) for name, score in rules.items()}
    best = max(scored, key=scored.get)
    return best, rules[best], scored[best]


def _best_ask_baseline(
    y_true: np.ndarray, rules: Dict[str, np.ndarray]
) -> Tuple[str, np.ndarray, float]:
    """The toughest (lowest-MAE) rule in a fixed set."""
    scored = {name: mean_absolute_error(y_true, score) for name, score in rules.items()}
    best = min(scored, key=scored.get)
    return best, rules[best], scored[best]


def _bootstrap_ci(
    y_true: np.ndarray, score: np.ndarray, metric_fn, n_resamples: int = N_BOOTSTRAP, seed: int = 0,
) -> Tuple[float, float]:
    """Bootstrap 95% interval (E.11a rule 4) for a single-split row: resample
    the test set with replacement ``n_resamples`` times and take the 2.5th
    and 97.5th percentile of ``metric_fn`` over the resamples."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_b = np.asarray(y_true)[idx]
        if len(np.unique(y_b)) < 2:
            continue
        values.append(metric_fn(y_b, np.asarray(score)[idx]))
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


# --------------------------------------------------------------------------- #
# Results table
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    """One (dataset, model, metric) result, with enough of a baseline to
    decide the ``verdict`` documented in this module's own docstring."""

    dataset: str
    model: str
    metric: str
    value: float
    baseline: Optional[float] = None
    lower_is_better: bool = False
    coverage_target: Optional[float] = None
    note: str = ""
    lo: Optional[float] = None
    hi: Optional[float] = None
    n_seeds: int = 1

    @property
    def verdict(self) -> str:
        if self.coverage_target is not None:
            gap = self.coverage_target - self.value
            if gap <= COVERAGE_TOL:
                return "wins"
            if gap <= 2 * COVERAGE_TOL:
                return "modest"
            return "loses"
        if self.baseline is None or self.baseline != self.baseline:
            return "n/a"
        if self.baseline == 0:
            return "wins" if self.value > 0 else "n/a"
        ratio = self.baseline / self.value if self.lower_is_better else self.value / self.baseline
        if ratio >= WIN_RATIO:
            return "wins"
        if ratio >= 1.0:
            return "modest"
        return "loses"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "model": self.model,
            "metric": self.metric,
            "value": self.value,
            "baseline": self.baseline,
            "lo": self.lo,
            "hi": self.hi,
            "n_seeds": self.n_seeds,
            "verdict": self.verdict,
            "note": self.note,
        }


def _aggregate(seed_rows: List[List[Row]]) -> List[Row]:
    """Collapse one row list per seed into mean/min/max per (dataset, model, metric)."""
    groups: Dict[Tuple[str, str, str], List[Row]] = defaultdict(list)
    for rows in seed_rows:
        for r in rows:
            groups[(r.dataset, r.model, r.metric)].append(r)

    out = []
    for (dataset, model, metric), rs in groups.items():
        values = [r.value for r in rs if r.value == r.value]
        if not values:
            continue
        baselines = [r.baseline for r in rs if r.baseline is not None and r.baseline == r.baseline]
        out.append(
            Row(
                dataset=dataset,
                model=model,
                metric=metric,
                value=float(np.mean(values)),
                baseline=float(np.mean(baselines)) if baselines else None,
                lower_is_better=rs[0].lower_is_better,
                coverage_target=rs[0].coverage_target,
                note=rs[0].note,
                lo=float(min(values)),
                hi=float(max(values)),
                n_seeds=len(values),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def _topn_rate(y_true: np.ndarray, score: np.ndarray, frac: float) -> Tuple[float, int]:
    n = max(1, int(round(len(y_true) * frac)))
    idx = np.argsort(-score)[:n]
    return float(np.asarray(y_true)[idx].mean()), n


def _decile_calibration_gap(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean |mean predicted - mean actual| over deciles of predicted probability."""
    df = pd.DataFrame({"y": y_true, "p": proba})
    df["decile"] = pd.qcut(df["p"], 10, labels=False, duplicates="drop")
    g = df.groupby("decile").agg(mean_p=("p", "mean"), mean_y=("y", "mean"))
    return float((g["mean_p"] - g["mean_y"]).abs().mean())


def _classifier_rows(
    dataset: str,
    model_name: str,
    y_true: np.ndarray,
    score: np.ndarray,
    rules: Dict[str, np.ndarray],
    split: str,
    n_configs: int = 1,
    bootstrap: bool = False,
) -> List[Row]:
    """One row per metric, scored against the best (E.11a rule 3) of a fixed
    ``rules`` dict. The rule that wins top-10% hit rate is the one baseline
    identity used for every metric below, so the comparison is against one
    named rule, not a different cherry-picked one per metric. ``split`` and
    ``n_configs`` (E.11a rules 1-2) are recorded in every row's note.
    ``bootstrap=True`` adds a 1,000-resample 95% interval (rule 4) on
    top10pct_hit_rate and roc_auc; that is for single-split (KDD98) rows
    only, synthetic rows already have a 5-seed range."""
    y_true = np.asarray(y_true)
    best_name, baseline_score, _ = _best_classifier_baseline(y_true, rules, lambda y, s: _topn_rate(y, s, 0.10)[0])
    header = f"rule_set={best_name} (best of {len(rules)}); split={split}; n_configs={n_configs}"
    rows = []
    for frac in (0.01, 0.05, 0.10):
        m_rate, n = _topn_rate(y_true, score, frac)
        b_rate, _ = _topn_rate(y_true, baseline_score, frac)
        metric = f"top{int(frac * 100)}pct_hit_rate"
        lo = hi = None
        if bootstrap and metric in BOOTSTRAP_METRICS:
            lo, hi = _bootstrap_ci(y_true, score, lambda y, s: _topn_rate(y, s, frac)[0])
        rows.append(Row(dataset, model_name, metric, m_rate, b_rate, note=f"n={n}; {header}", lo=lo, hi=hi))
    has_both_classes = len(np.unique(y_true)) > 1
    auc = roc_auc_score(y_true, score) if has_both_classes else float("nan")
    b_auc = roc_auc_score(y_true, baseline_score) if has_both_classes else float("nan")
    auc_lo = auc_hi = None
    if bootstrap and has_both_classes:
        auc_lo, auc_hi = _bootstrap_ci(y_true, score, roc_auc_score)
    rows.append(Row(dataset, model_name, "roc_auc", auc, b_auc, note=header, lo=auc_lo, hi=auc_hi))
    ap = average_precision_score(y_true, score) if has_both_classes else float("nan")
    rows.append(Row(dataset, model_name, "average_precision", ap, note=f"no baseline; base-rate dependent; {header}"))
    rows.append(
        Row(
            dataset, model_name, "decile_calibration_gap",
            _decile_calibration_gap(y_true, score), lower_is_better=True,
            note=f"mean |mean predicted - mean actual| across probability deciles; {header}",
        )
    )
    return rows


def _within_pct(pred: np.ndarray, y_true: np.ndarray, pct: float = 0.25) -> float:
    return float((np.abs(pred - y_true) <= pct * y_true).mean())


# --------------------------------------------------------------------------- #
# Synthetic donor panel (make_donor_panel)
# --------------------------------------------------------------------------- #
def _period_panel(n_donors: int, n_years: int, seed: int) -> pd.DataFrame:
    """As-of donor-period panel, one row per (donor, fiscal year).

    Mirrors ``scripts/real_data_leakage_experiment.py``'s panel: ``total``/``n``
    are cumulative through the row's own fiscal year inclusive, ``recent`` is
    that year's own gift amount (0 if none), ``y_response`` is "did this donor
    give the following year", and ``y_amount`` is that following year's gift
    amount when they did (``NaN`` otherwise). The final fiscal year contributes
    no row, because it has no following year to label.

    Also carries the shared as-of feature set every bench in this module
    draws its baseline rules from (E.11a rule 3): ``streak`` (consecutive
    fiscal years given, ending at and including this row's ``fy``),
    ``years_since_last`` (fiscal years since the donor's last gift, 0 if this
    row's own year), ``prev_recent`` (the prior fiscal year's own gift
    amount), ``max_gift`` (largest single-year total so far) and ``tenure``
    (fiscal years since the donor's first-ever gift). Every one of these is
    computed from data at or before this row's own ``fy``.
    """
    panel = make_donor_panel(n_donors=n_donors, n_years=n_years, random_state=seed)
    gifts, donors = panel["gifts"], panel["donors"]
    years = sorted(gifts["fiscal_year"].unique())
    donor_ids = donors["donor_id"].to_numpy()

    amount = (
        gifts.pivot_table(index="donor_id", columns="fiscal_year", values="gift_amount", aggfunc="sum")
        .reindex(index=donor_ids, columns=years, fill_value=0.0)
        .fillna(0.0)
    )
    gave = amount > 0
    first_gift_fy = donors.set_index("donor_id")["first_gift_fy"].reindex(donor_ids).to_numpy()

    rows = []
    cum_total = np.zeros(len(donor_ids))
    cum_n = np.zeros(len(donor_ids))
    streak = np.zeros(len(donor_ids))
    years_since_last = np.full(len(donor_ids), float(len(years)))
    max_gift = np.zeros(len(donor_ids))
    prev_recent = np.zeros(len(donor_ids))
    for i, fy in enumerate(years[:-1]):
        recent = amount[fy].to_numpy()
        gave_fy = gave[fy].to_numpy()
        cum_total = cum_total + recent
        cum_n = cum_n + gave_fy
        max_gift = np.maximum(max_gift, recent)
        streak = np.where(gave_fy, streak + 1, 0.0)
        years_since_last = np.where(gave_fy, 0.0, years_since_last + 1.0)
        tenure = np.maximum(0.0, fy - first_gift_fy)
        next_fy = years[i + 1]
        y_response = gave[next_fy].to_numpy().astype(int)
        y_amount = np.where(y_response == 1, amount[next_fy].to_numpy(), np.nan)
        rows.append(
            pd.DataFrame(
                {
                    "donor_id": donor_ids, "fy": fy,
                    "total": cum_total.copy(), "n": cum_n.copy(), "recent": recent,
                    "streak": streak.copy(), "years_since_last": years_since_last.copy(),
                    "prev_recent": prev_recent.copy(), "max_gift": max_gift.copy(), "tenure": tenure,
                    "y_response": y_response, "y_amount": y_amount,
                }
            )
        )
        prev_recent = recent
    return pd.concat(rows, ignore_index=True)


def _train_test_periods(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_fy = panel["fy"].max()
    return panel[panel["fy"] < test_fy], panel[panel["fy"] == test_fy]


def bench_response(seeds: Sequence[int], n_donors: int, n_years: int) -> List[Row]:
    """DonorPropensityModel / MajorGiftClassifier vs the response rule set
    (E.11a rule 3): lifetime monetary, RFM cell score."""
    seed_rows = []
    for seed in seeds:
        train, test = _train_test_periods(_period_panel(n_donors, n_years, seed))
        Xtr = train[["total", "n", "recent"]].to_numpy()
        Xte = test[["total", "n", "recent"]].to_numpy()
        ytr, yte = train["y_response"].to_numpy(), test["y_response"].to_numpy()
        rules = {
            "lifetime monetary": test["total"].to_numpy(),
            "RFM cell score": _rfm_cell_score(
                test["years_since_last"].to_numpy(), test["n"].to_numpy(), test["total"].to_numpy()
            ),
        }
        rows: List[Row] = []
        for name, cls in (("DonorPropensityModel", DonorPropensityModel), ("MajorGiftClassifier", MajorGiftClassifier)):
            model = cls(random_state=seed).fit(Xtr, ytr)
            proba = model.predict_proba(Xte)[:, 1]
            rows += _classifier_rows("synthetic_panel", name, yte, proba, rules, SYNTHETIC_SPLIT)
        seed_rows.append(rows)
    return _aggregate(seed_rows)


def bench_lapse(seeds: Sequence[int], n_donors: int, n_years: int) -> List[Row]:
    """LapsePredictor vs the lapse rule set (E.11a rule 3): LYBUNT/SYBUNT
    flag, years since last gift, shortest giving streak, gave nothing last
    period."""
    seed_rows = []
    for seed in seeds:
        train, test = _train_test_periods(_period_panel(n_donors, n_years, seed))
        Xtr, Xte = train[["total", "n", "recent"]].to_numpy(), test[["total", "n", "recent"]].to_numpy()
        ytr, yte = 1 - train["y_response"].to_numpy(), 1 - test["y_response"].to_numpy()
        rules = {
            "LYBUNT/SYBUNT flag": ((test["recent"].to_numpy() == 0) & (test["prev_recent"].to_numpy() > 0)).astype(float),
            "years since last gift": test["years_since_last"].to_numpy(),
            "shortest giving streak": -test["streak"].to_numpy(),
            "gave nothing last period": -test["recent"].to_numpy(),
        }
        model = LapsePredictor(random_state=seed).fit(Xtr, ytr)
        score = model.predict_lapse_score(Xte) / 100.0  # predict_lapse_score is 0-100, calibration needs 0-1
        seed_rows.append(
            _classifier_rows("synthetic_panel", "LapsePredictor", yte, score, rules, SYNTHETIC_SPLIT)
        )
    return _aggregate(seed_rows)


def bench_ask(seeds: Sequence[int], n_donors: int, n_years: int) -> List[Row]:
    """AskAmountRecommender vs the ask rule set (E.11a rule 3): last gift,
    max(last gift, average gift), median training gift."""
    seed_rows = []
    for seed in seeds:
        train, test = _train_test_periods(_period_panel(n_donors, n_years, seed))
        train_resp, test_resp = train[train["y_response"] == 1], test[test["y_response"] == 1]
        if len(train_resp) < 20 or len(test_resp) < 5:
            continue
        Xtr = train_resp[["total", "n", "recent"]].to_numpy()
        Xte = test_resp[["total", "n", "recent"]].to_numpy()
        ytr, yte = train_resp["y_amount"].to_numpy(), test_resp["y_amount"].to_numpy()

        model = AskAmountRecommender(random_state=seed).fit(Xtr, ytr)
        pred = model.predict(Xte)
        last_gift = test_resp["recent"].to_numpy()
        avg_gift = (test_resp["total"] / test_resp["n"].replace(0, np.nan)).fillna(pd.Series(last_gift, index=test_resp.index)).to_numpy()
        median_gift = np.full_like(yte, float(np.median(ytr)))
        rules = {
            "last gift": last_gift,
            "max(last gift, average gift)": np.maximum(last_gift, avg_gift),
            "median training gift": median_gift,
        }
        best_name, best_score, _ = _best_ask_baseline(yte, rules)
        header = f"rule_set={best_name} (best of {len(rules)}); split={SYNTHETIC_SPLIT}; n_configs=1"

        seed_rows.append(
            [
                Row(
                    "synthetic_panel", "AskAmountRecommender", "mae",
                    mean_absolute_error(yte, pred), mean_absolute_error(yte, best_score),
                    lower_is_better=True, note=header,
                ),
                Row(
                    "synthetic_panel", "AskAmountRecommender", "within25pct",
                    _within_pct(pred, yte), _within_pct(best_score, yte),
                    note=header,
                ),
            ]
        )
    return _aggregate(seed_rows)


def bench_upgrade(
    seeds: Sequence[int], n_donors: int, n_years: int,
    threshold: float = 1000.0, band: Tuple[float, float] = (100.0, 999.0),
) -> List[Row]:
    """Upgrade model (build_upgrade_snapshots + MajorGiftClassifier) vs the
    upgrade rule set (E.11a rule 3): this-year total, previous-year total
    plus this-year growth projected forward one more year, largest single
    gift in band."""
    seed_rows = []
    for seed in seeds:
        panel = make_donor_panel(n_donors=n_donors, n_years=n_years, random_state=seed)
        years = sorted(panel["gifts"]["fiscal_year"].unique())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            snaps = build_upgrade_snapshots(
                panel["gifts"], fiscal_years=years[:-1], threshold=threshold, band=band,
            )
        if snaps.empty or snaps["fiscal_year"].nunique() < 2:
            continue
        feature_cols = [c for c in snaps.columns if c != "target" and pd.api.types.is_numeric_dtype(snaps[c])]
        X = snaps[feature_cols].to_numpy(dtype="float64")
        y = snaps["target"].to_numpy()
        fy = snaps["fiscal_year"].to_numpy()
        splitter = FiscalYearGroupedSplitter(n_splits=1, drop_repeat_donors=False)
        train_idx, test_idx = list(splitter.split(X, groups=fy))[-1]

        model = MajorGiftClassifier(random_state=seed).fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        rules = {
            "this-year total": snaps["fy_total"].to_numpy()[test_idx],
            "previous-year total plus this-year growth": (snaps["fy_total"] + snaps["fy_trend"]).to_numpy()[test_idx],
            "largest single gift in band": snaps["largest_gift"].to_numpy()[test_idx],
        }
        seed_rows.append(
            _classifier_rows(
                "synthetic_panel", "upgrade_model (MajorGiftClassifier)",
                y[test_idx], proba, rules, "walk-forward (FiscalYearGroupedSplitter, last split)",
            )
        )
    return _aggregate(seed_rows)


def bench_planned_giving(seeds: Sequence[int], n_donors: int, n_years: int) -> List[Row]:
    """PlannedGivingIntentScorer coverage: does it beat chance?

    ``make_donor_panel`` has no bequest-intent label, so this reuses the
    giving-response label as the nearest available binary target. It checks
    that the estimator runs and separates classes on realistic features, not
    a claim about real planned-giving intent.
    """
    seed_rows = []
    for seed in seeds:
        train, test = _train_test_periods(_period_panel(n_donors, n_years, seed))
        Xtr, Xte = train[["total", "n", "recent"]].to_numpy(), test[["total", "n", "recent"]].to_numpy()
        ytr, yte = train["y_response"].to_numpy(), test["y_response"].to_numpy()
        model = PlannedGivingIntentScorer(random_state=seed).fit(Xtr, ytr)
        score = model.predict_intent_score(Xte)
        auc = roc_auc_score(yte, score) if len(np.unique(yte)) > 1 else float("nan")
        seed_rows.append(
            [
                Row(
                    "synthetic_panel", "PlannedGivingIntentScorer", "roc_auc",
                    auc, baseline=0.5,
                    note="baseline=chance; no bequest-intent label exists in make_donor_panel, "
                    "so this reuses the giving-response label as a coverage check only; "
                    "E.11a rule 3's planned-giving rule (age 60+, 10+ years, 5+ gifts) needs an AGE "
                    "field make_donor_panel does not generate, so it is not applied here; "
                    f"split={SYNTHETIC_SPLIT}",
                )
            ]
        )
    return _aggregate(seed_rows)


def bench_gift_interval(seeds: Sequence[int], n_donors: int, n_years: int, alpha: float = 0.1) -> List[Row]:
    """GiftIntervalCalibrator coverage: does the attained level match the request?"""
    seed_rows = []
    for seed in seeds:
        train, test = _train_test_periods(_period_panel(n_donors, n_years, seed))
        train_resp, test_resp = train[train["y_response"] == 1], test[test["y_response"] == 1]
        if len(train_resp) < 60 or len(test_resp) < 5:
            continue
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(train_resp))
        half = len(order) // 2
        fit_rows, cal_rows = train_resp.iloc[order[:half]], train_resp.iloc[order[half:]]

        Xfit = fit_rows[["total", "n", "recent"]].to_numpy()
        Xcal = cal_rows[["total", "n", "recent"]].to_numpy()
        Xte = test_resp[["total", "n", "recent"]].to_numpy()
        yte = test_resp["y_amount"].to_numpy()

        ask = AskAmountRecommender(random_state=seed).fit(Xfit, fit_rows["y_amount"].to_numpy())
        cal = GiftIntervalCalibrator(ask, alpha=alpha).fit(Xcal, cal_rows["y_amount"].to_numpy())
        interval = cal.predict_gift_interval(Xte)
        attained = float(((yte >= interval.lower) & (yte <= interval.upper)).mean())

        seed_rows.append(
            [
                Row(
                    "synthetic_panel", "GiftIntervalCalibrator",
                    f"empirical_coverage(target={1 - alpha:.2f})",
                    attained, coverage_target=1 - alpha,
                    note=f"n_test={len(yte)}, n_calibration={len(cal_rows)}",
                )
            ]
        )
    return _aggregate(seed_rows)


# --------------------------------------------------------------------------- #
# KDD Cup 1998 (opt-in download; see fetch_kdd98_donors)
# --------------------------------------------------------------------------- #
def _kdd_gift_log(donors: pd.DataFrame) -> pd.DataFrame:
    """Reshape KDD98's wide promotion history into a gift log.

    Exactly the recipe in examples/notebooks/04_kdd98_end_to_end.ipynb:
    RDATE_i/RAMNT_i for i=3..24 (i=2 is the held-out 97NK mailing scored by
    TARGET_B/TARGET_D and has no RDATE_2/RAMNT_2).
    """
    parts = []
    for i in range(3, 25):
        promo = donors[["CONTROLN", f"RDATE_{i}", f"RAMNT_{i}"]].dropna()
        promo.columns = ["donor_id", "yymm", "gift_amount"]
        parts.append(promo)
    gifts = pd.concat(parts, ignore_index=True)
    gifts["gift_date"] = pd.to_datetime(
        "19" + gifts["yymm"].astype(int).astype(str).str.zfill(4), format="%Y%m"
    )
    gifts = gifts[["donor_id", "gift_date", "gift_amount"]]
    gifts = (
        CRMCleaner(date_col="gift_date", amount_col="gift_amount")
        .set_output(transform="pandas")
        .fit_transform(gifts)
        .astype({"donor_id": int})
    )
    return gifts


def _kdd_rfm(gifts: pd.DataFrame) -> pd.DataFrame:
    return (
        RFMTransformer(as_of=KDD_AS_OF, include_tenure=True)
        .fit_transform(gifts[["donor_id", "gift_date", "gift_amount"]])
        .set_index("donor_id")
    )


def _split_55_15_30(n: int, y_strat: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """E.11a rule 1: KDD98 uses a 55/15/30 stratified split, not 70/30.
    First splits off the 30% test set, then splits the remaining 70% into
    55% train / 15% validation. Returns positional index arrays."""
    idx = np.arange(n)
    idx_trainval, idx_test = train_test_split(idx, test_size=0.30, random_state=seed, stratify=y_strat)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=15 / 70, random_state=seed, stratify=y_strat[idx_trainval]
    )
    return idx_train, idx_val, idx_test


_KDD_BASE_COLS = [
    "AGE", "INCOME", "WEALTH1", "WEALTH2", "NUMCHLD", "RAMNTALL", "NGIFTALL",
    "LASTGIFT", "AVGGIFT", "MAXRAMNT", "MINRAMNT", "TIMELAG",
]


def _kdd_feature_frame(donors: pd.DataFrame, rfm: pd.DataFrame) -> pd.DataFrame:
    """``_KDD_BASE_COLS`` plus the RFM features, indexed by ``CONTROLN``.
    Shared by ``_kdd_ask_design`` (the LRN file, then split 55/15/30) and
    ``bench_kdd_cost_aware_val`` (the VAL file, scored whole, never split)."""
    donors_idx = donors.set_index("CONTROLN")
    return donors_idx[_KDD_BASE_COLS].join(rfm.add_prefix("rfm_")).assign(
        HOMEOWNER=(donors_idx["HOMEOWNR"] == "H").astype(int)
    )


def _kdd_ask_design(
    donors: pd.DataFrame, rfm: pd.DataFrame, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Returns (train, val, test) X frames and (train, val, test) y_amount
    series, 55/15/30 stratified on response (E.11a rule 1). ``val`` is not
    used to pick anything in PR2 (no hyperparameter search yet); it is
    reserved for later PRs' loss/config choices so every PR shares one split."""
    X_full = _kdd_feature_frame(donors, rfm)
    donors_idx = donors.set_index("CONTROLN")
    y_amt = donors_idx["TARGET_D"]
    y_resp = donors_idx["TARGET_B"].to_numpy()
    idx_train, idx_val, idx_test = _split_55_15_30(len(X_full), y_resp, seed)
    return (
        X_full.iloc[idx_train], X_full.iloc[idx_val], X_full.iloc[idx_test],
        y_amt.iloc[idx_train], y_amt.iloc[idx_val], y_amt.iloc[idx_test],
    )


def bench_kdd_upgrade(seed: int, threshold: float = 50.0, band: Tuple[float, float] = (5.0, 49.0)) -> List[Row]:
    """Upgrade model on KDD98, threshold rescaled from the $1000/$100-999
    defaults: this file's per-donor annual giving tops out far lower than a
    major-gift program's, so $50/$5-49 keeps the same shape (threshold is
    roughly the 93rd percentile of per-donor annual giving on this file).
    Split is 55/15/30 stratified (E.11a rule 1: KDD98 is not on the
    walk-forward list), even though the snapshot table has one row per
    qualifying (donor, fiscal year): the same donor can land in more than one
    split across different years, a known limitation of applying the
    dataset's blanket split rule to a multi-year snapshot table."""
    donors = fetch_kdd98_donors()
    gifts = _kdd_gift_log(donors)
    fiscal = (
        FiscalYearTransformer(date_col="gift_date", fiscal_year_start=7)
        .set_output(transform="pandas")
        .fit_transform(gifts)
    )
    years = sorted(fiscal["fiscal_year"].astype(int).unique())
    max_date = gifts["gift_date"].max()
    # Only years fully resolved by max_date (T and T+1 both complete), same rule
    # score_upgrade_prospects uses internally: a handful of mis-dated RDATE_i
    # entries in this real file otherwise leave a near-empty trailing stub
    # fiscal year with too few (or zero) positive targets to evaluate on.
    fiscal_years = [t for t in years if _fy_end(t + 1, 7) <= max_date]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        snaps = build_upgrade_snapshots(
            gifts, fiscal_years=fiscal_years, threshold=threshold, band=band, fiscal_year_start=7,
        )
    feature_cols = [c for c in snaps.columns if c != "target" and pd.api.types.is_numeric_dtype(snaps[c])]
    X = snaps[feature_cols].to_numpy(dtype="float64")
    y = snaps["target"].to_numpy()
    idx_train, _idx_val, idx_test = _split_55_15_30(len(y), y, seed)

    model = MajorGiftClassifier(random_state=seed).fit(X[idx_train], y[idx_train])
    proba = model.predict_proba(X[idx_test])[:, 1]
    rules = {
        "this-year total": snaps["fy_total"].to_numpy()[idx_test],
        "previous-year total plus this-year growth": (snaps["fy_total"] + snaps["fy_trend"]).to_numpy()[idx_test],
        "largest single gift in band": snaps["largest_gift"].to_numpy()[idx_test],
    }
    return _classifier_rows(
        "kdd98", "upgrade_model (MajorGiftClassifier)", y[idx_test], proba, rules, KDD_SPLIT, bootstrap=True,
    )


def bench_kdd_response(seed: int) -> List[Row]:
    """DonorPropensityModel / MajorGiftClassifier vs the response rule set
    (E.11a rule 3): lifetime monetary, RFM cell score, and (KDD98 only) RFA_2
    frequency then last gift. ``RFA_2`` is safe to use here: per
    ``cup98dic.txt`` it is "donor's RFA status as of 97NK promotion date",
    i.e. computed when the 97NK mailing was sent and not updated by its
    response (``TARGET_B``/``TARGET_D``), the same "as of, not updated by"
    check rule 5 asks for. ``HIT`` and ``LASTDATE`` are not used by this
    script."""
    donors = fetch_kdd98_donors()
    rfm = _kdd_rfm(_kdd_gift_log(donors))
    donors_idx = donors.set_index("CONTROLN")
    X_rfm = rfm.reindex(donors_idx.index)
    X_rfm["recency"] = X_rfm["recency"].fillna(X_rfm["recency"].max() + 1)
    X_rfm[["frequency", "monetary", "tenure"]] = X_rfm[["frequency", "monetary", "tenure"]].fillna(0.0)
    y = donors_idx["TARGET_B"].to_numpy()

    idx_train, _idx_val, idx_test = _split_55_15_30(len(y), y, seed)
    Xtr, Xte, ytr, yte = X_rfm.iloc[idx_train], X_rfm.iloc[idx_test], y[idx_train], y[idx_test]
    rfa_2f_rank = donors_idx["RFA_2F"].map({"1": 1, "2": 2, "5": 3}).fillna(0).to_numpy()[idx_test]
    lastgift = donors_idx["LASTGIFT"].to_numpy()[idx_test]
    rules = {
        "lifetime monetary": Xte["monetary"].to_numpy(),
        "RFM cell score": _rfm_cell_score(Xte["recency"].to_numpy(), Xte["frequency"].to_numpy(), Xte["monetary"].to_numpy()),
        "RFA_2 frequency then last gift": rfa_2f_rank * 1e6 + lastgift,
    }
    rows: List[Row] = []
    for name, cls in (("DonorPropensityModel", DonorPropensityModel), ("MajorGiftClassifier", MajorGiftClassifier)):
        model = cls(random_state=seed).fit(Xtr.to_numpy(), ytr)
        proba = model.predict_proba(Xte.to_numpy())[:, 1]
        rows += _classifier_rows("kdd98", name, yte, proba, rules, KDD_SPLIT, bootstrap=True)
    return rows


def bench_kdd_lapse(seed: int) -> List[Row]:
    """LapsePredictor vs the lapse rule set (E.11a rule 3), over the
    22-promotion history: LYBUNT/SYBUNT flag, years since last gift,
    shortest giving streak, gave nothing last period.

    Split: the 23 promotion periods are naturally time-ordered, so this
    walks forward across them (train on periods < N-1, validate on N-1, test
    on N, the same shape as the synthetic panel split) rather than the
    dataset-level 55/15/30 stratified split (E.11a rule 1): mixing periods
    with a donor-level stratified split would let a donor's later promotion
    history leak into training for an earlier one."""
    donors = fetch_kdd98_donors()
    hist_promos = list(range(24, 2, -1))
    ramnt = donors[[f"RAMNT_{i}" for i in hist_promos]].fillna(0.0).to_numpy()
    gave = ramnt > 0
    n_periods = ramnt.shape[1]
    target_b = donors["TARGET_B"].to_numpy()

    cum_total, cum_n = np.zeros(len(donors)), np.zeros(len(donors))
    streak, years_since_last, prev_recent = (
        np.zeros(len(donors)), np.full(len(donors), float(n_periods)), np.zeros(len(donors)),
    )
    periods = []
    for p in range(n_periods):
        recent = ramnt[:, p]
        cum_total, cum_n = cum_total + recent, cum_n + gave[:, p]
        next_gave = gave[:, p + 1] if p < n_periods - 1 else (target_b == 1)
        periods.append(
            pd.DataFrame(
                dict(
                    period=p, total=cum_total.copy(), n=cum_n.copy(), recent=recent,
                    streak=streak.copy(), years_since_last=years_since_last.copy(), prev_recent=prev_recent.copy(),
                    lapsed=(~next_gave).astype(int),
                )
            )
        )
        streak = np.where(gave[:, p], streak + 1, 0.0)
        years_since_last = np.where(gave[:, p], 0.0, years_since_last + 1.0)
        prev_recent = recent
    panel = pd.concat(periods, ignore_index=True)
    last_period = panel["period"].max()
    val_period = last_period - 1
    train_p = panel[panel["period"] < val_period]
    test_p = panel[panel["period"] == last_period]

    model = LapsePredictor(n_estimators=100, max_depth=10, random_state=seed).fit(
        train_p[["total", "n", "recent"]].to_numpy(), train_p["lapsed"].to_numpy()
    )
    score = model.predict_lapse_score(test_p[["total", "n", "recent"]].to_numpy()) / 100.0
    y = test_p["lapsed"].to_numpy()
    rules = {
        "LYBUNT/SYBUNT flag": ((test_p["recent"].to_numpy() == 0) & (test_p["prev_recent"].to_numpy() > 0)).astype(float),
        "years since last gift": test_p["years_since_last"].to_numpy(),
        "shortest giving streak": -test_p["streak"].to_numpy(),
        "gave nothing last period": -test_p["recent"].to_numpy(),
    }
    return _classifier_rows(
        "kdd98", "LapsePredictor", y, score, rules,
        "walk-forward across KDD98 promotion periods (train < period N-1, val=N-1, test=N)",
        bootstrap=True,
    )


def bench_kdd_ask(seed: int) -> List[Row]:
    """AskAmountRecommender vs the ask rule set (E.11a rule 3): last gift,
    max(last gift, average gift), median training gift."""
    donors = fetch_kdd98_donors()
    rfm = _kdd_rfm(_kdd_gift_log(donors))
    Xd_train, _Xd_val, Xd_test, yd_train, _yd_val, yd_test = _kdd_ask_design(donors, rfm, seed)
    responders_train, responders_test = yd_train > 0, yd_test > 0

    ask_model = make_pipeline(
        WealthScreeningImputer(wealth_cols=["WEALTH1", "WEALTH2", "INCOME"]),
        AskAmountRecommender(random_state=seed),
    ).fit(Xd_train[responders_train], yd_train[responders_train])
    pred = ask_model.predict(Xd_test[responders_test])

    y_true = yd_test[responders_test].to_numpy()
    last_gift = Xd_test.loc[responders_test, "LASTGIFT"].to_numpy()
    avg_gift = Xd_test.loc[responders_test, "AVGGIFT"].to_numpy()
    median_gift = np.full_like(y_true, float(np.median(yd_train[responders_train])))
    rules = {
        "last gift": last_gift,
        "max(last gift, average gift)": np.maximum(last_gift, avg_gift),
        "median training gift": median_gift,
    }
    best_name, best_score, _ = _best_ask_baseline(y_true, rules)
    header = f"rule_set={best_name} (best of {len(rules)}); split={KDD_SPLIT}; n_configs=1"

    return [
        Row(
            "kdd98", "AskAmountRecommender", "mae",
            mean_absolute_error(y_true, pred), mean_absolute_error(y_true, best_score),
            lower_is_better=True, note=header,
        ),
        Row(
            "kdd98", "AskAmountRecommender", "within25pct",
            _within_pct(pred, y_true), _within_pct(best_score, y_true),
            note=header,
        ),
    ]


def bench_kdd_cost_aware(seed: int, cost: float = 0.68) -> List[Row]:
    """Mail if E[gift] > cost (the KDD Cup 1998 competition's own rule) vs mailing everyone."""
    donors = fetch_kdd98_donors()
    rfm = _kdd_rfm(_kdd_gift_log(donors))
    Xd_train, _Xd_val, Xd_test, yd_train, _yd_val, yd_test = _kdd_ask_design(donors, rfm, seed)
    responders_train = yd_train > 0

    resp_model = make_pipeline(
        WealthScreeningImputer(wealth_cols=["WEALTH1", "WEALTH2", "INCOME"]),
        MajorGiftClassifier(random_state=seed),
    ).fit(Xd_train, (yd_train > 0).astype(int))
    p_respond = resp_model.predict_proba(Xd_test)[:, 1]

    ask_model = make_pipeline(
        WealthScreeningImputer(wealth_cols=["WEALTH1", "WEALTH2", "INCOME"]),
        # Expected gift needs the conditional mean, not the ask default's median.
        AskAmountRecommender(loss="squared_error", random_state=seed),
    ).fit(Xd_train[responders_train], yd_train[responders_train])
    expected_gift = p_respond * ask_model.predict(Xd_test)
    mail = expected_gift > cost

    raised_all, cost_all = float(yd_test.sum()), cost * len(Xd_test)
    raised_mail, cost_mail = float(yd_test[mail].sum()), cost * int(mail.sum())
    roi_all = fundraising_roi(total_raised=raised_all, total_fundraising_expense=cost_all)
    roi_mail = fundraising_roi(total_raised=raised_mail, total_fundraising_expense=cost_mail)

    return [
        Row(
            "kdd98", "cost_aware_selection", "net_revenue",
            raised_mail - cost_mail, raised_all - cost_all,
            note=f"mail if E[gift]>${cost:.2f}; pieces={int(mail.sum())}/{len(Xd_test)}",
        ),
        Row(
            "kdd98", "cost_aware_selection", "roi",
            roi_mail, roi_all, note="baseline=mail everyone",
        ),
    ]


def bench_kdd_cost_aware_val(seed: int, cost: float = 0.68) -> List[Row]:
    """Cost-aware selection, scored on KDD98's own held-out
    validation file instead of a random split of the learning file.

    Fits the same response model (``MajorGiftClassifier``) and ask model
    (``AskAmountRecommender``) as ``bench_kdd_cost_aware``, on the LRN file's
    own 55% train split (``_kdd_ask_design``'s ``Xd_train``/``yd_train``),
    then scores "mail if E[gift] > cost" on ``cup98VAL`` + ``valtargt``
    (:func:`~philanthropy.datasets.fetch_kdd98_val_donors`): 96,367 donors
    that were never part of the learning file and never touched by any split
    of it. Reported *next to* ``bench_kdd_cost_aware``'s random-split number
    (dataset ``"cup98val"`` vs ``"kdd98"``), not replacing it."""
    donors = fetch_kdd98_donors()
    rfm = _kdd_rfm(_kdd_gift_log(donors))
    Xd_train, _Xd_val, _Xd_test, yd_train, _yd_val, _yd_test = _kdd_ask_design(donors, rfm, seed)
    responders_train = yd_train > 0

    resp_model = make_pipeline(
        WealthScreeningImputer(wealth_cols=["WEALTH1", "WEALTH2", "INCOME"]),
        MajorGiftClassifier(random_state=seed),
    ).fit(Xd_train, (yd_train > 0).astype(int))

    ask_model = make_pipeline(
        WealthScreeningImputer(wealth_cols=["WEALTH1", "WEALTH2", "INCOME"]),
        # Same pin as bench_kdd_cost_aware: expected gift needs the conditional mean.
        AskAmountRecommender(loss="squared_error", random_state=seed),
    ).fit(Xd_train[responders_train], yd_train[responders_train])

    val_donors = fetch_kdd98_val_donors()
    val_rfm = _kdd_rfm(_kdd_gift_log(val_donors))
    X_val = _kdd_feature_frame(val_donors, val_rfm)
    y_val = val_donors.set_index("CONTROLN")["TARGET_D"]

    p_respond = resp_model.predict_proba(X_val)[:, 1]
    expected_gift = p_respond * ask_model.predict(X_val)
    mail = expected_gift > cost

    raised_all, cost_all = float(y_val.sum()), cost * len(X_val)
    raised_mail, cost_mail = float(y_val[mail].sum()), cost * int(mail.sum())
    roi_all = fundraising_roi(total_raised=raised_all, total_fundraising_expense=cost_all)
    roi_mail = fundraising_roi(total_raised=raised_mail, total_fundraising_expense=cost_mail)

    return [
        Row(
            "cup98val", "cost_aware_selection", "net_revenue",
            raised_mail - cost_mail, raised_all - cost_all,
            note=(
                f"mail if E[gift]>${cost:.2f}; pieces={int(mail.sum())}/{len(X_val)}; "
                "model fit on cup98LRN's own 55% train split only, scored on KDD98's "
                "own held-out validation file (cup98VAL+valtargt), reported next to "
                "bench_kdd_cost_aware's random-split number, not replacing it"
            ),
        ),
        Row(
            "cup98val", "cost_aware_selection", "roi",
            roi_mail, roi_all, note="baseline=mail everyone; KDD98's own held-out validation file",
        ),
    ]


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _print_table(rows: List[Row]) -> None:
    header = f"{'dataset':<16} {'model':<33} {'metric':<33} {'value':>9} {'baseline':>9}  {'verdict':<7}note"
    print(header)
    print("-" * len(header))
    for r in rows:
        baseline_s = f"{r.baseline:.4g}" if r.baseline is not None else "-"
        range_s = f" [{r.lo:.4g}-{r.hi:.4g}, n={r.n_seeds}]" if r.n_seeds > 1 else ""
        print(
            f"{r.dataset:<16} {r.model:<33} {r.metric:<33} {r.value:>9.4g} {baseline_s:>9}  "
            f"{r.verdict:<7}{r.note}{range_s}"
        )


def _write_json(rows: List[Row], path: str) -> None:
    with open(path, "w") as fh:
        json.dump([r.as_dict() for r in rows], fh, indent=2)


def _write_csv(rows: List[Row], path: str) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.as_dict())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--skip-kdd98", action="store_true", help="Skip the KDD Cup 1998 section entirely (no download).")
    parser.add_argument(
        "--with-cup98val", action="store_true",
        help="Also score cost-aware selection on KDD98's own held-out cup98VAL+valtargt "
        "file (a second ~37 MB download; opt-in). No effect with --skip-kdd98.",
    )
    parser.add_argument("--fast", action="store_true", help="One seed and a small synthetic panel: a smoke run, not a benchmark.")
    parser.add_argument("--out", type=str, default=None, help="Path prefix; also writes <out>.json and <out>.csv.")
    args = parser.parse_args()

    seeds: Sequence[int] = (DEFAULT_SEEDS[0],) if args.fast else DEFAULT_SEEDS
    n_donors = 300 if args.fast else 3000
    n_years = 5 if args.fast else 7

    start = time.time()
    rows: List[Row] = []
    rows += bench_response(seeds, n_donors, n_years)
    rows += bench_lapse(seeds, n_donors, n_years)
    rows += bench_ask(seeds, n_donors, n_years)
    rows += bench_upgrade(seeds, n_donors, n_years)
    rows += bench_planned_giving(seeds, n_donors, n_years)
    rows += bench_gift_interval(seeds, n_donors, n_years)

    if not args.skip_kdd98:
        rows += bench_kdd_upgrade(KDD_SEED)
        rows += bench_kdd_response(KDD_SEED)
        rows += bench_kdd_lapse(KDD_SEED)
        rows += bench_kdd_ask(KDD_SEED)
        rows += bench_kdd_cost_aware(KDD_SEED)
        if args.with_cup98val:
            rows += bench_kdd_cost_aware_val(KDD_SEED)

    runtime = time.time() - start
    _print_table(rows)
    print(f"\nRuntime: {runtime:.1f}s")

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        _write_json(rows, args.out + ".json")
        _write_csv(rows, args.out + ".csv")
        print(f"Wrote {args.out}.json and {args.out}.csv")


if __name__ == "__main__":
    main()
