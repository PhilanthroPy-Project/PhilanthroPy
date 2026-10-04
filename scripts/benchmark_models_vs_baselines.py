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
    python scripts/benchmark_models_vs_baselines.py --with-blood         # also UCI Blood Transfusion (opt-in, ~12KB)
    python scripts/benchmark_models_vs_baselines.py --karlan-list-path ~/data/karlan_list/AERtables1-5.dta

The KDD Cup 1998 section downloads ``cup98lrn.zip`` (~36 MB) to
``~/philanthropy_data`` on first use (see ``fetch_kdd98_donors``); pass
``--skip-kdd98`` to stay offline entirely. ``--with-cup98val`` additionally
scores cost-aware selection, response, lapse and ask on KDD98's own held-out
validation file (``cup98VAL.zip`` + ``valtargt.txt``, another ~37 MB, see
``fetch_kdd98_val_donors``), reported next to the learning-file numbers
rather than replacing them; off by default and has no effect with
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

A verdict is never read off one point. Each row with a baseline carries an
interval on the paired difference ``model - baseline``, measured on the same
donors: on a single split, a 1,000-resample bootstrap of that difference
(2.5th to 97.5th percentile); across seeds or walk-forward folds, the
smallest and largest per-seed difference, so the worst seed decides. The
model ``"wins"`` if the whole interval is on its side of zero (above zero
where higher is better: hit rate, ROC-AUC, share within 25%, net revenue;
below zero where lower is better: MAE), ``"loses"`` if the whole interval is
on the rule's side, and is ``"modest"`` (about the same) if it straddles
zero. A row with no baseline (average precision, the calibration gap) or no
interval uses ``"n/a"``. The two coverage checks use a coverage-specific
rule instead: attained coverage within 3 points of the requested level is
``"wins"``, within 6 points is ``"modest"``, more than 6 points short is
``"loses"``.

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
import urllib.request
import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from philanthropy.datasets import (
    fetch_kdd98_donors,
    fetch_kdd98_val_donors,
    load_donorschoose,
    load_karlan_list,
    load_psid_philanthropy,
    make_donor_panel,
)
from philanthropy.ingest import build_leadership_snapshots, build_snapshots
from philanthropy.ingest._snapshots import period_snapshots
from philanthropy.ingest._upgrade_snapshots import _column, _prepare_gifts, _snapshot_features_for_year
from philanthropy.experimental import UpliftTLearner
from philanthropy.metrics import fundraising_roi
from philanthropy.model_selection import FiscalYearGroupedSplitter
from philanthropy.utils import trailing_slope_features
from philanthropy.models import (
    AskAmountRecommender,
    DonorPropensityModel,
    FinancialForecastModel,
    GiftIntervalCalibrator,
    LapsePredictor,
    MajorGiftClassifier,
    PlannedGivingIntentScorer,
)
from philanthropy.preprocessing import (
    CRMCleaner,
    RFMTransformer,
    WealthScreeningImputer,
)

DEFAULT_SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46)
GIFT_INTERVAL_SEEDS: Tuple[int, ...] = tuple(range(42, 52))
PARITY_FEATURES: Tuple[str, ...] = (
    "total", "n", "recent", "streak", "years_since_last", "prev_recent", "max_gift", "tenure",
)
KDD_SEED = 42
MOMENTUM_COLS: Tuple[str, ...] = tuple(
    f"fy_total_{stat}_{k}y" for stat in ("slope", "rel_slope") for k in (3, 5)
)
KDD_AS_OF = "1997-06-01"
COVERAGE_TOL = 0.03
N_BOOTSTRAP = 1000
BOOTSTRAP_METRICS = ("top10pct_hit_rate", "roc_auc")
KDD_SPLIT = "55/15/30 stratified (train/validation/test)"
SYNTHETIC_SPLIT = "walk-forward (train on fiscal years < T, test on T)"
DONORSCHOOSE_SEED = 42
DONORSCHOOSE_SUBSAMPLE = 0.10
DONORSCHOOSE_N_FOLDS = 4
PSID_SEED = 42
PSID_N_FOLDS = 4

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


def _bootstrap_diff_ci(n: int, diff_fn, n_resamples: int = N_BOOTSTRAP, seed: int = 0) -> Tuple[float, float]:
    """Bootstrap 95% interval on a paired difference: ``diff_fn(idx)``
    returns ``model metric - rule metric`` on the resampled rows ``idx``, so
    both sides see the same donors in every resample. ``nan`` resamples
    (e.g. a single-class draw for ROC-AUC) are skipped."""
    rng = np.random.default_rng(seed)
    values = [diff_fn(rng.integers(0, n, size=n)) for _ in range(n_resamples)]
    values = [v for v in values if v == v]
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


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
    diff_lo: Optional[float] = None
    diff_hi: Optional[float] = None
    baseline_lo: Optional[float] = None
    baseline_hi: Optional[float] = None

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
        if self.diff_lo is None or self.diff_hi is None or self.diff_lo != self.diff_lo:
            return "n/a"
        lo, hi = (-self.diff_hi, -self.diff_lo) if self.lower_is_better else (self.diff_lo, self.diff_hi)
        if lo > 0:
            return "wins"
        if hi < 0:
            return "loses"
        return "modest"

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
            "diff_lo": self.diff_lo,
            "diff_hi": self.diff_hi,
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
        # Worst seed decides (E.11f rule 5): the interval on model - rule is
        # the smallest and largest per-seed paired difference.
        diffs = [
            r.value - r.baseline for r in rs
            if r.value == r.value and r.baseline is not None and r.baseline == r.baseline
        ]
        if len(rs) == 1:
            # One split is a point estimate, not a range: keep that row's own
            # bootstrap interval, or none (so its verdict is "n/a").
            diffs = [d for d in (rs[0].diff_lo, rs[0].diff_hi) if d is not None]
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
                diff_lo=float(min(diffs)) if diffs else None,
                diff_hi=float(max(diffs)) if diffs else None,
                baseline_lo=float(min(baselines)) if baselines else None,
                baseline_hi=float(max(baselines)) if baselines else None,
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
    top10pct_hit_rate and roc_auc, and a paired-difference interval
    (model - rule on the same resample) on every top-N hit rate and ROC-AUC,
    which is what the verdict reads; that is for single-split rows only.
    Walk-forward and multi-seed rows get their difference interval from
    :func:`_aggregate` instead."""
    y_true, score = np.asarray(y_true), np.asarray(score)
    best_name, baseline_score, _ = _best_classifier_baseline(y_true, rules, lambda y, s: _topn_rate(y, s, 0.10)[0])
    header = f"rule_set={best_name} (best of {len(rules)}); split={split}; n_configs={n_configs}"
    rows = []
    for frac in (0.01, 0.05, 0.10):
        m_rate, n = _topn_rate(y_true, score, frac)
        b_rate, _ = _topn_rate(y_true, baseline_score, frac)
        metric = f"top{int(frac * 100)}pct_hit_rate"
        lo = hi = d_lo = d_hi = None
        if bootstrap and metric in BOOTSTRAP_METRICS:
            lo, hi = _bootstrap_ci(y_true, score, lambda y, s: _topn_rate(y, s, frac)[0])
        if bootstrap:
            d_lo, d_hi = _bootstrap_diff_ci(len(y_true), lambda idx, f=frac: (
                _topn_rate(y_true[idx], score[idx], f)[0] - _topn_rate(y_true[idx], baseline_score[idx], f)[0]
            ))
        rows.append(Row(
            dataset, model_name, metric, m_rate, b_rate, note=f"n={n}; {header}", lo=lo, hi=hi,
            diff_lo=d_lo, diff_hi=d_hi,
        ))
    has_both_classes = len(np.unique(y_true)) > 1
    auc = roc_auc_score(y_true, score) if has_both_classes else float("nan")
    b_auc = roc_auc_score(y_true, baseline_score) if has_both_classes else float("nan")
    auc_lo = auc_hi = auc_d_lo = auc_d_hi = None
    if bootstrap and has_both_classes:
        auc_lo, auc_hi = _bootstrap_ci(y_true, score, roc_auc_score)

        def _auc_diff(idx: np.ndarray) -> float:
            if len(np.unique(y_true[idx])) < 2:
                return float("nan")
            return roc_auc_score(y_true[idx], score[idx]) - roc_auc_score(y_true[idx], baseline_score[idx])

        auc_d_lo, auc_d_hi = _bootstrap_diff_ci(len(y_true), _auc_diff)
    rows.append(Row(
        dataset, model_name, "roc_auc", auc, b_auc, note=header, lo=auc_lo, hi=auc_hi,
        diff_lo=auc_d_lo, diff_hi=auc_d_hi,
    ))
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


def _revenue_top_share(y_true: np.ndarray, score: np.ndarray, frac: float = 0.10) -> float:
    """Share of all next-gift dollars that comes from the top ``frac`` of
    donors ranked by ``score``: the revenue a shop captures if it can only
    work that slice of the list. Ties keep file order."""
    y_true = np.asarray(y_true, dtype="float64")
    total = y_true.sum()
    if total <= 0:
        return float("nan")
    k = max(1, int(round(len(y_true) * frac)))
    top = np.argsort(-np.asarray(score), kind="stable")[:k]
    return float(y_true[top].sum() / total)


# --------------------------------------------------------------------------- #
# Synthetic donor panel (make_donor_panel)
# --------------------------------------------------------------------------- #
def _period_panel(n_donors: int, n_years: int, seed: int, include_momentum: bool = False) -> pd.DataFrame:
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

    if include_momentum:
        # Ad hoc scratch reuse of the shared helper: treat each fiscal year
        # as a Dec-31 year-end bin so trailing_slope_features's 12-month
        # bins line up with this panel's integer fiscal years.
        long = amount.reset_index().melt(id_vars="donor_id", var_name="fy", value_name="gift_amount")
        long["gift_date"] = pd.to_datetime(long["fy"].astype(int).astype(str) + "-12-31")
        data_start = pd.Timestamp(f"{years[0]}-12-31")

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
        row = {
            "donor_id": donor_ids, "fy": fy,
            "total": cum_total.copy(), "n": cum_n.copy(), "recent": recent,
            "streak": streak.copy(), "years_since_last": years_since_last.copy(),
            "prev_recent": prev_recent.copy(), "max_gift": max_gift.copy(), "tenure": tenure,
            "y_response": y_response, "y_amount": y_amount,
        }
        if include_momentum:
            cutoff = pd.Timestamp(f"{fy}-12-31")
            momentum = trailing_slope_features(
                long, pd.Index(donor_ids), cutoff, date_col="gift_date", value_col="gift_amount",
                agg="sum", prefix="fy_total", donor_col="donor_id", data_start=data_start,
            ).reindex(donor_ids)
            for col in momentum.columns:
                row[col] = momentum[col].to_numpy()
        rows.append(pd.DataFrame(row))
        prev_recent = recent
    return pd.concat(rows, ignore_index=True)


def _train_test_periods(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_fy = panel["fy"].max()
    return panel[panel["fy"] < test_fy], panel[panel["fy"] == test_fy]


def bench_response(
    seeds: Sequence[int], n_donors: int, n_years: int,
    feature_cols: Sequence[str] = ("total", "n", "recent"), dataset: str = "synthetic_panel",
    include_momentum: bool = False,
) -> List[Row]:
    """DonorPropensityModel / MajorGiftClassifier vs the response rule set
    (E.11a rule 3): lifetime monetary, RFM cell score.

    ``feature_cols`` defaults to the 3 features the rules are not fed
    (E.12b's "feature parity probe"); pass :data:`PARITY_FEATURES` for the
    same comparison on the full 8 as-of columns."""
    if include_momentum:
        feature_cols = tuple(feature_cols) + MOMENTUM_COLS
    seed_rows = []
    for seed in seeds:
        train, test = _train_test_periods(_period_panel(n_donors, n_years, seed, include_momentum=include_momentum))
        Xtr = train[list(feature_cols)].fillna(0.0).to_numpy()
        Xte = test[list(feature_cols)].fillna(0.0).to_numpy()
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
            rows += _classifier_rows(dataset, name, yte, proba, rules, SYNTHETIC_SPLIT)
        seed_rows.append(rows)
    return _aggregate(seed_rows)


def bench_lapse(
    seeds: Sequence[int], n_donors: int, n_years: int,
    feature_cols: Sequence[str] = ("total", "n", "recent"), dataset: str = "synthetic_panel",
    include_momentum: bool = False,
) -> List[Row]:
    """LapsePredictor vs the lapse rule set (E.11a rule 3): LYBUNT/SYBUNT
    flag, years since last gift, shortest giving streak, gave nothing last
    period.

    ``feature_cols`` defaults to the 3 features the rules are not fed
    (E.12b's "feature parity probe"); pass :data:`PARITY_FEATURES` for the
    same comparison on the full 8 as-of columns."""
    if include_momentum:
        feature_cols = tuple(feature_cols) + MOMENTUM_COLS
    seed_rows = []
    for seed in seeds:
        train, test = _train_test_periods(_period_panel(n_donors, n_years, seed, include_momentum=include_momentum))
        Xtr = train[list(feature_cols)].fillna(0.0).to_numpy()
        Xte = test[list(feature_cols)].fillna(0.0).to_numpy()
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
            _classifier_rows(dataset, "LapsePredictor", yte, score, rules, SYNTHETIC_SPLIT)
        )
    return _aggregate(seed_rows)


def bench_ask(seeds: Sequence[int], n_donors: int, n_years: int, include_momentum: bool = False) -> List[Row]:
    """AskAmountRecommender vs the ask rule set (E.11a rule 3): last gift,
    max(last gift, average gift), median training gift."""
    base_cols = ("total", "n", "recent")
    feature_cols = base_cols + MOMENTUM_COLS if include_momentum else base_cols
    seed_rows = []
    for seed in seeds:
        train, test = _train_test_periods(_period_panel(n_donors, n_years, seed, include_momentum=include_momentum))
        train_resp, test_resp = train[train["y_response"] == 1], test[test["y_response"] == 1]
        if len(train_resp) < 20 or len(test_resp) < 5:
            continue
        Xtr = train_resp[list(feature_cols)].fillna(0.0).to_numpy()
        Xte = test_resp[list(feature_cols)].fillna(0.0).to_numpy()
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
                Row(
                    "synthetic_panel", "AskAmountRecommender", "revenue_top10pct",
                    _revenue_top_share(yte, pred), _revenue_top_share(yte, best_score),
                    note=header,
                ),
            ]
        )
    return _aggregate(seed_rows)


def bench_upgrade(
    seeds: Sequence[int], n_donors: int, n_years: int,
    threshold: float = 1000.0, band: Tuple[float, float] = (100.0, 999.0),
    include_momentum: bool = False,
) -> List[Row]:
    """Upgrade model (build_leadership_snapshots + MajorGiftClassifier) vs the
    upgrade rule set (E.11a rule 3): this-year total, previous-year total
    plus this-year growth projected forward one more year, largest single
    gift in band."""
    seed_rows = []
    for seed in seeds:
        panel = make_donor_panel(n_donors=n_donors, n_years=n_years, random_state=seed)
        years = sorted(panel["gifts"]["fiscal_year"].unique())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            snaps = build_leadership_snapshots(
                panel["gifts"], fiscal_years=years[:-1], threshold=threshold, band=band,
                include_momentum=include_momentum,
            )
        if snaps.empty or snaps["fiscal_year"].nunique() < 2:
            continue
        feature_cols = [
            c for c in snaps.columns
            if c not in ("target", "fiscal_year") and pd.api.types.is_numeric_dtype(snaps[c])
        ]
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


def bench_forecast(seeds: Sequence[int], n_donors: int, n_years: int, horizon: int = 12) -> List[Row]:
    """FinancialForecastModel on monthly giving totals, last ``horizon``
    months held out, against the forecast rule set: the mean of the last 12
    training months, and seasonal naive (same month one year earlier).

    Two ways to use the model are scored, because they answer differently:
    ``predict_revenue_forecast`` rolls a frozen AR model of the target
    forward from its own recent values, so it never sees the future months'
    features; ``predict`` on the future months' calendar features (month
    index, sine and cosine of the month) uses them directly. Metrics: mean
    absolute percentage error per month, and the error on the held-out
    period's total (what a budget forecast is judged on)."""
    seed_rows = []
    for seed in seeds:
        gifts = make_donor_panel(n_donors=n_donors, n_years=n_years, random_state=seed)["gifts"]
        y = gifts.set_index("gift_date")["gift_amount"].resample("MS").sum().to_numpy(dtype="float64")
        t = np.arange(len(y))
        X = np.column_stack([t, np.sin(2 * np.pi * t / 12), np.cos(2 * np.pi * t / 12)])
        n_train = len(y) - horizon
        if n_train < 24:
            continue
        y_test = y[n_train:]
        model = FinancialForecastModel(random_state=seed).fit(X[:n_train], y[:n_train])
        preds = {
            "predict_revenue_forecast": model.predict_revenue_forecast(X[:n_train], horizon=horizon),
            "predict on future calendar features": model.predict(X[n_train:]),
        }
        rules = {
            "12-month mean": np.full(horizon, y[n_train - 12:n_train].mean()),
            "seasonal naive": y[n_train - 12:n_train][:horizon],
        }

        def mape(p: np.ndarray) -> float:
            return float(np.mean(np.abs(p - y_test) / np.maximum(np.abs(y_test), 1e-9)))

        def total_error(p: np.ndarray) -> float:
            return float(abs(p.sum() - y_test.sum()) / y_test.sum())

        rows = []
        for metric, fn in (("monthly_mape", mape), ("total_error", total_error)):
            best = min(rules, key=lambda k: fn(rules[k]))
            for name, p in preds.items():
                rows.append(
                    Row(
                        "synthetic_panel", f"FinancialForecastModel ({name})", metric, fn(p), fn(rules[best]),
                        lower_is_better=True,
                        note=f"rule_set={best} (best of {len(rules)}); split=last {horizon} months held out; n_configs=1",
                    )
                )
        seed_rows.append(rows)
    return _aggregate(seed_rows)


# --------------------------------------------------------------------------- #
# DonorsChoose (opt-in, local file; see load_donorschoose)
#
# ICPSR 37898 (DonorsChoose Open Data, United States, doi:10.3886/ICPSR37898.v1).
# Never downloaded or redistributed by this script; the caller points
# --donorschoose-path at a file obtained under their own ICPSR account, and
# only aggregate statistics are ever printed or written out.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _donorschoose_gift_log(path: str, subsample_fraction: float, seed: int) -> pd.DataFrame:
    """Citizen donors, positive amounts, subsampled to ``subsample_fraction``
    of distinct donors (seed ``seed``): the full file has ~3.1M distinct
    citizen donors, too many for build_leadership_snapshots' own per-donor
    streak loop (``_consecutive_years_given``) to finish in reasonable time
    for the lapse/ask population (anyone who gave, not just the $100-999
    upgrade band). Cached: the ~1.6 GB TSV would otherwise get re-parsed
    once per (bench function, momentum setting) call in the same process,
    and every caller only reads the returned frame."""
    gifts = load_donorschoose(path)
    gifts = gifts[(gifts["donor_type"] == "citizen donor") & (gifts["gift_amount"] > 0)].copy()
    rng = np.random.RandomState(seed)
    donors = gifts["donor_id"].unique()
    keep = set(rng.choice(donors, size=int(len(donors) * subsample_fraction), replace=False))
    return gifts[gifts["donor_id"].isin(keep)].copy()


def _gift_log_period_snapshots(
    gifts: pd.DataFrame, fiscal_year_start: int, kind: str, include_momentum: bool,
    threshold: float, band: Tuple[float, float],
) -> pd.DataFrame:
    """Shared donor x fiscal-year feature table for a gift-level log (any
    dataset in :func:`~philanthropy.ingest.build_leadership_snapshots`'s own
    shape), reusing that function's internals
    (``philanthropy.ingest._upgrade_snapshots``) for ``kind`` other than
    ``"upgrade"``, which it does not itself support:

    - ``"upgrade"``: candidates are donors whose FY T total falls in
      ``band`` (below ``threshold``); target is 1 if FY T+1 reaches
      ``threshold``.
    - ``"lapse"``: candidates are donors who gave anything (>0) in FY T;
      target is 1 if FY T+1 total is 0.
    - ``"ask"``: candidates are donors who gave in FY T *and* FY T+1; the
      (non-target) rows are the same feature columns, and ``target`` is the
      FY T+1 total (continuous), i.e. "next fiscal-year total, given they
      give again" (matching the synthetic ask bench's own yearly target).

    Only fiscal years with a resolvable T+1 (observed in the file) are
    included, so "gave nothing" and "year not in file" are never conflated.
    """
    df, pivot_sum, pivot_max, pivot_count = _prepare_gifts(gifts, fiscal_year_start)
    if df.empty:
        return pd.DataFrame(index=pd.Index([], name="donor_id"))
    all_years = sorted(int(y) for y in df["_fy"].unique())
    candidate_years = [y for y in all_years if y < max(all_years)]

    rows = []
    for fy in candidate_years:
        totals_t = _column(pivot_sum, fy)
        if totals_t.empty:
            continue
        next_totals = _column(pivot_sum, fy + 1)
        if kind == "upgrade":
            cand = totals_t[(totals_t >= band[0]) & (totals_t <= band[1]) & (totals_t < threshold)]
        else:
            cand = totals_t[totals_t > 0]
        if cand.empty:
            continue
        donor_ids = cand.index

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            snap = _snapshot_features_for_year(
                df, pivot_sum, pivot_max, pivot_count, donor_ids, fy, fiscal_year_start,
                activities=None, donors=None, donors_norm=None, include_momentum=include_momentum,
            )
        nxt = next_totals.reindex(donor_ids, fill_value=0.0)
        if kind == "upgrade":
            snap["target"] = (nxt >= threshold).astype("int64")
        elif kind == "lapse":
            snap["target"] = (nxt <= 0).astype("int64")
        elif kind == "ask":
            keep = (nxt > 0).to_numpy()
            snap = snap[keep]
            snap["target"] = nxt.to_numpy()[keep]
        else:
            raise ValueError(kind)
        rows.append(snap)

    if not rows:
        return pd.DataFrame(index=pd.Index([], name="donor_id"))
    out = pd.concat(rows)
    return out.reset_index().sort_values(["fiscal_year", "donor_id"], kind="stable")


def _snapshot_feature_cols(snap: pd.DataFrame, id_col: str) -> List[str]:
    return [
        c for c in snap.columns
        if c not in ("target", "fiscal_year", "wave", "period", id_col) and pd.api.types.is_numeric_dtype(snap[c])
    ]


# Lapse is asked about donors with at least two giving years (E.15 2.3.2):
# the donors a retention program can act on, not one-time givers whose lapse
# is near-certain. Every lapse bench builds its rows with the shared
# philanthropy.ingest builder so the model sees the same gift columns on
# every file; a bench's own frame only adds what that file has beyond them
# (momentum, PSID household income, wealth and volunteering).
LAPSE_MIN_YEARS_GIVEN = 2
_BUILDER_OWNED_COLS = (
    "fy_total", "fy_total_prior1", "fy_total_prior2", "fy_trend", "largest_gift", "gift_count",
    "consecutive_years_given", "months_since_last_gift", "fy_total_growth_ratio",
    "total_giving", "prior_wave_total", "trend", "largest_giving_category", "waves_given_streak", "target",
)


def _join_extras(snap: pd.DataFrame, extra: pd.DataFrame, id_col: str, period_col: str) -> pd.DataFrame:
    extra = extra.rename(columns={period_col: "period"})
    extra = extra.drop(columns=[c for c in _BUILDER_OWNED_COLS if c in extra.columns])
    return snap.merge(extra, on=[id_col, "period"], how="left")


def _donorschoose_lapse_snapshots(gifts: pd.DataFrame, include_momentum: bool) -> pd.DataFrame:
    snap = build_snapshots(gifts, kind="lapse", min_years_given=LAPSE_MIN_YEARS_GIVEN, fiscal_year_start=7)
    snap = snap.reset_index()
    if include_momentum:
        extra = _gift_log_period_snapshots(gifts, 7, "lapse", True, 1000.0, (100.0, 999.0))
        snap = _join_extras(snap, extra, "donor_id", "fiscal_year")
    return snap


def _psid_lapse_snapshots(data_path: str, do_path: str, include_momentum: bool) -> pd.DataFrame:
    """PSID waves through the shared period builder: a wave is a period and
    "prior" is the previous wave in the file. ``largest_gift`` is the
    largest single giving category, the closest this survey has to a
    largest gift."""
    long_df = _psid_long_table(data_path, do_path)
    giving_cols = [c for c in long_df.columns if c.startswith("giving_")]
    totals = long_df.pivot(index="household_key", columns="year", values="total_giving").sort_index(axis=1)
    largest = (
        long_df.assign(_largest=long_df[giving_cols].max(axis=1))
        .pivot(index="household_key", columns="year", values="_largest")
        .reindex(index=totals.index, columns=totals.columns)
    )
    snap = period_snapshots(totals, kind="lapse", largest=largest, min_years_given=LAPSE_MIN_YEARS_GIVEN).reset_index()
    extra = _psid_wave_period_snapshots(data_path, do_path, "lapse", include_momentum, 1000.0, (100.0, 999.0))
    return _join_extras(snap, extra, "household_key", "wave")


def _walk_forward_test_periods(snap: pd.DataFrame, period_col: str, n_folds: int) -> List[int]:
    periods = sorted(int(p) for p in snap[period_col].unique())
    return periods[-n_folds:]


def bench_upgrade_donorschoose(
    path: str, seed: int = DONORSCHOOSE_SEED, include_momentum: bool = False,
    subsample_fraction: float = DONORSCHOOSE_SUBSAMPLE, n_test_folds: int = DONORSCHOOSE_N_FOLDS,
    threshold: float = 1000.0, band: Tuple[float, float] = (100.0, 999.0),
) -> List[Row]:
    """Upgrade model vs the upgrade rule set (E.11a rule 3) on real
    DonorsChoose giving history: citizen donors, positive amounts, fiscal
    years Jul-Jun. Walk-forward, train on fiscal years < T, test on T, for
    the last ``n_test_folds`` fiscal years with a resolvable T+1 target."""
    gifts = _donorschoose_gift_log(path, subsample_fraction, seed)
    snap = _gift_log_period_snapshots(gifts, 7, "upgrade", include_momentum, threshold, band)
    if snap.empty:
        return []
    cols = _snapshot_feature_cols(snap, "donor_id")
    test_years = _walk_forward_test_periods(snap, "fiscal_year", n_test_folds)

    seed_rows = []
    for t in test_years:
        train, test = snap[snap["fiscal_year"] < t], snap[snap["fiscal_year"] == t]
        if train.empty or test.empty or test["target"].nunique() < 2:
            continue
        model = MajorGiftClassifier(random_state=seed).fit(
            train[cols].to_numpy("float64"), train["target"].to_numpy()
        )
        proba = model.predict_proba(test[cols].to_numpy("float64"))[:, 1]
        rules = {
            "this-year total": test["fy_total"].to_numpy(),
            "previous-year total plus this-year growth": (test["fy_total"] + test["fy_trend"]).to_numpy(),
            "largest single gift in band": test["largest_gift"].to_numpy(),
        }
        seed_rows.append(_classifier_rows(
            "donorschoose", "upgrade_model (MajorGiftClassifier)",
            test["target"].to_numpy(), proba, rules,
            f"walk-forward (subsample={subsample_fraction}, seed={seed}, fold=FY{t}, momentum={include_momentum})",
        ))
    return _aggregate(seed_rows)


def _donorschoose_lapse_fold(
    gifts: pd.DataFrame, seed: int, include_momentum: bool, n_test_folds: int, retention: bool,
) -> List[Row]:
    snap = _donorschoose_lapse_snapshots(gifts, include_momentum)
    if snap.empty:
        return []
    cols = _snapshot_feature_cols(snap, "donor_id")
    test_years = _walk_forward_test_periods(snap, "period", n_test_folds)

    seed_rows = []
    for t in test_years:
        train, test = snap[snap["period"] < t], snap[snap["period"] == t]
        if train.empty or test.empty or test["target"].nunique() < 2:
            continue
        model = LapsePredictor(random_state=seed).fit(
            train[cols].to_numpy("float64"), train["target"].to_numpy()
        )
        lapse_score = model.predict_lapse_score(test[cols].to_numpy("float64")) / 100.0
        rules = {
            "months since last gift": test["months_since_last_gift"].to_numpy(),
            "shortest giving streak (negated)": -test["consecutive_periods_given"].to_numpy(),
            "declining trend (negated growth)": -test["period_trend"].to_numpy(),
        }
        y_lapsed = test["target"].to_numpy()
        split = f"walk-forward (seed={seed}, fold=FY{t}, momentum={include_momentum})"
        if not retention:
            seed_rows.append(_classifier_rows("donorschoose", "LapsePredictor", y_lapsed, lapse_score, rules, split))
        else:
            retained = 1 - y_lapsed
            inv_rules = {name: -score for name, score in rules.items()}
            seed_rows.append(_classifier_rows(
                "donorschoose", "LapsePredictor", retained, -lapse_score, inv_rules,
                split + "; bottom decile by lapse score (retention read)",
            ))
    return _aggregate(seed_rows)


def bench_lapse_donorschoose(
    path: str, seed: int = DONORSCHOOSE_SEED, include_momentum: bool = False,
    subsample_fraction: float = DONORSCHOOSE_SUBSAMPLE, n_test_folds: int = DONORSCHOOSE_N_FOLDS,
) -> List[Row]:
    """LapsePredictor vs the lapse rule set on real DonorsChoose giving
    history: gave in FY T, $0 in FY T+1. Most donors here are one-and-done
    (base rate ~84% lapsed), so see :func:`bench_lapse_donorschoose_retention`
    for the more useful "who keeps giving" read."""
    gifts = _donorschoose_gift_log(path, subsample_fraction, seed)
    return _donorschoose_lapse_fold(gifts, seed, include_momentum, n_test_folds, retention=False)


def bench_lapse_donorschoose_retention(
    path: str, seed: int = DONORSCHOOSE_SEED, include_momentum: bool = False,
    subsample_fraction: float = DONORSCHOOSE_SUBSAMPLE, n_test_folds: int = DONORSCHOOSE_N_FOLDS,
) -> List[Row]:
    """The useful list on a file where most donors lapse: not the top decile
    by lapse score, but the bottom decile (least likely to lapse), with the
    label flipped to "retained" (base rate ~16%). Same fit, same rules, same
    split as :func:`bench_lapse_donorschoose`, ranked in the opposite
    direction."""
    gifts = _donorschoose_gift_log(path, subsample_fraction, seed)
    return _donorschoose_lapse_fold(gifts, seed, include_momentum, n_test_folds, retention=True)


def bench_ask_donorschoose(
    path: str, seed: int = DONORSCHOOSE_SEED, include_momentum: bool = False,
    subsample_fraction: float = DONORSCHOOSE_SUBSAMPLE, n_test_folds: int = DONORSCHOOSE_N_FOLDS,
) -> List[Row]:
    """AskAmountRecommender vs the ask rule set on real DonorsChoose giving
    history: predicts "next fiscal-year total, given they give again" (gift
    dates in this file are month-resolution, so a true next-single-gift
    amount is not well defined; this matches the synthetic ask bench's own
    yearly target instead). Rules: last period's total, max(last period,
    average next-gift amount in train), median training next-gift amount."""
    gifts = _donorschoose_gift_log(path, subsample_fraction, seed)
    snap = _gift_log_period_snapshots(gifts, 7, "ask", include_momentum, threshold=1000.0, band=(100.0, 999.0))
    if snap.empty:
        return []
    cols = _snapshot_feature_cols(snap, "donor_id")
    test_years = _walk_forward_test_periods(snap, "fiscal_year", n_test_folds)

    rows: List[Row] = []
    for t in test_years:
        train, test = snap[snap["fiscal_year"] < t], snap[snap["fiscal_year"] == t]
        if len(train) < 20 or len(test) < 5:
            continue
        ytr, yte = train["target"].to_numpy("float64"), test["target"].to_numpy("float64")
        model = AskAmountRecommender(random_state=seed).fit(train[cols].to_numpy("float64"), ytr)
        pred = model.predict(test[cols].to_numpy("float64"))
        last_total = test["fy_total"].to_numpy()
        avg_next_gift = float(ytr.mean())
        median_next_gift = np.full_like(yte, float(np.median(ytr)))
        rules = {
            "last period's total": last_total,
            "max(last period, average next-gift in train)": np.maximum(last_total, avg_next_gift),
            "median training next-gift": median_next_gift,
        }
        best_name, best_score, _ = _best_ask_baseline(yte, rules)
        header = f"rule_set={best_name} (best of {len(rules)}); walk-forward (seed={seed}, fold=FY{t}, momentum={include_momentum})"
        rows.append(Row(
            "donorschoose", "AskAmountRecommender", "mae",
            mean_absolute_error(yte, pred), mean_absolute_error(yte, best_score),
            lower_is_better=True, note=header,
        ))
        rows.append(Row(
            "donorschoose", "AskAmountRecommender", "within25pct",
            _within_pct(pred, yte), _within_pct(best_score, yte), note=header,
        ))
        rows.append(Row(
            "donorschoose", "AskAmountRecommender", "revenue_top10pct",
            _revenue_top_share(yte, pred), _revenue_top_share(yte, best_score), note=header,
        ))
    return _aggregate([rows]) if rows else []


# --------------------------------------------------------------------------- #
# PSID (opt-in, local file; see load_psid_philanthropy)
#
# Panel Study of Income Dynamics, public use dataset, produced and
# distributed by the Survey Research Center, Institute for Social Research,
# University of Michigan, Ann Arbor, MI. Never downloaded or redistributed by
# this script; the caller points --psid-data/--psid-do at files obtained
# themselves from the PSID Data Center, and only aggregate statistics are
# ever printed or written out.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _psid_long_table(data_path: str, do_path: str) -> pd.DataFrame:
    """Cached wrapper around :func:`load_psid_philanthropy`: the extract
    would otherwise get re-parsed once per (bench function, momentum
    setting) call in the same process, and every caller only reads the
    returned frame."""
    return load_psid_philanthropy(data_path, do_path)


def _psid_wave_period_snapshots(
    data_path: str, do_path: str, kind: str, include_momentum: bool,
    threshold: float, band: Tuple[float, float],
) -> pd.DataFrame:
    """Household x wave feature table from :func:`load_psid_philanthropy`'s
    own long table (one row per household per wave already): waves are
    biennial and not every household appears in every wave, so "next wave"
    is the next wave *year actually present in the file*, not year+2
    arithmetic. Same three ``kind`` values as
    :func:`_gift_log_period_snapshots`, scored on ``total_giving`` instead of
    a gift-level pivot.

    A household is only ever labelled from a next wave it was actually
    observed in with a resolvable (non-NaN) ``total_giving``: attrition
    (death, no longer Head, dropped from the extract) and a wave where every
    giving category came back Don't-Know/NA (``total_giving`` itself NaN)
    both mean "unknown", not "gave $0", and are excluded from that fold
    rather than counted as a lapse or a missed upgrade."""
    long_df = _psid_long_table(data_path, do_path)
    waves = sorted(int(w) for w in long_df["year"].unique())
    if len(waves) < 2:
        return pd.DataFrame(index=pd.Index([], name="household_key"))
    next_wave = {w: waves[i + 1] for i, w in enumerate(waves[:-1])}
    giving_cols = [c for c in long_df.columns if c.startswith("giving_")]
    carry_cols = [
        "total_giving", "itemized_charitable_contrib_amount", "family_income", "wealth1", "wealth2",
        "head_volunteer_hours_annual", "spouse_volunteer_hours_annual", "household_volunteer_hours_regular",
        "head_volunteer_hours_typical_week", "spouse_volunteer_hours_typical_week",
    ] + giving_cols

    # Household x wave pivot of total_giving, for the streak walk below:
    # O(households x waves) via vectorised cumulative-run logic instead of
    # the O(households x waves x rows) row-filtering a per-household loop
    # over `long_df` would do.
    pivot_total = long_df.pivot(index="household_key", columns="year", values="total_giving")
    gave = (pivot_total[waves].fillna(0.0) > 0)
    streak_by_wave = pd.DataFrame(index=gave.index, columns=waves, dtype="int64")
    running = pd.Series(0, index=gave.index, dtype="int64")
    for w in waves:
        running = (running + 1).where(gave[w], 0)
        streak_by_wave[w] = running

    long_df = long_df.copy()
    long_df["_date"] = pd.to_datetime(long_df["year"].astype(str) + "-12-31")
    data_start = long_df["_date"].min()

    rows = []
    for w in waves:
        if w not in next_wave:
            continue
        cur = long_df[long_df["year"] == w].set_index("household_key")
        nxt_observed = long_df[long_df["year"] == next_wave[w]].set_index("household_key")["total_giving"].dropna()

        if kind == "upgrade":
            cand = cur[(cur["total_giving"] >= band[0]) & (cur["total_giving"] <= band[1]) & (cur["total_giving"] < threshold)]
        else:
            cand = cur[cur["total_giving"] > 0]
        cand = cand[cand.index.isin(nxt_observed.index)]
        if cand.empty:
            continue
        household_ids = cand.index

        snap = cand[carry_cols].copy()
        snap["wave"] = w
        snap["largest_giving_category"] = cand[giving_cols].max(axis=1)

        prior_idx = waves.index(w) - 1
        if prior_idx >= 0:
            prior_total = long_df[long_df["year"] == waves[prior_idx]].set_index("household_key")["total_giving"]
        else:
            prior_total = pd.Series(dtype="float64")
        snap["prior_wave_total"] = prior_total.reindex(household_ids)
        snap["trend"] = snap["total_giving"] - snap["prior_wave_total"]
        snap["waves_given_streak"] = streak_by_wave.loc[household_ids, w].to_numpy()

        if include_momentum:
            momentum = trailing_slope_features(
                long_df[["household_key", "_date", "total_giving"]].rename(columns={"total_giving": "_amount"}),
                household_ids, pd.Timestamp(f"{w}-12-31"), date_col="_date", value_col="_amount", agg="sum",
                prefix="total_giving", donor_col="household_key", period_months=24, data_start=data_start,
            )
            snap = snap.join(momentum, how="left")

        nxt = nxt_observed.reindex(household_ids)
        if kind == "upgrade":
            snap["target"] = (nxt >= threshold).astype("int64")
        elif kind == "lapse":
            snap["target"] = (nxt <= 0).astype("int64")
        elif kind == "ask":
            keep = (nxt > 0).to_numpy()
            snap = snap[keep]
            snap["target"] = nxt.to_numpy()[keep]
        else:
            raise ValueError(kind)
        rows.append(snap)

    if not rows:
        return pd.DataFrame(index=pd.Index([], name="household_key"))
    out = pd.concat(rows)
    return out.reset_index().sort_values(["wave", "household_key"], kind="stable")


def bench_upgrade_psid(
    data_path: str, do_path: str, seed: int = PSID_SEED, include_momentum: bool = False,
    n_test_folds: int = PSID_N_FOLDS, threshold: float = 1000.0, band: Tuple[float, float] = (100.0, 999.0),
) -> List[Row]:
    """Upgrade model vs the upgrade rule set on real PSID household giving:
    band $100-999 in wave W, crossing $1,000 in the next observed wave.
    Walk-forward over the last ``n_test_folds`` wave-pairs with a resolvable
    next-wave target."""
    snap = _psid_wave_period_snapshots(data_path, do_path, "upgrade", include_momentum, threshold, band)
    if snap.empty:
        return []
    cols = _snapshot_feature_cols(snap, "household_key")
    test_waves = _walk_forward_test_periods(snap, "wave", n_test_folds)

    seed_rows = []
    for w in test_waves:
        train, test = snap[snap["wave"] < w], snap[snap["wave"] == w]
        if train.empty or test.empty or test["target"].nunique() < 2:
            continue
        model = MajorGiftClassifier(random_state=seed).fit(
            train[cols].to_numpy("float64"), train["target"].to_numpy()
        )
        proba = model.predict_proba(test[cols].to_numpy("float64"))[:, 1]
        rules = {
            "this-wave total": test["total_giving"].to_numpy(),
            "previous-wave total plus this-wave growth": (test["total_giving"] + test["trend"]).to_numpy(),
            "largest giving category in band": test["largest_giving_category"].to_numpy(),
        }
        seed_rows.append(_classifier_rows(
            "psid", "upgrade_model (MajorGiftClassifier)", test["target"].to_numpy(), proba, rules,
            f"walk-forward (seed={seed}, fold=wave{w}, momentum={include_momentum})",
        ))
    return _aggregate(seed_rows)


def _psid_lapse_fold(
    data_path: str, do_path: str, seed: int, include_momentum: bool, n_test_folds: int, retention: bool,
) -> List[Row]:
    snap = _psid_lapse_snapshots(data_path, do_path, include_momentum)
    if snap.empty:
        return []
    cols = _snapshot_feature_cols(snap, "household_key")
    test_waves = _walk_forward_test_periods(snap, "period", n_test_folds)

    seed_rows = []
    for w in test_waves:
        train, test = snap[snap["period"] < w], snap[snap["period"] == w]
        if train.empty or test.empty or test["target"].nunique() < 2:
            continue
        model = LapsePredictor(random_state=seed).fit(train[cols].to_numpy("float64"), train["target"].to_numpy())
        lapse_score = model.predict_lapse_score(test[cols].to_numpy("float64")) / 100.0
        rules = {
            "shortest giving streak (negated)": -test["consecutive_periods_given"].to_numpy(),
            "declining trend (negated growth)": -test["period_trend"].to_numpy(),
            # Smallest givers lapse first. "Months since last gift" and the
            # LYBUNT flag are not here: every household in this population
            # gave in wave W and dates are wave-level, so both are constant.
            "this-wave total (negated)": -test["period_total"].to_numpy(),
        }
        y_lapsed = test["target"].to_numpy()
        split = f"walk-forward (seed={seed}, fold=wave{w}, momentum={include_momentum})"
        if not retention:
            seed_rows.append(_classifier_rows("psid", "LapsePredictor", y_lapsed, lapse_score, rules, split))
        else:
            retained = 1 - y_lapsed
            inv_rules = {name: -score for name, score in rules.items()}
            seed_rows.append(_classifier_rows(
                "psid", "LapsePredictor", retained, -lapse_score, inv_rules,
                split + "; bottom decile by lapse score (retention read)",
            ))
    return _aggregate(seed_rows)


def bench_lapse_psid(
    data_path: str, do_path: str, seed: int = PSID_SEED, include_momentum: bool = False,
    n_test_folds: int = PSID_N_FOLDS,
) -> List[Row]:
    """LapsePredictor vs the lapse rule set on real PSID household giving:
    gave in wave W, $0 in the next observed wave."""
    return _psid_lapse_fold(data_path, do_path, seed, include_momentum, n_test_folds, retention=False)


def bench_lapse_psid_retention(
    data_path: str, do_path: str, seed: int = PSID_SEED, include_momentum: bool = False,
    n_test_folds: int = PSID_N_FOLDS,
) -> List[Row]:
    """The bottom-decile-by-lapse-score, label-flipped "retained" read of
    :func:`bench_lapse_psid`, for whichever PSID waves lapse dominates."""
    return _psid_lapse_fold(data_path, do_path, seed, include_momentum, n_test_folds, retention=True)


def bench_ask_psid(
    data_path: str, do_path: str, seed: int = PSID_SEED, include_momentum: bool = False,
    n_test_folds: int = PSID_N_FOLDS,
) -> List[Row]:
    """AskAmountRecommender vs the ask rule set on real PSID household
    giving: predicts next-wave total, given the household gives again.
    Rules: last wave's total, max(last wave, average next-wave amount in
    train), median training next-wave amount."""
    snap = _psid_wave_period_snapshots(data_path, do_path, "ask", include_momentum, threshold=1000.0, band=(100.0, 999.0))
    if snap.empty:
        return []
    cols = _snapshot_feature_cols(snap, "household_key")
    test_waves = _walk_forward_test_periods(snap, "wave", n_test_folds)

    rows: List[Row] = []
    for w in test_waves:
        train, test = snap[snap["wave"] < w], snap[snap["wave"] == w]
        if len(train) < 20 or len(test) < 5:
            continue
        ytr, yte = train["target"].to_numpy("float64"), test["target"].to_numpy("float64")
        model = AskAmountRecommender(random_state=seed).fit(train[cols].to_numpy("float64"), ytr)
        pred = model.predict(test[cols].to_numpy("float64"))
        last_total = test["total_giving"].to_numpy()
        avg_next = float(ytr.mean())
        rules = {
            "last wave's total": last_total,
            "max(last wave, average next-wave in train)": np.maximum(last_total, avg_next),
            "median training next-wave": np.full_like(yte, float(np.median(ytr))),
        }
        best_name, best_score, _ = _best_ask_baseline(yte, rules)
        header = f"rule_set={best_name} (best of {len(rules)}); walk-forward (seed={seed}, fold=wave{w}, momentum={include_momentum})"
        rows.append(Row(
            "psid", "AskAmountRecommender", "mae",
            mean_absolute_error(yte, pred), mean_absolute_error(yte, best_score),
            lower_is_better=True, note=header,
        ))
        rows.append(Row(
            "psid", "AskAmountRecommender", "within25pct",
            _within_pct(pred, yte), _within_pct(best_score, yte), note=header,
        ))
        rows.append(Row(
            "psid", "AskAmountRecommender", "revenue_top10pct",
            _revenue_top_share(yte, pred), _revenue_top_share(yte, best_score), note=header,
        ))
    return _aggregate([rows]) if rows else []


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


def bench_kdd_upgrade(
    seed: int, threshold: float = 50.0, band: Tuple[float, float] = (5.0, 49.0),
    include_momentum: bool = False,
) -> List[Row]:
    """Upgrade model on KDD98, threshold rescaled from the $1000/$100-999
    defaults: this file's per-donor annual giving tops out far lower than a
    major-gift program's, so $50/$5-49 keeps the same shape (threshold is
    roughly the 93rd percentile of per-donor annual giving on this file).

    Split is walk-forward by fiscal year (July start): train on the FY1994
    snapshot (outcome FY1995), test on FY1995 (outcome FY1996). The RDATE
    gift log effectively ends in March 1996 (10,000+ gifts a month up to
    then, under 150 a month after, plus a few mis-dated 1998 entries), so
    FY1996 as a test year has an outcome year that is not in the file (3
    positives in 75,272 rows) and is dropped; FY1995's outcome year is
    observed for 9 of its 12 months, so "upgraded" there means "reached the
    threshold by March 1996", the same cut for every donor. A random row
    split over the multi-year snapshot table is not used: with
    ``fiscal_year`` as a feature it let the model learn which year a row
    came from (the near-empty FY1996) instead of who upgrades, and it put
    the same donor in train and test. ``fiscal_year`` is dropped from the
    features, since it is constant within each side of the split."""
    donors = fetch_kdd98_donors()
    gifts = _kdd_gift_log(donors)
    train_fy, test_fy = 1994, 1995
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        snaps = build_leadership_snapshots(
            gifts, fiscal_years=[train_fy, test_fy], threshold=threshold, band=band, fiscal_year_start=7,
            include_momentum=include_momentum,
        )
    feature_cols = [
        c for c in snaps.columns
        if c not in ("target", "fiscal_year") and pd.api.types.is_numeric_dtype(snaps[c])
    ]
    train, test = snaps[snaps["fiscal_year"] == train_fy], snaps[snaps["fiscal_year"] == test_fy]
    y_test = test["target"].to_numpy()

    model = MajorGiftClassifier(random_state=seed).fit(
        train[feature_cols].to_numpy(dtype="float64"), train["target"].to_numpy()
    )
    proba = model.predict_proba(test[feature_cols].to_numpy(dtype="float64"))[:, 1]
    rules = {
        "this-year total": test["fy_total"].to_numpy(),
        "previous-year total plus this-year growth": (test["fy_total"] + test["fy_trend"]).to_numpy(),
        "largest single gift in band": test["largest_gift"].to_numpy(),
    }
    return _classifier_rows(
        "kdd98", "upgrade_model (MajorGiftClassifier)", y_test, proba, rules,
        f"walk-forward by fiscal year (train FY{train_fy}, test FY{test_fy})", bootstrap=True,
    )


def _kdd_response_frame(donors: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """RFM + tenure features for every donor in a KDD98 file (never-gave
    donors get recency one past the file maximum and zero counts), plus the
    donor table indexed by ``CONTROLN``. Shared by the learning-file split
    and the cup98VAL rows."""
    rfm = _kdd_rfm(_kdd_gift_log(donors))
    donors_idx = donors.set_index("CONTROLN")
    X_rfm = rfm.reindex(donors_idx.index)
    X_rfm["recency"] = X_rfm["recency"].fillna(X_rfm["recency"].max() + 1)
    X_rfm[["frequency", "monetary", "tenure"]] = X_rfm[["frequency", "monetary", "tenure"]].fillna(0.0)
    return X_rfm, donors_idx


def _kdd_response_rules(X_rfm: pd.DataFrame, donors_idx: pd.DataFrame) -> Dict[str, np.ndarray]:
    """The fixed KDD98 response rule set (E.11a rule 3) for the given rows.
    ``RFA_2F`` is the frequency code of the RFA_2 segment, an integer 1 to 4
    (4 = four or more gifts in the period), used directly as the rank."""
    rfa_2f = donors_idx["RFA_2F"].to_numpy(dtype="float64")
    return {
        "lifetime monetary": X_rfm["monetary"].to_numpy(),
        "RFM cell score": _rfm_cell_score(
            X_rfm["recency"].to_numpy(), X_rfm["frequency"].to_numpy(), X_rfm["monetary"].to_numpy()
        ),
        "RFA_2 frequency then last gift": rfa_2f * 1e6 + donors_idx["LASTGIFT"].to_numpy(),
    }


def bench_kdd_response(seed: int) -> List[Row]:
    """DonorPropensityModel / MajorGiftClassifier vs the response rule set
    (E.11a rule 3): lifetime monetary, RFM cell score, and (KDD98 only) RFA_2
    frequency then last gift. ``RFA_2`` is safe to use here: per
    ``cup98dic.txt`` it is "donor's RFA status as of 97NK promotion date",
    i.e. computed when the 97NK mailing was sent and not updated by its
    response (``TARGET_B``/``TARGET_D``), the same "as of, not updated by"
    check rule 5 asks for. ``HIT`` and ``LASTDATE`` are not used by this
    script."""
    X_rfm, donors_idx = _kdd_response_frame(fetch_kdd98_donors())
    y = donors_idx["TARGET_B"].to_numpy()

    idx_train, _idx_val, idx_test = _split_55_15_30(len(y), y, seed)
    Xtr, Xte, ytr, yte = X_rfm.iloc[idx_train], X_rfm.iloc[idx_test], y[idx_train], y[idx_test]
    rules = _kdd_response_rules(Xte, donors_idx.iloc[idx_test])
    rows: List[Row] = []
    for name, cls in (("DonorPropensityModel", DonorPropensityModel), ("MajorGiftClassifier", MajorGiftClassifier)):
        model = cls(random_state=seed).fit(Xtr.to_numpy(), ytr)
        proba = model.predict_proba(Xte.to_numpy())[:, 1]
        rows += _classifier_rows("kdd98", name, yte, proba, rules, KDD_SPLIT, bootstrap=True)
    return rows


def _kdd_lapse_panel(donors: pd.DataFrame) -> pd.DataFrame:
    """One row per (donor, promotion period) over the 22-promotion history:
    as-of cumulative total and count, that period's gift, streak, periods
    since last gift and the previous period's gift, labelled ``lapsed`` if
    the donor gave nothing in the next period (the last period's next one is
    the 97NK mailing, ``TARGET_B``)."""
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
    return pd.concat(periods, ignore_index=True)


def _kdd_lapse_rules(test_p: pd.DataFrame) -> Dict[str, np.ndarray]:
    """The fixed lapse rule set (E.11a rule 3) for the given panel rows."""
    return {
        "LYBUNT/SYBUNT flag": ((test_p["recent"].to_numpy() == 0) & (test_p["prev_recent"].to_numpy() > 0)).astype(float),
        "years since last gift": test_p["years_since_last"].to_numpy(),
        "shortest giving streak": -test_p["streak"].to_numpy(),
        "gave nothing last period": -test_p["recent"].to_numpy(),
    }


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
    panel = _kdd_lapse_panel(fetch_kdd98_donors())
    last_period = panel["period"].max()
    val_period = last_period - 1
    train_p = panel[panel["period"] < val_period]
    test_p = panel[panel["period"] == last_period]

    model = LapsePredictor(n_estimators=100, max_depth=10, random_state=seed).fit(
        train_p[["total", "n", "recent"]].to_numpy(), train_p["lapsed"].to_numpy()
    )
    score = model.predict_lapse_score(test_p[["total", "n", "recent"]].to_numpy()) / 100.0
    y = test_p["lapsed"].to_numpy()
    rules = _kdd_lapse_rules(test_p)
    return _classifier_rows(
        "kdd98", "LapsePredictor", y, score, rules,
        "walk-forward across KDD98 promotion periods (train < period N-1, val=N-1, test=N)",
        bootstrap=True,
    )


def bench_kdd_lapse_retention(seed: int) -> List[Row]:
    """The useful list on a file where almost everyone lapses (E.12d): not
    the top decile by lapse score, but the bottom decile, the donors the
    model is *least* confident will lapse. Same fit, same rules, same
    split as :func:`bench_kdd_lapse`, ranked in the opposite direction with
    the label flipped to "retained", so the reported hit rate reads as a
    retention rate rather than a lapse rate."""
    panel = _kdd_lapse_panel(fetch_kdd98_donors())
    last_period = panel["period"].max()
    val_period = last_period - 1
    train_p = panel[panel["period"] < val_period]
    test_p = panel[panel["period"] == last_period]

    model = LapsePredictor(n_estimators=100, max_depth=10, random_state=seed).fit(
        train_p[["total", "n", "recent"]].to_numpy(), train_p["lapsed"].to_numpy()
    )
    lapse_score = model.predict_lapse_score(test_p[["total", "n", "recent"]].to_numpy()) / 100.0
    retained = 1 - test_p["lapsed"].to_numpy()
    inv_rules = {name: -score for name, score in _kdd_lapse_rules(test_p).items()}
    return _classifier_rows(
        "kdd98", "LapsePredictor", retained, -lapse_score, inv_rules,
        "walk-forward across KDD98 promotion periods; bottom decile by lapse score (E.12d retention read)",
        bootstrap=True,
    )


def bench_kdd_lapse_backend_trial(seed: int) -> List[Row]:
    """E.12d LapsePredictor backend trial: RandomForest (today's default) vs
    ``backend="hist_gradient_boosting"``, picked on cup98val rather than
    KDD98's own period N-1 (E.12c item 3: that period is 99.7% base lapse,
    too degenerate to pick anything on). The winner's KDD98-split test
    number (the same held-out period :func:`bench_kdd_lapse` reports) is the
    only one that counts. Adopt only if the winner beats "years since last
    gift" at top-10% on one real file without losing on the other."""
    lrn, val = fetch_kdd98_donors(), fetch_kdd98_val_donors()
    panel_lrn, panel_val = _kdd_lapse_panel(lrn), _kdd_lapse_panel(val)
    last_period = panel_lrn["period"].max()
    train_p = panel_lrn[panel_lrn["period"] < last_period]
    test_p = panel_lrn[panel_lrn["period"] == last_period]
    val_p = panel_val[panel_val["period"] == last_period]

    candidates = {
        "random_forest": LapsePredictor(n_estimators=100, max_depth=10, random_state=seed),
        "hist_gradient_boosting": LapsePredictor(backend="hist_gradient_boosting", max_depth=10, random_state=seed),
    }
    val_auc: Dict[str, float] = {}
    for name, model in candidates.items():
        model.fit(train_p[["total", "n", "recent"]].to_numpy(), train_p["lapsed"].to_numpy())
        score = model.predict_lapse_score(val_p[["total", "n", "recent"]].to_numpy()) / 100.0
        val_auc[name] = roc_auc_score(val_p["lapsed"].to_numpy(), score)

    winner = max(val_auc, key=val_auc.get)
    runner_up = [n for n in val_auc if n != winner][0]
    model = candidates[winner]
    score_test = model.predict_lapse_score(test_p[["total", "n", "recent"]].to_numpy()) / 100.0
    y_test = test_p["lapsed"].to_numpy()
    rules = _kdd_lapse_rules(test_p)
    return _classifier_rows(
        "kdd98", "LapsePredictor (E.12d backend trial)", y_test, score_test, rules,
        f"walk-forward across KDD98 promotion periods; backend picked on cup98val's last period "
        f"(winner={winner}, AUC {val_auc[winner]:.3f} vs {runner_up} {val_auc[runner_up]:.3f})",
        n_configs=2, bootstrap=True,
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
    mae_d = _bootstrap_diff_ci(len(y_true), lambda idx: (
        mean_absolute_error(y_true[idx], pred[idx]) - mean_absolute_error(y_true[idx], best_score[idx])
    ))
    within_d = _bootstrap_diff_ci(len(y_true), lambda idx: (
        _within_pct(pred[idx], y_true[idx]) - _within_pct(best_score[idx], y_true[idx])
    ))
    revenue_d = _bootstrap_diff_ci(len(y_true), lambda idx: (
        _revenue_top_share(y_true[idx], pred[idx]) - _revenue_top_share(y_true[idx], best_score[idx])
    ))

    return [
        Row(
            "kdd98", "AskAmountRecommender", "mae",
            mean_absolute_error(y_true, pred), mean_absolute_error(y_true, best_score),
            lower_is_better=True, note=header, diff_lo=mae_d[0], diff_hi=mae_d[1],
        ),
        Row(
            "kdd98", "AskAmountRecommender", "within25pct",
            _within_pct(pred, y_true), _within_pct(best_score, y_true),
            note=header, diff_lo=within_d[0], diff_hi=within_d[1],
        ),
        Row(
            "kdd98", "AskAmountRecommender", "revenue_top10pct",
            _revenue_top_share(y_true, pred), _revenue_top_share(y_true, best_score),
            note=header, diff_lo=revenue_d[0], diff_hi=revenue_d[1],
        ),
    ]


def _with_max_last_avg_feature(X: pd.DataFrame) -> pd.DataFrame:
    return X.assign(max_last_avg=np.maximum(X["LASTGIFT"], X["AVGGIFT"]))


def _pick_ask_candidate(
    Xd_train: pd.DataFrame, yd_train: pd.Series, Xd_val: pd.DataFrame, yd_val: pd.Series, seed: int,
) -> Tuple[str, Any, Any, Dict[str, float]]:
    """E.12d ask-model improvement attempt, under the E.11a protocol: two
    candidate configurations, picked on a validation fold that
    :func:`bench_kdd_ask` does not use, so only the winner's later test-fold
    number counts.

    Candidate 1, ``absolute+max_feature``: today's absolute target, with
    max(last gift, average gift) added as an explicit input column (the
    model already has LASTGIFT and AVGGIFT separately and still loses to
    their max). Candidate 2, ``relative_target``: predicts
    log(next gift / max(last, average)) and multiplies back
    (``AskAmountRecommender(target_mode="relative")``), targeting the exact
    rule this model keeps losing to. Returns the winner's name, fitted
    pipeline, its featurizer and every candidate's validation MAE."""
    responders_train, responders_val = yd_train > 0, yd_val > 0
    last_idx, avg_idx = Xd_train.columns.get_loc("LASTGIFT"), Xd_train.columns.get_loc("AVGGIFT")

    candidates = {
        "absolute+max_feature": (
            make_pipeline(
                WealthScreeningImputer(wealth_cols=["WEALTH1", "WEALTH2", "INCOME"]),
                AskAmountRecommender(random_state=seed),
            ),
            _with_max_last_avg_feature,
        ),
        "relative_target": (
            make_pipeline(
                WealthScreeningImputer(wealth_cols=["WEALTH1", "WEALTH2", "INCOME"]),
                AskAmountRecommender(
                    target_mode="relative", last_gift_idx=last_idx, avg_gift_idx=avg_idx, random_state=seed,
                ),
            ),
            lambda X: X,
        ),
    }

    val_mae: Dict[str, float] = {}
    for name, (pipe, featurize) in candidates.items():
        pipe.fit(featurize(Xd_train)[responders_train], yd_train[responders_train])
        pred_val = pipe.predict(featurize(Xd_val)[responders_val])
        val_mae[name] = mean_absolute_error(yd_val[responders_val], pred_val)

    winner = min(val_mae, key=val_mae.get)
    pipe, featurize = candidates[winner]
    return winner, pipe, featurize, val_mae


def bench_kdd_ask_relative(seed: int) -> List[Row]:
    """Test-fold number for the E.12d ask-model candidate picked by
    :func:`_pick_ask_candidate` on the KDD98 learning-file validation fold.
    Reported as a separate model name so it sits next to, not instead of,
    :func:`bench_kdd_ask`'s current-default row until an adopt decision is
    made (E.12d: adopt only if a candidate wins on both KDD files)."""
    donors = fetch_kdd98_donors()
    rfm = _kdd_rfm(_kdd_gift_log(donors))
    Xd_train, Xd_val, Xd_test, yd_train, yd_val, yd_test = _kdd_ask_design(donors, rfm, seed)
    responders_train, responders_test = yd_train > 0, yd_test > 0
    winner, pipe, featurize, val_mae = _pick_ask_candidate(Xd_train, yd_train, Xd_val, yd_val, seed)
    runner_up = [n for n in val_mae if n != winner][0]
    pred_test = pipe.predict(featurize(Xd_test)[responders_test])

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
    header = (
        f"E.12d relative-target experiment; winner={winner} (validation-fold MAE "
        f"{val_mae[winner]:.2f} vs {runner_up}'s {val_mae[runner_up]:.2f}); "
        f"rule_set={best_name} (best of {len(rules)}); split={KDD_SPLIT}; n_configs=2"
    )
    return [
        Row(
            "kdd98", "AskAmountRecommender (E.12d candidate)", "mae",
            mean_absolute_error(y_true, pred_test), mean_absolute_error(y_true, best_score),
            lower_is_better=True, note=header,
        ),
        Row(
            "kdd98", "AskAmountRecommender (E.12d candidate)", "within25pct",
            _within_pct(pred_test, y_true), _within_pct(best_score, y_true),
            note=header,
        ),
    ]


def _net_revenue_diff_ci(y: np.ndarray, mail: np.ndarray, cost: float) -> Tuple[float, float]:
    """Paired bootstrap interval on (net revenue mailing ``mail``) - (net
    revenue mailing everyone). Per donor that difference is ``-(y - cost)``
    for every donor not mailed and 0 otherwise, so a resample sums it."""
    per_donor = np.where(mail, 0.0, -(np.asarray(y, dtype="float64") - cost))
    return _bootstrap_diff_ci(len(per_donor), lambda idx: float(per_donor[idx].sum()))


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
    net_d = _net_revenue_diff_ci(yd_test.to_numpy(), np.asarray(mail), cost)
    roi_all = fundraising_roi(total_raised=raised_all, total_fundraising_expense=cost_all)
    roi_mail = fundraising_roi(total_raised=raised_mail, total_fundraising_expense=cost_mail)

    return [
        Row(
            "kdd98", "cost_aware_selection", "net_revenue",
            raised_mail - cost_mail, raised_all - cost_all,
            note=f"mail if E[gift]>${cost:.2f}; pieces={int(mail.sum())}/{len(Xd_test)}",
            diff_lo=net_d[0], diff_hi=net_d[1],
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
    net_d = _net_revenue_diff_ci(y_val.to_numpy(), np.asarray(mail), cost)
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
            diff_lo=net_d[0], diff_hi=net_d[1],
        ),
        Row(
            "cup98val", "cost_aware_selection", "roi",
            roi_mail, roi_all, note="baseline=mail everyone; KDD98's own held-out validation file",
        ),
    ]


def _mail_profit_curve(y: np.ndarray, expected_gift: np.ndarray, cost: float, n_points: int = 40) -> Dict[str, Any]:
    """Cumulative net revenue if the top-k donors by expected gift are
    mailed, for k stepped from 0 to the full test set (E.13c: the profit
    curve replacing the two-bar who-to-mail chart). Also reports the
    model's own "mail if E[gift]>cost" stopping point on the same curve, so
    a chart can mark both it and the mail-everyone endpoint."""
    order = np.argsort(-expected_gift)
    y_sorted = np.asarray(y)[order]
    cum_raised = np.concatenate([[0.0], np.cumsum(y_sorted)])
    n = len(y_sorted)
    ks = np.unique(np.linspace(0, n, n_points).astype(int))
    net = cum_raised[ks] - cost * ks
    stop_k = int((expected_gift > cost).sum())
    return {
        "mailed": ks.tolist(),
        "net_revenue": net.tolist(),
        "n_total": n,
        "stop_k": stop_k,
        "stop_net_revenue": float(cum_raised[stop_k] - cost * stop_k),
        "everyone_net_revenue": float(net[-1]),
    }


def kdd_mail_profit_curve(seed: int, cost: float = 0.68) -> Dict[str, Any]:
    """Profit curve for bench_kdd_cost_aware's own fit and test split."""
    donors = fetch_kdd98_donors()
    rfm = _kdd_rfm(_kdd_gift_log(donors))
    Xd_train, _Xd_val, Xd_test, yd_train, _yd_val, yd_test = _kdd_ask_design(donors, rfm, seed)
    responders_train = yd_train > 0

    resp_model = make_pipeline(
        WealthScreeningImputer(wealth_cols=["WEALTH1", "WEALTH2", "INCOME"]),
        MajorGiftClassifier(random_state=seed),
    ).fit(Xd_train, (yd_train > 0).astype(int))
    ask_model = make_pipeline(
        WealthScreeningImputer(wealth_cols=["WEALTH1", "WEALTH2", "INCOME"]),
        AskAmountRecommender(loss="squared_error", random_state=seed),
    ).fit(Xd_train[responders_train], yd_train[responders_train])

    p_respond = resp_model.predict_proba(Xd_test)[:, 1]
    expected_gift = p_respond * ask_model.predict(Xd_test)
    return _mail_profit_curve(yd_test.to_numpy(), expected_gift, cost)


def kdd_mail_profit_curve_val(seed: int, cost: float = 0.68) -> Dict[str, Any]:
    """Profit curve for bench_kdd_cost_aware_val's own fit (learning file's
    train split) and test split (cup98VAL, never touched during fitting)."""
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
        AskAmountRecommender(loss="squared_error", random_state=seed),
    ).fit(Xd_train[responders_train], yd_train[responders_train])

    val_donors = fetch_kdd98_val_donors()
    val_rfm = _kdd_rfm(_kdd_gift_log(val_donors))
    X_val = _kdd_feature_frame(val_donors, val_rfm)
    y_val = val_donors.set_index("CONTROLN")["TARGET_D"]

    p_respond = resp_model.predict_proba(X_val)[:, 1]
    expected_gift = p_respond * ask_model.predict(X_val)
    return _mail_profit_curve(y_val.to_numpy(), expected_gift, cost)


def bench_kdd_val_models(seed: int) -> List[Row]:
    """Response, lapse and ask models scored on KDD98's own held-out
    validation file (``cup98VAL`` + ``valtargt``, 96,367 donors never in the
    learning file), each fit exactly as in its learning-file bench: response
    and ask on the learning file's 55% train split, lapse on its periods
    before N-1. The rule sets are the same fixed lists. Reported next to the
    learning-file rows (dataset ``"cup98val"``), not replacing them."""
    lrn, val = fetch_kdd98_donors(), fetch_kdd98_val_donors()
    split_note = "fit on cup98LRN (same split as the kdd98 row), scored on cup98VAL+valtargt"
    rows: List[Row] = []

    X_lrn, idx_lrn = _kdd_response_frame(lrn)
    X_val, idx_val = _kdd_response_frame(val)
    y_lrn, y_val = idx_lrn["TARGET_B"].to_numpy(), idx_val["TARGET_B"].to_numpy()
    idx_train, _idx_v, _idx_t = _split_55_15_30(len(y_lrn), y_lrn, seed)
    rules = _kdd_response_rules(X_val, idx_val)
    for name, cls in (("DonorPropensityModel", DonorPropensityModel), ("MajorGiftClassifier", MajorGiftClassifier)):
        model = cls(random_state=seed).fit(X_lrn.iloc[idx_train].to_numpy(), y_lrn[idx_train])
        proba = model.predict_proba(X_val.to_numpy())[:, 1]
        rows += _classifier_rows("cup98val", name, y_val, proba, rules, split_note, bootstrap=True)

    panel_lrn, panel_val = _kdd_lapse_panel(lrn), _kdd_lapse_panel(val)
    last_period = panel_lrn["period"].max()
    train_p = panel_lrn[panel_lrn["period"] < last_period - 1]
    test_p = panel_val[panel_val["period"] == last_period]
    lapse = LapsePredictor(n_estimators=100, max_depth=10, random_state=seed).fit(
        train_p[["total", "n", "recent"]].to_numpy(), train_p["lapsed"].to_numpy()
    )
    score = lapse.predict_lapse_score(test_p[["total", "n", "recent"]].to_numpy()) / 100.0
    rows += _classifier_rows(
        "cup98val", "LapsePredictor", test_p["lapsed"].to_numpy(), score, _kdd_lapse_rules(test_p),
        split_note, bootstrap=True,
    )

    Xd_train, Xd_val_lrn, _Xd_t, yd_train, yd_val_lrn, _yd_t = _kdd_ask_design(lrn, _kdd_rfm(_kdd_gift_log(lrn)), seed)
    responders_train = yd_train > 0
    ask_model = make_pipeline(
        WealthScreeningImputer(wealth_cols=["WEALTH1", "WEALTH2", "INCOME"]),
        AskAmountRecommender(random_state=seed),
    ).fit(Xd_train[responders_train], yd_train[responders_train])
    Xd_val = _kdd_feature_frame(val, _kdd_rfm(_kdd_gift_log(val)))
    yd_val = val.set_index("CONTROLN")["TARGET_D"]
    responders_val = (yd_val > 0).to_numpy()
    pred = ask_model.predict(Xd_val[responders_val])
    y_true = yd_val[responders_val].to_numpy()
    last_gift = Xd_val.loc[responders_val, "LASTGIFT"].to_numpy()
    avg_gift = Xd_val.loc[responders_val, "AVGGIFT"].to_numpy()
    ask_rules = {
        "last gift": last_gift,
        "max(last gift, average gift)": np.maximum(last_gift, avg_gift),
        "median training gift": np.full_like(y_true, float(np.median(yd_train[responders_train]))),
    }
    best_name, best_score, _ = _best_ask_baseline(y_true, ask_rules)
    header = f"rule_set={best_name} (best of {len(ask_rules)}); split={split_note}; n_configs=1"
    rows += [
        Row(
            "cup98val", "AskAmountRecommender", "mae",
            mean_absolute_error(y_true, pred), mean_absolute_error(y_true, best_score),
            lower_is_better=True, note=header,
        ),
        Row(
            "cup98val", "AskAmountRecommender", "within25pct",
            _within_pct(pred, y_true), _within_pct(best_score, y_true), note=header,
        ),
    ]

    # E.12d adopt bar: a candidate must win on both KDD files, so the
    # winner picked on the LRN train/val split (bench_kdd_ask_relative's
    # candidate) is confirmed here on cup98val, not re-picked.
    winner, cand_pipe, featurize, val_mae = _pick_ask_candidate(Xd_train, yd_train, Xd_val_lrn, yd_val_lrn, seed)
    cand_pred = cand_pipe.predict(featurize(Xd_val)[responders_val])
    cand_header = (
        f"E.12d relative-target experiment, confirmed on cup98val; winner={winner} "
        f"(picked on the LRN val fold, MAE {val_mae[winner]:.2f}); "
        f"rule_set={best_name} (best of {len(ask_rules)}); split={split_note}; n_configs=2"
    )
    rows += [
        Row(
            "cup98val", "AskAmountRecommender (E.12d candidate)", "mae",
            mean_absolute_error(y_true, cand_pred), mean_absolute_error(y_true, best_score),
            lower_is_better=True, note=cand_header,
        ),
        Row(
            "cup98val", "AskAmountRecommender (E.12d candidate)", "within25pct",
            _within_pct(cand_pred, y_true), _within_pct(best_score, y_true), note=cand_header,
        ),
    ]
    return rows


# --------------------------------------------------------------------------- #
# UCI Blood Transfusion (opt-in download)
# --------------------------------------------------------------------------- #
BLOOD_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data"


def _fetch_blood() -> pd.DataFrame:
    """UCI Blood Transfusion Service Center (Yeh, Yang and Ting 2009; DOI
    10.24432/C5GS39; CC BY 4.0): 748 real repeat blood donors from one
    Taiwanese centre, four as-of features (months since last donation,
    number of donations, total volume, months since first donation) and
    whether each donated in March 2007. Cached in ``~/philanthropy_data``
    like the KDD98 files. Benchmark-only: blood, not money, so it tests
    response and lapse ranking on a real repeat-donor file, nothing else."""
    path = os.path.join(os.path.expanduser("~"), "philanthropy_data", "transfusion.data")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(BLOOD_URL, path)
    df = pd.read_csv(path)
    df.columns = ["recency", "frequency", "monetary", "tenure", "donated"]
    return df


def bench_blood(seeds: Sequence[int]) -> List[Row]:
    """Response (``DonorPropensityModel``, ``MajorGiftClassifier``) and lapse
    (``LapsePredictor``, label = did not donate) on the blood file, one
    stratified 70/30 split per seed. Rule sets are the synthetic ones
    restricted to what the file has: response against lifetime volume and
    the RFM cell score; lapse against months since last donation and fewest
    donations."""
    df = _fetch_blood()
    X = df[["recency", "frequency", "monetary", "tenure"]].to_numpy(dtype="float64")
    y = df["donated"].to_numpy()
    split = "stratified 70/30 per seed"
    seed_rows = []
    for seed in seeds:
        idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.30, random_state=seed, stratify=y)
        Xte = X[idx_test]
        response_rules = {
            "lifetime monetary": Xte[:, 2],
            "RFM cell score": _rfm_cell_score(Xte[:, 0], Xte[:, 1], Xte[:, 2]),
        }
        lapse_rules = {"years since last gift": Xte[:, 0], "fewest gifts": -Xte[:, 1]}
        rows: List[Row] = []
        for name, cls in (("DonorPropensityModel", DonorPropensityModel), ("MajorGiftClassifier", MajorGiftClassifier)):
            model = cls(random_state=seed).fit(X[idx_train], y[idx_train])
            rows += _classifier_rows("uci_blood", name, y[idx_test], model.predict_proba(Xte)[:, 1], response_rules, split)
        lapse = LapsePredictor(random_state=seed).fit(X[idx_train], 1 - y[idx_train])
        rows += _classifier_rows(
            "uci_blood", "LapsePredictor", 1 - y[idx_test], lapse.predict_lapse_score(Xte) / 100.0, lapse_rules, split,
        )
        seed_rows.append(rows)
    return _aggregate(seed_rows)


# --------------------------------------------------------------------------- #
# Karlan and List (2007) matching-grant experiment (openICPSR 113224; data
# CC BY 4.0, copyright American Economic Association 2007). Opt-in, local
# file. One letter, so it tests response, the amount given and, because the
# matching-grant offer was randomised, uplift; nothing multi-year.
# Fixed before any run: the 55/15/30 split (stratified on gave x matched),
# the features, and the rule sets below. One configuration each.
# --------------------------------------------------------------------------- #
KARLAN_LIST_SEED = 42
KARLAN_LIST_FEATURES = [
    "prior_gifts", "highest_previous_amount", "months_since_last_gift", "years_since_first_gift",
    "female", "couple", "red_state", "red_county",
]


def _karlan_list_split(path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    df = load_karlan_list(path)
    strat = df["gave"].to_numpy() * 2 + df["matched"].to_numpy()
    idx_train, idx_val, idx_test = _split_55_15_30(len(df), strat, KARLAN_LIST_SEED)
    return df, idx_train, idx_val, idx_test


def _karlan_list_rules(rows: pd.DataFrame) -> Dict[str, np.ndarray]:
    """The fixed response rule set: most recent, past spend (highest previous
    gift, the only amount the file has), most gifts, and the RFM cell score
    on those three."""
    recency = rows["months_since_last_gift"].to_numpy()
    return {
        "most recent": -recency,
        "past spend": rows["highest_previous_amount"].to_numpy(),
        "most gifts": rows["prior_gifts"].to_numpy(),
        "RFM cell score": _rfm_cell_score(
            recency, rows["prior_gifts"].to_numpy(), rows["highest_previous_amount"].to_numpy()
        ),
    }


def _karlan_list_design(df: pd.DataFrame, idx_train: np.ndarray) -> pd.DataFrame:
    """Features with NaN filled by the training rows' medians (frozen)."""
    medians = df.iloc[idx_train][KARLAN_LIST_FEATURES].median()
    return df[KARLAN_LIST_FEATURES].fillna(medians)


def bench_response_karlan_list(path: str) -> List[Row]:
    """DonorPropensityModel / MajorGiftClassifier vs the response rule set:
    who answers one fundraising letter (about 2 in 100 do)."""
    df, idx_train, _idx_val, idx_test = _karlan_list_split(path)
    X = _karlan_list_design(df, idx_train)
    y = df["gave"].to_numpy()
    rules = _karlan_list_rules(X.iloc[idx_test])
    rows: List[Row] = []
    for name, cls in (("DonorPropensityModel", DonorPropensityModel), ("MajorGiftClassifier", MajorGiftClassifier)):
        model = cls(random_state=KARLAN_LIST_SEED).fit(X.iloc[idx_train].to_numpy(), y[idx_train])
        proba = model.predict_proba(X.iloc[idx_test].to_numpy())[:, 1]
        rows += _classifier_rows("karlan_list", name, y[idx_test], proba, rules, KDD_SPLIT, bootstrap=True)
    return rows


def bench_ask_karlan_list(path: str) -> List[Row]:
    """AskAmountRecommender vs the ask rule set on donors who gave: highest
    previous gift (the file has no last or average gift) and the median
    training gift."""
    df, idx_train, _idx_val, idx_test = _karlan_list_split(path)
    gave = df["gave"].to_numpy() == 1
    tr, te = idx_train[gave[idx_train]], idx_test[gave[idx_test]]
    X = df[KARLAN_LIST_FEATURES]
    y = df["amount"].to_numpy()
    model = AskAmountRecommender(random_state=KARLAN_LIST_SEED).fit(X.iloc[tr].to_numpy(), y[tr])
    pred = model.predict(X.iloc[te].to_numpy())
    y_true = y[te]
    rules = {
        "highest previous gift": df["highest_previous_amount"].to_numpy()[te],
        "median training gift": np.full_like(y_true, float(np.median(y[tr]))),
    }
    best_name, best_score, _ = _best_ask_baseline(y_true, rules)
    header = f"rule_set={best_name} (best of {len(rules)}); split={KDD_SPLIT}; n_configs=1; n={len(te)}"
    mae_d = _bootstrap_diff_ci(len(y_true), lambda idx: (
        mean_absolute_error(y_true[idx], pred[idx]) - mean_absolute_error(y_true[idx], best_score[idx])
    ))
    within_d = _bootstrap_diff_ci(len(y_true), lambda idx: (
        _within_pct(pred[idx], y_true[idx]) - _within_pct(best_score[idx], y_true[idx])
    ))
    revenue_d = _bootstrap_diff_ci(len(y_true), lambda idx: (
        _revenue_top_share(y_true[idx], pred[idx]) - _revenue_top_share(y_true[idx], best_score[idx])
    ))
    return [
        Row(
            "karlan_list", "AskAmountRecommender", "revenue_top10pct",
            _revenue_top_share(y_true, pred), _revenue_top_share(y_true, best_score),
            note=header, diff_lo=revenue_d[0], diff_hi=revenue_d[1],
        ),
        Row(
            "karlan_list", "AskAmountRecommender", "mae",
            mean_absolute_error(y_true, pred), mean_absolute_error(y_true, best_score),
            lower_is_better=True, note=header, diff_lo=mae_d[0], diff_hi=mae_d[1],
        ),
        Row(
            "karlan_list", "AskAmountRecommender", "within25pct",
            _within_pct(pred, y_true), _within_pct(best_score, y_true),
            note=header, diff_lo=within_d[0], diff_hi=within_d[1],
        ),
    ]


def _uplift_in_top(y: np.ndarray, treated: np.ndarray, score: np.ndarray, frac: float) -> float:
    """Response rate with the matching-grant offer minus without it, among
    the top ``frac`` of donors by ``score``. ``nan`` if either arm is empty."""
    top = np.argsort(-score, kind="stable")[: max(1, int(round(len(score) * frac)))]
    t, c = y[top][treated[top] == 1], y[top][treated[top] == 0]
    if len(t) == 0 or len(c) == 0:
        return float("nan")
    return float(t.mean() - c.mean())


def bench_uplift_karlan_list(path: str) -> List[Row]:
    """UpliftTLearner vs "most recent" and "past spend": rank donors, keep
    the top 10% or 30%, and measure how much the randomised matching-grant
    offer raised their response rate (in proportion points). The rule with
    the larger lift at 30% is the baseline for both rows."""
    df, idx_train, _idx_val, idx_test = _karlan_list_split(path)
    X = _karlan_list_design(df, idx_train)
    y, treated = df["gave"].to_numpy(), df["matched"].to_numpy()
    model = UpliftTLearner(random_state=KARLAN_LIST_SEED).fit(
        X.iloc[idx_train].to_numpy(), y[idx_train], treated[idx_train]
    )
    score = model.predict_uplift_score(X.iloc[idx_test].to_numpy())
    yt, tt = y[idx_test], treated[idx_test]
    all_rules = _karlan_list_rules(X.iloc[idx_test])
    rules = {k: all_rules[k] for k in ("most recent", "past spend")}
    best = max(rules, key=lambda k: _uplift_in_top(yt, tt, rules[k], 0.30))
    header = (
        f"rule_set={best} (best of {len(rules)}); split={KDD_SPLIT}; n_configs=1; "
        f"everyone={_uplift_in_top(yt, tt, np.zeros(len(yt)), 1.0):.6f}"
    )
    rows = []
    for frac in (0.10, 0.30):
        d_lo, d_hi = _bootstrap_diff_ci(len(yt), lambda idx, f=frac: (
            _uplift_in_top(yt[idx], tt[idx], score[idx], f) - _uplift_in_top(yt[idx], tt[idx], rules[best][idx], f)
        ))
        rows.append(Row(
            "karlan_list", "UpliftTLearner", f"uplift_top{int(frac * 100)}pct",
            _uplift_in_top(yt, tt, score, frac), _uplift_in_top(yt, tt, rules[best], frac),
            note=header, diff_lo=d_lo, diff_hi=d_hi,
        ))
    return rows


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
        help="Also score cost-aware selection, response, lapse and ask on KDD98's own held-out "
        "cup98VAL+valtargt file (a second ~37 MB download; opt-in). No effect with --skip-kdd98.",
    )
    parser.add_argument(
        "--with-blood", action="store_true",
        help="Also score response and lapse on the UCI Blood Transfusion file (748 real repeat "
        "donors, ~12 KB download, CC BY 4.0; opt-in).",
    )
    parser.add_argument("--fast", action="store_true", help="One seed and a small synthetic panel: a smoke run, not a benchmark.")
    parser.add_argument(
        "--donorschoose-path", type=str, default=None,
        help="Path to a user-obtained ICPSR 37898 DS0001 Donations file (.tsv or .dta); "
        "skipped entirely when not given.",
    )
    parser.add_argument(
        "--psid-data", type=str, default=None,
        help="Path to a user-obtained PSID Data Center fixed-width extract (.txt). Requires "
        "--psid-do too; skipped entirely when either is missing.",
    )
    parser.add_argument(
        "--psid-do", type=str, default=None,
        help="Path to the PSID extract's accompanying Stata .do file. Requires --psid-data too.",
    )
    parser.add_argument(
        "--karlan-list-path", type=str, default=None,
        help="Path to a user-obtained AERtables1-5.dta from openICPSR 113224 (Karlan and List 2007); "
        "skipped entirely when not given.",
    )
    parser.add_argument("--out", type=str, default=None, help="Path prefix; also writes <out>.json and <out>.csv.")
    args = parser.parse_args()

    seeds: Sequence[int] = (DEFAULT_SEEDS[0],) if args.fast else DEFAULT_SEEDS
    n_donors = 300 if args.fast else 3000
    n_years = 5 if args.fast else 7

    start = time.time()
    rows: List[Row] = []
    rows += bench_response(seeds, n_donors, n_years)
    rows += bench_lapse(seeds, n_donors, n_years)
    # E.12b feature-parity probe: the rules above see 5+ as-of columns
    # (years_since_last, streak, ...) while the models above see only 3.
    # This second column re-runs both on the same full 8 columns so the
    # comparison is no longer lopsided; reported next to the default row,
    # not replacing it (E.12e item 2).
    rows += bench_response(seeds, n_donors, n_years, feature_cols=PARITY_FEATURES, dataset="synthetic_panel_full_features")
    rows += bench_lapse(seeds, n_donors, n_years, feature_cols=PARITY_FEATURES, dataset="synthetic_panel_full_features")
    rows += bench_ask(seeds, n_donors, n_years)
    rows += bench_upgrade(seeds, n_donors, n_years)
    rows += bench_planned_giving(seeds, n_donors, n_years)
    # E.9: 10 seeds at each requested level, not just 5 seeds at 90%, since a
    # single-seed 88% vs 90% gap on its own does not tell a bug from noise.
    gift_interval_seeds = (GIFT_INTERVAL_SEEDS[0],) if args.fast else GIFT_INTERVAL_SEEDS
    for alpha in (0.2, 0.1, 0.05):
        rows += bench_gift_interval(gift_interval_seeds, n_donors, n_years, alpha=alpha)
    rows += bench_forecast(seeds, n_donors, n_years)

    if not args.skip_kdd98:
        rows += bench_kdd_upgrade(KDD_SEED)
        rows += bench_kdd_response(KDD_SEED)
        rows += bench_kdd_lapse(KDD_SEED)
        rows += bench_kdd_lapse_retention(KDD_SEED)
        rows += bench_kdd_ask(KDD_SEED)
        rows += bench_kdd_ask_relative(KDD_SEED)
        rows += bench_kdd_cost_aware(KDD_SEED)
        if args.with_cup98val:
            rows += bench_kdd_lapse_backend_trial(KDD_SEED)
            rows += bench_kdd_cost_aware_val(KDD_SEED)
            rows += bench_kdd_val_models(KDD_SEED)

    if args.with_blood:
        rows += bench_blood(seeds)

    if args.donorschoose_path:
        for momentum in (False, True):
            rows += bench_upgrade_donorschoose(args.donorschoose_path, include_momentum=momentum)
            rows += bench_lapse_donorschoose(args.donorschoose_path, include_momentum=momentum)
            rows += bench_lapse_donorschoose_retention(args.donorschoose_path, include_momentum=momentum)
            rows += bench_ask_donorschoose(args.donorschoose_path, include_momentum=momentum)

    if args.psid_data and args.psid_do:
        for momentum in (False, True):
            rows += bench_upgrade_psid(args.psid_data, args.psid_do, include_momentum=momentum)
            rows += bench_lapse_psid(args.psid_data, args.psid_do, include_momentum=momentum)
            rows += bench_lapse_psid_retention(args.psid_data, args.psid_do, include_momentum=momentum)
            rows += bench_ask_psid(args.psid_data, args.psid_do, include_momentum=momentum)

    if args.karlan_list_path:
        rows += bench_response_karlan_list(args.karlan_list_path)
        rows += bench_ask_karlan_list(args.karlan_list_path)
        rows += bench_uplift_karlan_list(args.karlan_list_path)

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
