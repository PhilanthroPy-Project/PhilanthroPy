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

The KDD Cup 1998 section downloads ``cup98lrn.zip`` (~36 MB) to
``~/philanthropy_data`` on first use (see ``fetch_kdd98_donors``); pass
``--skip-kdd98`` to stay offline entirely.

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

from philanthropy.datasets import fetch_kdd98_donors, make_donor_panel
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

CSV_FIELDS = ("dataset", "model", "metric", "value", "baseline", "lo", "hi", "n_seeds", "verdict", "note")


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
    baseline_score: np.ndarray,
    baseline_name: str,
) -> List[Row]:
    y_true = np.asarray(y_true)
    rows = []
    for frac in (0.01, 0.05, 0.10):
        m_rate, n = _topn_rate(y_true, score, frac)
        b_rate, _ = _topn_rate(y_true, baseline_score, frac)
        rows.append(
            Row(
                dataset, model_name, f"top{int(frac * 100)}pct_hit_rate",
                m_rate, b_rate, note=f"n={n}; baseline={baseline_name}",
            )
        )
    has_both_classes = len(np.unique(y_true)) > 1
    auc = roc_auc_score(y_true, score) if has_both_classes else float("nan")
    b_auc = roc_auc_score(y_true, baseline_score) if has_both_classes else float("nan")
    rows.append(Row(dataset, model_name, "roc_auc", auc, b_auc, note=f"baseline={baseline_name}"))
    ap = average_precision_score(y_true, score) if has_both_classes else float("nan")
    rows.append(Row(dataset, model_name, "average_precision", ap, note="no baseline; base-rate dependent"))
    rows.append(
        Row(
            dataset, model_name, "decile_calibration_gap",
            _decile_calibration_gap(y_true, score), lower_is_better=True,
            note="mean |mean predicted - mean actual| across probability deciles",
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

    rows = []
    cum_total = np.zeros(len(donor_ids))
    cum_n = np.zeros(len(donor_ids))
    for i, fy in enumerate(years[:-1]):
        recent = amount[fy].to_numpy()
        cum_total = cum_total + recent
        cum_n = cum_n + gave[fy].to_numpy()
        next_fy = years[i + 1]
        y_response = gave[next_fy].to_numpy().astype(int)
        y_amount = np.where(y_response == 1, amount[next_fy].to_numpy(), np.nan)
        rows.append(
            pd.DataFrame(
                {
                    "donor_id": donor_ids, "fy": fy,
                    "total": cum_total.copy(), "n": cum_n.copy(), "recent": recent,
                    "y_response": y_response, "y_amount": y_amount,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _train_test_periods(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_fy = panel["fy"].max()
    return panel[panel["fy"] < test_fy], panel[panel["fy"] == test_fy]


def bench_response(seeds: Sequence[int], n_donors: int, n_years: int) -> List[Row]:
    """DonorPropensityModel / MajorGiftClassifier vs an RFM/monetary rule."""
    seed_rows = []
    for seed in seeds:
        train, test = _train_test_periods(_period_panel(n_donors, n_years, seed))
        Xtr = train[["total", "n", "recent"]].to_numpy()
        Xte = test[["total", "n", "recent"]].to_numpy()
        ytr, yte = train["y_response"].to_numpy(), test["y_response"].to_numpy()
        baseline_score = test["total"].to_numpy()
        rows: List[Row] = []
        for name, cls in (("DonorPropensityModel", DonorPropensityModel), ("MajorGiftClassifier", MajorGiftClassifier)):
            model = cls(random_state=seed).fit(Xtr, ytr)
            proba = model.predict_proba(Xte)[:, 1]
            rows += _classifier_rows(
                "synthetic_panel", name, yte, proba, baseline_score,
                "RFM/monetary rule (highest cumulative giving)",
            )
        seed_rows.append(rows)
    return _aggregate(seed_rows)


def bench_lapse(seeds: Sequence[int], n_donors: int, n_years: int) -> List[Row]:
    """LapsePredictor vs the "gave nothing last period" rule."""
    seed_rows = []
    for seed in seeds:
        train, test = _train_test_periods(_period_panel(n_donors, n_years, seed))
        Xtr, Xte = train[["total", "n", "recent"]].to_numpy(), test[["total", "n", "recent"]].to_numpy()
        ytr, yte = 1 - train["y_response"].to_numpy(), 1 - test["y_response"].to_numpy()
        baseline_score = -test["recent"].to_numpy()
        model = LapsePredictor(random_state=seed).fit(Xtr, ytr)
        score = model.predict_lapse_score(Xte) / 100.0  # predict_lapse_score is 0-100, calibration needs 0-1
        seed_rows.append(
            _classifier_rows("synthetic_panel", "LapsePredictor", yte, score, baseline_score, "gave nothing last period")
        )
    return _aggregate(seed_rows)


def bench_ask(seeds: Sequence[int], n_donors: int, n_years: int) -> List[Row]:
    """AskAmountRecommender vs "last gift" and "median" baselines."""
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
        median_gift = np.full_like(yte, float(np.median(ytr)))

        seed_rows.append(
            [
                Row(
                    "synthetic_panel", "AskAmountRecommender", "mae",
                    mean_absolute_error(yte, pred), mean_absolute_error(yte, last_gift),
                    lower_is_better=True, note="baseline=last gift",
                ),
                Row(
                    "synthetic_panel", "AskAmountRecommender", "mae_vs_median",
                    mean_absolute_error(yte, pred), mean_absolute_error(yte, median_gift),
                    lower_is_better=True, note="baseline=median training gift",
                ),
                Row(
                    "synthetic_panel", "AskAmountRecommender", "within25pct",
                    _within_pct(pred, yte), _within_pct(last_gift, yte),
                    note="baseline=last gift",
                ),
            ]
        )
    return _aggregate(seed_rows)


def bench_upgrade(
    seeds: Sequence[int], n_donors: int, n_years: int,
    threshold: float = 1000.0, band: Tuple[float, float] = (100.0, 999.0),
) -> List[Row]:
    """Upgrade model (build_upgrade_snapshots + MajorGiftClassifier) vs the
    naive "highest current-year total" rule."""
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
        baseline_score = snaps["fy_total"].to_numpy()[test_idx]
        seed_rows.append(
            _classifier_rows(
                "synthetic_panel", "upgrade_model (MajorGiftClassifier)",
                y[test_idx], proba, baseline_score, "highest current-year total",
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
                    "so this reuses the giving-response label as a coverage check only",
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


def _kdd_ask_design(
    donors: pd.DataFrame, rfm: pd.DataFrame, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    base_cols = [
        "AGE", "INCOME", "WEALTH1", "WEALTH2", "NUMCHLD", "RAMNTALL", "NGIFTALL",
        "LASTGIFT", "AVGGIFT", "MAXRAMNT", "MINRAMNT", "TIMELAG",
    ]
    donors_idx = donors.set_index("CONTROLN")
    X_full = donors_idx[base_cols].join(rfm.add_prefix("rfm_")).assign(
        HOMEOWNER=(donors_idx["HOMEOWNR"] == "H").astype(int)
    )
    y_amt = donors_idx["TARGET_D"]
    y_resp = donors_idx["TARGET_B"]
    return train_test_split(X_full, y_amt, test_size=0.3, stratify=y_resp, random_state=seed)


def bench_kdd_upgrade(seed: int, threshold: float = 50.0, band: Tuple[float, float] = (5.0, 49.0)) -> List[Row]:
    """Upgrade model on KDD98, threshold rescaled from the $1000/$100-999
    defaults: this file's per-donor annual giving tops out far lower than a
    major-gift program's, so $50/$5-49 keeps the same shape (threshold is
    roughly the 93rd percentile of per-donor annual giving on this file)."""
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
    fy = snaps["fiscal_year"].to_numpy()
    splitter = FiscalYearGroupedSplitter(n_splits=1, drop_repeat_donors=False)
    train_idx, test_idx = list(splitter.split(X, groups=fy))[-1]

    model = MajorGiftClassifier(random_state=seed).fit(X[train_idx], y[train_idx])
    proba = model.predict_proba(X[test_idx])[:, 1]
    baseline_score = snaps["fy_total"].to_numpy()[test_idx]
    return _classifier_rows(
        "kdd98", "upgrade_model (MajorGiftClassifier)", y[test_idx], proba, baseline_score,
        f"highest current-year total (threshold=${threshold:.0f}, band={band}, "
        "rescaled from the $1000/$100-999 defaults for this file's gift sizes)",
    )


def bench_kdd_response(seed: int) -> List[Row]:
    """DonorPropensityModel / MajorGiftClassifier vs an RFM/monetary rule, scoring TARGET_B."""
    donors = fetch_kdd98_donors()
    rfm = _kdd_rfm(_kdd_gift_log(donors))
    donors_idx = donors.set_index("CONTROLN")
    X_rfm = rfm.reindex(donors_idx.index)
    X_rfm["recency"] = X_rfm["recency"].fillna(X_rfm["recency"].max() + 1)
    X_rfm[["frequency", "monetary", "tenure"]] = X_rfm[["frequency", "monetary", "tenure"]].fillna(0.0)
    y = donors_idx["TARGET_B"].to_numpy()

    Xtr, Xte, ytr, yte = train_test_split(X_rfm, y, test_size=0.3, stratify=y, random_state=seed)
    rows: List[Row] = []
    for name, cls in (("DonorPropensityModel", DonorPropensityModel), ("MajorGiftClassifier", MajorGiftClassifier)):
        model = cls(random_state=seed).fit(Xtr.to_numpy(), ytr)
        proba = model.predict_proba(Xte.to_numpy())[:, 1]
        baseline_score = Xte["monetary"].to_numpy()
        rows += _classifier_rows("kdd98", name, yte, proba, baseline_score, "RFM rule (highest monetary)")
    return rows


def bench_kdd_lapse(seed: int) -> List[Row]:
    """LapsePredictor vs "gave nothing last period", over the 22-promotion history."""
    donors = fetch_kdd98_donors()
    hist_promos = list(range(24, 2, -1))
    ramnt = donors[[f"RAMNT_{i}" for i in hist_promos]].fillna(0.0).to_numpy()
    gave = ramnt > 0
    n_periods = ramnt.shape[1]
    target_b = donors["TARGET_B"].to_numpy()

    cum_total, cum_n = np.zeros(len(donors)), np.zeros(len(donors))
    periods = []
    for p in range(n_periods):
        cum_total, cum_n = cum_total + ramnt[:, p], cum_n + gave[:, p]
        next_gave = gave[:, p + 1] if p < n_periods - 1 else (target_b == 1)
        periods.append(
            pd.DataFrame(
                dict(period=p, total=cum_total.copy(), n=cum_n.copy(), recent=ramnt[:, p], lapsed=(~next_gave).astype(int))
            )
        )
    panel = pd.concat(periods, ignore_index=True)
    last_period = panel["period"].max()
    train_p, test_p = panel[panel["period"] < last_period], panel[panel["period"] == last_period]

    model = LapsePredictor(n_estimators=100, max_depth=10, random_state=seed).fit(
        train_p[["total", "n", "recent"]].to_numpy(), train_p["lapsed"].to_numpy()
    )
    score = model.predict_lapse_score(test_p[["total", "n", "recent"]].to_numpy()) / 100.0
    y = test_p["lapsed"].to_numpy()
    baseline_score = -test_p["recent"].to_numpy()
    return _classifier_rows("kdd98", "LapsePredictor", y, score, baseline_score, "gave nothing last period")


def bench_kdd_ask(seed: int) -> List[Row]:
    """AskAmountRecommender vs "last gift" (LASTGIFT) and "average gift" (AVGGIFT)."""
    donors = fetch_kdd98_donors()
    rfm = _kdd_rfm(_kdd_gift_log(donors))
    Xd_train, Xd_test, yd_train, yd_test = _kdd_ask_design(donors, rfm, seed)
    responders_train, responders_test = yd_train > 0, yd_test > 0

    ask_model = make_pipeline(
        WealthScreeningImputer(wealth_cols=["WEALTH1", "WEALTH2", "INCOME"]),
        AskAmountRecommender(random_state=seed),
    ).fit(Xd_train[responders_train], yd_train[responders_train])
    pred = ask_model.predict(Xd_test[responders_test])

    y_true = yd_test[responders_test].to_numpy()
    last_gift = Xd_test.loc[responders_test, "LASTGIFT"].to_numpy()
    avg_gift = Xd_test.loc[responders_test, "AVGGIFT"].to_numpy()

    return [
        Row(
            "kdd98", "AskAmountRecommender", "mae",
            mean_absolute_error(y_true, pred), mean_absolute_error(y_true, last_gift),
            lower_is_better=True, note="baseline=last gift (LASTGIFT)",
        ),
        Row(
            "kdd98", "AskAmountRecommender", "mae_vs_average_gift",
            mean_absolute_error(y_true, pred), mean_absolute_error(y_true, avg_gift),
            lower_is_better=True, note="baseline=average gift (AVGGIFT)",
        ),
        Row(
            "kdd98", "AskAmountRecommender", "within25pct",
            _within_pct(pred, y_true), _within_pct(last_gift, y_true),
            note="baseline=last gift",
        ),
    ]


def bench_kdd_cost_aware(seed: int, cost: float = 0.68) -> List[Row]:
    """Mail if E[gift] > cost (the KDD Cup 1998 competition's own rule) vs mailing everyone."""
    donors = fetch_kdd98_donors()
    rfm = _kdd_rfm(_kdd_gift_log(donors))
    Xd_train, Xd_test, yd_train, yd_test = _kdd_ask_design(donors, rfm, seed)
    responders_train = yd_train > 0

    resp_model = make_pipeline(
        WealthScreeningImputer(wealth_cols=["WEALTH1", "WEALTH2", "INCOME"]),
        MajorGiftClassifier(random_state=seed),
    ).fit(Xd_train, (yd_train > 0).astype(int))
    p_respond = resp_model.predict_proba(Xd_test)[:, 1]

    ask_model = make_pipeline(
        WealthScreeningImputer(wealth_cols=["WEALTH1", "WEALTH2", "INCOME"]),
        AskAmountRecommender(random_state=seed),
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

    runtime = time.time() - start
    _print_table(rows)
    print(f"\nRuntime: {runtime:.1f}s")

    if args.out:
        _write_json(rows, args.out + ".json")
        _write_csv(rows, args.out + ".csv")
        print(f"Wrote {args.out}.json and {args.out}.csv")


if __name__ == "__main__":
    main()
