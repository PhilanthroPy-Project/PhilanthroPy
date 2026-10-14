"""
philanthropy.model_selection._rule_check
========================================
Does the model beat the simple rule on *this* file?

``compare_to_rule`` runs the same test the Results pages run, on the
caller's own gift log: build leakage-safe donor x fiscal-year snapshots,
train on every labelled fiscal year but the last, score the last one, and
compare the model's top-10% hit rate (mean absolute error for ask amounts)
against the best of a short list of rules fixed in code before any run.
The verdict is read off a paired bootstrap of ``model - rule`` on the same
donors, never off one point, and the rule wins every tie: it is free and
easy to explain.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from philanthropy.ingest._snapshots import build_snapshots
from philanthropy.ingest._upgrade_snapshots import _fy_end, _prepare_gifts
from philanthropy.utils._label_floor import check_label_floor
from philanthropy.utils._validation import fiscal_year_for, validate_fiscal_year_start

__all__ = ["compare_to_rule"]

TASKS = ("upgrade", "lapse", "ask")

_KIND = {
    "upgrade": "upgrade", "lapse": "lapse", "ask": "next_amount",
}
# Lapse is asked about donors with at least two giving years, the donors a
# retention program can act on, as on the Results pages.
_MIN_YEARS_GIVEN = {"upgrade": 1, "lapse": 2, "ask": 1}
_LABEL_FLOOR_TASK = {"upgrade": "major_gift", "lapse": "lapse"}
_TOP_FRACTION = 0.10


# The rule sets, fixed here before any run (the same ones
# scripts/benchmark_models_vs_baselines.py scores the Results pages
# against). Each maps a name to a score over the test rows; higher means
# "more likely" for the three classifiers and is the predicted amount for ask.
def _rules(task: str, test: pd.DataFrame, y_train: np.ndarray) -> Dict[str, np.ndarray]:
    if task == "upgrade":
        return {
            "this-year total": test["period_total"].to_numpy(),
            "previous-year total plus this-year growth": (test["period_total"] + test["period_trend"]).to_numpy(),
            "largest single gift in band": test["largest_gift"].to_numpy(),
        }
    if task == "lapse":
        return {
            "months since last gift": test["months_since_last_gift"].to_numpy(),
            "shortest giving streak": -test["consecutive_periods_given"].to_numpy(),
            "declining trend": -test["period_trend"].to_numpy(),
        }
    last_total = test["period_total"].to_numpy()
    return {
        "last period's total": last_total,
        "max(last period's total, average training next-gift)": np.maximum(last_total, float(y_train.mean())),
        "median training next-gift": np.full(len(test), float(np.median(y_train))),
    }


def _top_rate(y_true: np.ndarray, score: np.ndarray) -> float:
    """Positive rate among the top 10% of rows ranked by ``score``."""
    n = max(1, int(round(len(y_true) * _TOP_FRACTION)))
    return float(y_true[np.argsort(-score, kind="stable")[:n]].mean())


def _mae(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(y_true - pred).mean())


def _bootstrap_difference(
    y: np.ndarray, model: np.ndarray, rule: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int, random_state: int,
) -> Tuple[float, float]:
    """95% interval on ``metric(model) - metric(rule)`` over resamples of the
    same test rows, so both sides see the same donors in every draw."""
    rng = np.random.default_rng(random_state)
    n = len(y)
    diffs = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[b] = metric(y[idx], model[idx]) - metric(y[idx], rule[idx])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def compare_to_rule(
    gifts: Union[Iterable[Mapping], pd.DataFrame],
    *,
    task: str,
    as_of: Optional[Union[str, pd.Timestamp]] = None,
    fiscal_year_start: int = 7,
    threshold: float = 1000.0,
    band: Tuple[float, float] = (100.0, 999.0),
    n_bootstrap: int = 1000,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Decide, on one gift log, whether the model or the simple rule should
    rank the donors for ``task``.

    The check is the one the Results pages run, applied to the caller's own
    file: :func:`~philanthropy.ingest.build_snapshots` builds one row per
    ``(donor, fiscal year T)`` with features from gifts dated at or before
    the end of T and a label read from T+1. Every labelled fiscal year but
    the last trains the model; the last one is the held-out test year. On
    that year the model is scored against the best of a short list of rules
    fixed in code before any run, on the top-10% hit rate (ask: mean
    absolute error), and the verdict is read off a 95% bootstrap interval
    of ``model - rule`` over the same test donors. Only a whole interval on
    the model's side picks the model; a tie goes to the rule, which is free
    and easy to explain.

    Parameters
    ----------
    gifts : iterable of mapping, or DataFrame
        Gift-level rows with ``donor_id``, ``gift_date`` and ``gift_amount``.
    task : {"upgrade", "lapse", "ask"}
        Which question: will the donor reach ``threshold`` next year (from
        inside ``band``), stop giving, or how much will they give next year
        given that they do.
    as_of : str or Timestamp, optional
        Gifts after this date are ignored, and only fiscal years whose
        following year is complete by ``as_of`` can carry a label, so a
        partial current year never masquerades as "gave nothing". Defaults
        to the latest gift date in ``gifts``.
    fiscal_year_start : int, default=7
        Month (1-12) the fiscal year begins.
    threshold, band : float, (float, float)
        The upgrade question's leadership level and candidate band; ignored
        by the other tasks.
    n_bootstrap : int, default=1000
        Resamples behind the interval.
    random_state : int, default=0
        Seed for the model fit and the bootstrap.

    Returns
    -------
    dict
        ``task``; ``method`` (``"model"`` or ``"rule"``); ``verdict``, one of
        ``"Beats the rule"``, ``"About the same as the rule"``,
        ``"Loses to the rule"``, ``"not enough labels"`` or ``"can't test
        (reason)"``, the words the Results pages use; ``rule``, the name of
        the best rule on the test year; ``n_rules_tried``;
        ``test_fiscal_year``, ``n_train``, ``n_test``; ``metric``
        (``"top10pct_hit_rate"`` or ``"mae"``); ``model_value`` and
        ``rule_value`` on the test year (``None`` for the model when it was
        not fit); ``difference_ci``, the ``(low, high)`` interval on
        ``model - rule`` (``None`` when the model was not fit); and
        ``label_floor``, :func:`~philanthropy.utils.check_label_floor`'s
        answer on the training labels (``None`` for ask, which has no
        floor and instead needs 20 training and 5 test rows).

    Raises
    ------
    ValueError
        If ``task`` is unknown or ``fiscal_year_start`` is not 1-12.
    KeyError
        If ``gifts`` is missing a required column.

    Examples
    --------
    >>> from philanthropy.datasets import make_donor_panel
    >>> from philanthropy.model_selection import compare_to_rule
    >>> gifts = make_donor_panel(n_donors=400, n_years=5, random_state=0)["gifts"]
    >>> out = compare_to_rule(gifts, task="lapse", n_bootstrap=50)
    >>> out["method"] in ("model", "rule")
    True
    >>> out["n_rules_tried"]
    3
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}; got {task!r}.")
    validate_fiscal_year_start(fiscal_year_start)

    df = _prepare_gifts(gifts, fiscal_year_start)[0]
    if as_of is None:
        as_of_ts = df["_date"].max() if not df.empty else pd.NaT
    else:
        as_of_ts = pd.Timestamp(as_of)
        df = df[df["_date"] <= as_of_ts]

    report: Dict[str, Any] = {
        "task": task, "method": "rule", "verdict": "", "rule": None, "n_rules_tried": 0,
        "test_fiscal_year": None, "n_train": 0, "n_test": 0,
        "metric": "mae" if task == "ask" else "top10pct_hit_rate",
        "model_value": None, "rule_value": None, "difference_ci": None, "label_floor": None,
    }
    if df.empty:
        report["verdict"] = "can't test (no usable gift rows)"
        return report

    # A fiscal year can only label the one before it once it is complete.
    fy_as_of = fiscal_year_for(int(as_of_ts.year), int(as_of_ts.month), fiscal_year_start)
    last_complete = fy_as_of if as_of_ts >= _fy_end(fy_as_of, fiscal_year_start) else fy_as_of - 1
    snap = build_snapshots(
        df[["donor_id", "_date", "_amount"]].rename(columns={"_date": "gift_date", "_amount": "gift_amount"}),
        kind=_KIND[task],
        fiscal_years=range(int(df["_fy"].min()), last_complete),
        threshold=threshold, band=band,
        min_years_given=_MIN_YEARS_GIVEN[task], fiscal_year_start=fiscal_year_start,
    )
    if snap.empty or snap["period"].nunique() < 2:
        report["verdict"] = "can't test (fewer than two labelled fiscal years)"
        return report

    test_fy = int(snap["period"].max())
    train, test = snap[snap["period"] < test_fy], snap[snap["period"] == test_fy]
    feature_cols = [
        c for c in snap.columns
        if c not in ("period", "target") and pd.api.types.is_numeric_dtype(snap[c])
    ]
    y_train = train["target"].to_numpy("float64")
    y_test = test["target"].to_numpy("float64")
    report.update({"test_fiscal_year": test_fy, "n_train": int(len(train)), "n_test": int(len(test))})

    metric = _mae if task == "ask" else _top_rate
    rules = _rules(task, test, y_train)
    scored = {name: metric(y_test, score) for name, score in rules.items()}
    best = (min if task == "ask" else max)(scored, key=lambda name: scored[name])
    report.update({"rule": best, "n_rules_tried": len(rules), "rule_value": scored[best]})

    if task == "ask":
        enough = len(train) >= 20 and len(test) >= 5
    else:
        report["label_floor"] = check_label_floor(y_train, _LABEL_FLOOR_TASK[task])
        enough = report["label_floor"] == "run"
    if not enough:
        report["verdict"] = "not enough labels"
        return report

    # Imported here: philanthropy.models already imports this subpackage
    # for FiscalYearGroupedSplitter, so a module-level import would cycle.
    from philanthropy.models import (
        AskAmountRecommender, LapsePredictor, MajorGiftClassifier,
    )

    X_train = train[feature_cols].fillna(0.0).to_numpy("float64")
    X_test = test[feature_cols].fillna(0.0).to_numpy("float64")
    if task == "ask":
        model_score = AskAmountRecommender(random_state=random_state).fit(X_train, y_train).predict(X_test)
    elif task == "lapse":
        model_score = LapsePredictor(random_state=random_state).fit(X_train, y_train).predict_lapse_score(X_test)
    else:
        model_score = MajorGiftClassifier(random_state=random_state).fit(X_train, y_train).predict_proba(X_test)[:, 1]
    model_score = np.asarray(model_score, dtype="float64")

    lo, hi = _bootstrap_difference(y_test, model_score, rules[best], metric, n_bootstrap, random_state)
    model_wins = hi < 0 if task == "ask" else lo > 0
    rule_wins = lo > 0 if task == "ask" else hi < 0
    report.update({
        "model_value": metric(y_test, model_score),
        "difference_ci": (lo, hi),
        "verdict": (
            "Beats the rule" if model_wins
            else "Loses to the rule" if rule_wins
            else "About the same as the rule"
        ),
        "method": "model" if model_wins else "rule",
    })
    return report
