"""
scripts/make_results_pages.py
==============================
Generates every number and chart the docs "Results" section cites.

It never types a result by hand: it imports and calls the ``bench_*``
functions in ``scripts/benchmark_models_vs_baselines.py`` (the committed,
reproducible model-vs-baseline benchmark), adds a "random pick" baseline for
each hit-rate comparison (the expected hit rate of picking that many donors
uniformly at random, which is just the fold's own positive rate, computed
from the same train/test construction), and runs one worked example of
``score_upgrade_prospects`` on ``make_donor_panel(random_state=0)`` for the
$1K-upgrade page's donor counts and decile chart.

Writes ``docs/assets/results/results.json`` (every number, for the docs pages
to cite) and one PNG chart per results page into ``docs/assets/results/``.

Run:

    python scripts/make_results_pages.py                # synthetic only
    python scripts/make_results_pages.py --with-kdd98    # + KDD Cup 1998 (downloads ~36MB)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_models_vs_baselines as bm  # noqa: E402

from philanthropy.datasets import make_donor_panel  # noqa: E402
from philanthropy.models import PlannedGivingIntentScorer, score_upgrade_prospects  # noqa: E402

OUT_DIR = ROOT / "docs" / "assets" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = bm.DEFAULT_SEEDS
N_DONORS, N_YEARS = 3000, 7

# Categorical palette slots 1/2/3 (blue/orange/aqua) from the dataviz skill's
# validated default palette: model, simple rule, random.
COLOR_MODEL = "#2a78d6"
COLOR_RULE = "#eb6834"
COLOR_RANDOM = "#898781"  # muted ink: "random" is a floor, not a series
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"


def _bar_chart(
    path: Path,
    groups: List[str],
    series: Dict[str, List[float]],
    colors: Dict[str, str],
    ylabel: str,
    title: str,
) -> None:
    """Grouped bar chart: one group of bars per `groups` entry, one bar per
    series. Thin bars, direct value labels, recessive gridlines, a legend."""
    n_groups, n_series = len(groups), len(series)
    width = 0.8 / n_series
    x = np.arange(n_groups)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    for i, (name, values) in enumerate(series.items()):
        offset = (i - (n_series - 1) / 2) * width
        bars = ax.bar(
            x + offset, values, width=width * 0.9, label=name,
            color=colors[name], edgecolor="none",
        )
        for b, v in zip(bars, values):
            ax.text(
                b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.0f}",
                ha="center", va="bottom", fontsize=9, color=INK_PRIMARY,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _row(rows, model, metric):
    for r in rows:
        if r.model == model and r.metric == metric:
            return r
    return None


# --------------------------------------------------------------------------- #
# Random-pick baselines (expected hit rate of a uniform-random pick = the
# fold's own positive rate). Reuses the exact train/test construction the
# benchmark script uses, so this is not a separately-typed number.
# --------------------------------------------------------------------------- #
def random_rate_response() -> float:
    rates = []
    for seed in SEEDS:
        _, test = bm._train_test_periods(bm._period_panel(N_DONORS, N_YEARS, seed))
        rates.append(test["y_response"].mean())
    return float(np.mean(rates))


def random_rate_lapse() -> float:
    rates = []
    for seed in SEEDS:
        _, test = bm._train_test_periods(bm._period_panel(N_DONORS, N_YEARS, seed))
        rates.append(1.0 - test["y_response"].mean())
    return float(np.mean(rates))


def random_rate_upgrade(threshold: float = 1000.0, band=(100.0, 999.0)) -> float:
    rates = []
    for seed in SEEDS:
        panel = bm.make_donor_panel(n_donors=N_DONORS, n_years=N_YEARS, random_state=seed)
        years = sorted(panel["gifts"]["fiscal_year"].unique())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            snaps = bm.build_upgrade_snapshots(
                panel["gifts"], fiscal_years=years[:-1], threshold=threshold, band=band,
            )
        if snaps.empty or snaps["fiscal_year"].nunique() < 2:
            continue
        fy = snaps["fiscal_year"].to_numpy()
        y = snaps["target"].to_numpy()
        splitter = bm.FiscalYearGroupedSplitter(n_splits=1, drop_repeat_donors=False)
        train_idx, test_idx = list(splitter.split(snaps[["fy_total"]].to_numpy(), groups=fy))[-1]
        rates.append(y[test_idx].mean())
    return float(np.mean(rates))


def planned_giving_hit_rates() -> Dict[str, Any]:
    """PlannedGivingIntentScorer has no bequest-intent label in the sample
    data (see benchmark_models_vs_baselines.bench_planned_giving's own
    docstring): this reuses the giving-response label as the nearest
    available proxy, purely as a coverage check against random picking, not
    a claim about real planned-giving intent."""
    fracs = (0.01, 0.05, 0.10)
    model_rates: Dict[float, list] = {f: [] for f in fracs}
    random_rates: Dict[float, list] = {f: [] for f in fracs}
    n_test = []
    for seed in SEEDS:
        train, test = bm._train_test_periods(bm._period_panel(N_DONORS, N_YEARS, seed))
        Xtr = train[["total", "n", "recent"]].to_numpy()
        Xte = test[["total", "n", "recent"]].to_numpy()
        ytr, yte = train["y_response"].to_numpy(), test["y_response"].to_numpy()
        model = PlannedGivingIntentScorer(random_state=seed).fit(Xtr, ytr)
        score = model.predict_intent_score(Xte)
        n_test.append(len(yte))
        for f in fracs:
            rate, _ = bm._topn_rate(yte, score, f)
            model_rates[f].append(rate)
            random_rates[f].append(yte.mean())
    return {
        "n_test": int(np.mean(n_test)),
        "model": {str(f): float(np.mean(v)) for f, v in model_rates.items()},
        "random": {str(f): float(np.mean(v)) for f, v in random_rates.items()},
    }


# --------------------------------------------------------------------------- #
# $1K upgrade worked example: score_upgrade_prospects on a fixed seeded panel
# --------------------------------------------------------------------------- #
def upgrade_worked_example() -> Dict[str, Any]:
    panel = make_donor_panel(n_donors=N_DONORS, n_years=N_YEARS, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, report = score_upgrade_prospects(panel["gifts"], random_state=0)
    return {
        "validation_fiscal_year": report["validation_fiscal_year"],
        "n_validation_rows": report["n_validation_rows"],
        "top_n": report["top_n"],
        "model_upgrade_rate_top_n": report["model_upgrade_rate_top_n"],
        "baseline_topn_fy_total_upgrade_rate": report["baseline_topn_fy_total_upgrade_rate"],
        "overall_upgrade_rate": report["overall_upgrade_rate"],
        "deciles": report["deciles"],
        "n_scored": report["n_scored"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-kdd98", action="store_true", help="Also run the KDD Cup 1998 section (downloads ~36MB).")
    args = parser.parse_args()

    results: Dict[str, Any] = {}

    # --- synthetic: response / major gift -----------------------------------
    resp_rows = bm.bench_response(SEEDS, N_DONORS, N_YEARS)
    rand_resp = random_rate_response() * 100
    results["response_synthetic"] = {
        f"top{p}pct": {
            "model": _row(resp_rows, "MajorGiftClassifier", f"top{p}pct_hit_rate").value * 100,
            "rule": _row(resp_rows, "MajorGiftClassifier", f"top{p}pct_hit_rate").baseline * 100,
            "random": rand_resp,
        }
        for p in (1, 5, 10)
    }
    results["response_synthetic"]["verdict"] = _row(resp_rows, "MajorGiftClassifier", "top10pct_hit_rate").verdict
    _bar_chart(
        OUT_DIR / "response.png",
        ["Top 1%", "Top 5%", "Top 10%"],
        {
            "Model": [results["response_synthetic"][f"top{p}pct"]["model"] for p in (1, 5, 10)],
            "Rank by giving so far": [results["response_synthetic"][f"top{p}pct"]["rule"] for p in (1, 5, 10)],
            "Random pick": [results["response_synthetic"][f"top{p}pct"]["random"] for p in (1, 5, 10)],
        },
        {"Model": COLOR_MODEL, "Rank by giving so far": COLOR_RULE, "Random pick": COLOR_RANDOM},
        ylabel="Gave next year, out of every 100 picked (%)",
        title="Who responds next year: model vs. rank-by-giving vs. random",
    )

    # --- synthetic: lapse ----------------------------------------------------
    lapse_rows = bm.bench_lapse(SEEDS, N_DONORS, N_YEARS)
    rand_lapse = random_rate_lapse() * 100
    results["lapse_synthetic"] = {
        f"top{p}pct": {
            "model": _row(lapse_rows, "LapsePredictor", f"top{p}pct_hit_rate").value * 100,
            "rule": _row(lapse_rows, "LapsePredictor", f"top{p}pct_hit_rate").baseline * 100,
            "random": rand_lapse,
        }
        for p in (1, 5, 10)
    }
    results["lapse_synthetic"]["verdict"] = _row(lapse_rows, "LapsePredictor", "top10pct_hit_rate").verdict

    # --- synthetic: ask -------------------------------------------------
    ask_rows = bm.bench_ask(SEEDS, N_DONORS, N_YEARS)
    within = _row(ask_rows, "AskAmountRecommender", "within25pct")
    results["ask_synthetic"] = {
        "within25pct_model": within.value * 100,
        "within25pct_last_gift": within.baseline * 100,
        "verdict": within.verdict,
    }

    # --- synthetic: $1K upgrade (bench_upgrade) ------------------------------
    upgrade_rows = bm.bench_upgrade(SEEDS, N_DONORS, N_YEARS)
    rand_upgrade = random_rate_upgrade() * 100
    results["upgrade_synthetic"] = {
        f"top{p}pct": {
            "model": _row(upgrade_rows, "upgrade_model (MajorGiftClassifier)", f"top{p}pct_hit_rate").value * 100,
            "rule": _row(upgrade_rows, "upgrade_model (MajorGiftClassifier)", f"top{p}pct_hit_rate").baseline * 100,
            "random": rand_upgrade,
        }
        for p in (1, 5, 10)
    }
    results["upgrade_synthetic"]["verdict"] = _row(
        upgrade_rows, "upgrade_model (MajorGiftClassifier)", "top10pct_hit_rate"
    ).verdict
    _bar_chart(
        OUT_DIR / "upgrade_topn.png",
        ["Top 1%", "Top 5%", "Top 10%"],
        {
            "Model": [results["upgrade_synthetic"][f"top{p}pct"]["model"] for p in (1, 5, 10)],
            "Rank by this year's total": [results["upgrade_synthetic"][f"top{p}pct"]["rule"] for p in (1, 5, 10)],
            "Random pick": [results["upgrade_synthetic"][f"top{p}pct"]["random"] for p in (1, 5, 10)],
        },
        {"Model": COLOR_MODEL, "Rank by this year's total": COLOR_RULE, "Random pick": COLOR_RANDOM},
        ylabel="Crossed $1,000 next year, out of every 100 picked (%)",
        title="Who upgrades to $1,000+: model vs. rank-by-total vs. random",
    )

    # --- synthetic: planned giving (coverage vs. chance) ---------------------
    pg = planned_giving_hit_rates()
    results["planned_giving_synthetic"] = pg
    _bar_chart(
        OUT_DIR / "planned_giving.png",
        ["Top 1%", "Top 5%", "Top 10%"],
        {
            "Model": [pg["model"][str(f)] * 100 for f in (0.01, 0.05, 0.10)],
            "Random pick": [pg["random"][str(f)] * 100 for f in (0.01, 0.05, 0.10)],
        },
        {"Model": COLOR_MODEL, "Random pick": COLOR_RANDOM},
        ylabel="Gave next year, out of every 100 picked (%)",
        title="Planned-giving score vs. random (giving-response used as a stand-in label)",
    )

    # --- $1K upgrade worked example ------------------------------------------
    ex = upgrade_worked_example()
    results["upgrade_worked_example"] = ex
    deciles = ex["deciles"]
    _bar_chart(
        OUT_DIR / "upgrade_deciles.png",
        [f"D{d['decile']}" for d in deciles],
        {"Upgrade rate": [d["actual_rate"] * 100 if d["actual_rate"] is not None else 0.0 for d in deciles]},
        {"Upgrade rate": COLOR_MODEL},
        ylabel="Crossed $1,000 next year (%)",
        title="Upgrade rate by decile (D1 = top 10% of picks, D10 = bottom 10%)",
    )

    # --- KDD98 (opt-in) --------------------------------------------------
    if args.with_kdd98:
        seed = bm.KDD_SEED
        kdd_resp = bm.bench_kdd_response(seed)
        kdd_lapse = bm.bench_kdd_lapse(seed)
        kdd_ask = bm.bench_kdd_ask(seed)
        kdd_cost = bm.bench_kdd_cost_aware(seed)

        results["response_kdd98"] = {
            f"top{p}pct": {
                "model": _row(kdd_resp, "MajorGiftClassifier", f"top{p}pct_hit_rate").value * 100,
                "rule": _row(kdd_resp, "MajorGiftClassifier", f"top{p}pct_hit_rate").baseline * 100,
            }
            for p in (1, 5, 10)
        }
        results["response_kdd98"]["verdict"] = _row(kdd_resp, "MajorGiftClassifier", "top10pct_hit_rate").verdict

        lapse_top = {p: _row(kdd_lapse, "LapsePredictor", f"top{p}pct_hit_rate") for p in (1, 5, 10)}
        donors = bm.fetch_kdd98_donors()
        base_rate_lapse_kdd = float((donors["TARGET_B"].to_numpy() == 0).mean()) * 100
        results["lapse_kdd98"] = {
            "base_rate_pct": base_rate_lapse_kdd,
            **{f"top{p}pct": {"model": r.value * 100, "rule": r.baseline * 100} for p, r in lapse_top.items()},
        }
        _bar_chart(
            OUT_DIR / "lapse_kdd98.png",
            ["Top 1%", "Top 5%", "Top 10%"],
            {
                "Model": [results["lapse_kdd98"][f"top{p}pct"]["model"] for p in (1, 5, 10)],
                "Gave nothing last time": [results["lapse_kdd98"][f"top{p}pct"]["rule"] for p in (1, 5, 10)],
                "Random pick": [base_rate_lapse_kdd] * 3,
            },
            {"Model": COLOR_MODEL, "Gave nothing last time": COLOR_RULE, "Random pick": COLOR_RANDOM},
            ylabel="Lapsed next period, out of every 100 picked (%)",
            title="Who lapses next: KDD Cup 1998 (almost everyone lapses here)",
        )

        ask_row = _row(kdd_ask, "AskAmountRecommender", "within25pct")
        results["ask_kdd98"] = {
            "within25pct_model": ask_row.value * 100,
            "within25pct_last_gift": ask_row.baseline * 100,
            "verdict": ask_row.verdict,
        }
        _bar_chart(
            OUT_DIR / "ask_kdd98.png",
            ["Suggested ask"],
            {
                "Model": [results["ask_kdd98"]["within25pct_model"]],
                "Ask what they gave last time": [results["ask_kdd98"]["within25pct_last_gift"]],
            },
            {"Model": COLOR_MODEL, "Ask what they gave last time": COLOR_RULE},
            ylabel="Within 25% of what the donor actually gave (%)",
            title="How close is the suggested ask: KDD Cup 1998",
        )

        net_row = _row(kdd_cost, "cost_aware_selection", "net_revenue")
        results["who_to_mail_kdd98"] = {
            "net_revenue_model": net_row.value,
            "net_revenue_mail_everyone": net_row.baseline,
            "verdict": net_row.verdict,
            "note": net_row.note,
        }
        _bar_chart(
            OUT_DIR / "who_to_mail.png",
            ["Net revenue"],
            {
                "Only mail likely responders": [results["who_to_mail_kdd98"]["net_revenue_model"]],
                "Mail everyone": [results["who_to_mail_kdd98"]["net_revenue_mail_everyone"]],
            },
            {"Only mail likely responders": COLOR_MODEL, "Mail everyone": COLOR_RULE},
            ylabel="Net revenue after mailing cost ($)",
            title="Who to mail: KDD Cup 1998",
        )

    with open(OUT_DIR / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Wrote {OUT_DIR / 'results.json'} and PNGs to {OUT_DIR}")


if __name__ == "__main__":
    main()
