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
``score_leadership_prospects`` on ``make_donor_panel(random_state=0)`` for the
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
from collections import Counter
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas  # noqa: E402
import sklearn  # noqa: E402
from sklearn.inspection import partial_dependence  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_models_vs_baselines as bm  # noqa: E402

from philanthropy.datasets import make_donor_panel  # noqa: E402
from philanthropy.inspection import donor_feature_importance  # noqa: E402
from philanthropy.models import PlannedGivingIntentScorer, score_leadership_prospects  # noqa: E402


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"

OUT_DIR = ROOT / "docs" / "assets" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = bm.DEFAULT_SEEDS
N_DONORS, N_YEARS = 3000, 7

# Categorical palette slots 1/2/3 (blue/orange/aqua) from the dataviz skill's
# validated default palette: model, simple rule, random. Every chart is
# rendered once per THEMES entry (docs/stylesheets/extra.css ships both a
# light and a dark `data-md-color-scheme`, and a chart baked for one reads as
# a bright or a black rectangle in the other); each markdown page then embeds
# both PNGs with the `#only-light` / `#only-dark` suffixes Material's theme
# switches on natively. Dark steps are the same palette's dark column, not
# improvised: "random" keeps the same muted ink in both modes.
LIGHT = {
    "model": "#2a78d6", "rule": "#eb6834", "random": "#898781",
    "surface": "#fcfcfb", "ink1": "#0b0b0b", "ink2": "#52514e", "grid": "#e1e0d9",
}
DARK = {
    "model": "#3987e5", "rule": "#d95926", "random": "#898781",
    "surface": "#1a1a19", "ink1": "#ffffff", "ink2": "#c3c2b7", "grid": "#2c2c2a",
}
THEMES = (LIGHT, DARK)


def _themed_path(path: Path, theme: dict) -> Path:
    return path if theme is LIGHT else path.with_name(f"{path.stem}-dark{path.suffix}")


def _render_themed(draw_fn, path: Path, *args, **kwargs) -> None:
    """Calls `draw_fn(path, *args, theme=..., **kwargs)` once per entry in
    THEMES, writing `name.png` for light and `name-dark.png` for dark."""
    for theme in THEMES:
        draw_fn(_themed_path(path, theme), *args, theme=theme, **kwargs)


def _fmt_count(v: float) -> str:
    """'24 of 100' style label for a hit-rate percentage."""
    return f"{v:.0f} of 100"


def _fmt_money(v: float) -> str:
    return f"${v:,.0f}"


def _takeaway(model_v: float, rule_v: float, model_name: str = "The model", rule_name: str = "the rule") -> str:
    """One clause naming which of model/rule is ahead at top 10%, for a
    chart title. Computed from the numbers, not hand-written, so it cannot
    drift from the data the way a hardcoded title can."""
    gap = abs(model_v - rule_v)
    if gap < 1.5:
        return f"{model_name} and {rule_name} are about the same here"
    if model_v > rule_v:
        return f"{model_name} beats {rule_name}: {model_v:.0f} of 100 vs {rule_v:.0f} of 100 in the top 10%"
    return f"{rule_name.capitalize()} beats {model_name.lower()}: {rule_v:.0f} of 100 vs {model_v:.0f} of 100 in the top 10%"


# Fixed-inch header band (title, legend, subtitle) so the three never
# collide regardless of how many bar groups a chart has: only the body
# (the axes) grows with `n_groups`. `_finish_chart` draws the shared header
# and frame; the two chart kinds below just plot into the returned axes.
_HEADER_IN = 1.55
_TITLE_Y_IN = 0.32   # from the top, for a one-line title
_TITLE_LINE_IN = 0.30  # extra height per wrapped title line beyond the first
_LEGEND_Y_IN = 0.78  # from the top, for a one-line title
_FIG_W = 7.6
_TITLE_WRAP_CHARS = 62
_SUBTITLE_WRAP_CHARS = 100
_SUBTITLE_LINE_IN = 0.16  # extra height per wrapped subtitle line beyond the first


def _wrap_title(title: str) -> str:
    """Wraps a takeaway title to the figure width instead of letting
    matplotlib clip it, since `suptitle` does not wrap on its own."""
    return "\n".join(textwrap.wrap(title, _TITLE_WRAP_CHARS)) or title


def _wrap_subtitle(subtitle: str) -> str:
    return "\n".join(textwrap.wrap(subtitle, _SUBTITLE_WRAP_CHARS)) or subtitle


def _new_chart_figure(
    n_groups: int, groups: List[str], title: str, subtitle: str, theme: dict,
    body_in_per_group: float = 0.62, body_min_in: float = 0.9,
):
    wrapped_title = _wrap_title(title)
    wrapped_subtitle = _wrap_subtitle(subtitle)
    extra_in = _TITLE_LINE_IN * wrapped_title.count("\n") + _SUBTITLE_LINE_IN * wrapped_subtitle.count("\n")
    body_in = max(body_min_in, body_in_per_group * n_groups)
    fig_h = _HEADER_IN + extra_in + body_in
    fig, ax = plt.subplots(figsize=(_FIG_W, fig_h), dpi=150)
    ax.set_facecolor(theme["surface"])
    fig.patch.set_facecolor(theme["surface"])
    top_frac = body_in / fig_h
    left_frac = min(0.32, 0.025 + 0.011 * max((len(g) for g in groups), default=0))
    fig.subplots_adjust(top=top_frac, left=left_frac, right=0.98, bottom=max(0.06, 0.5 / fig_h))
    return fig, ax, fig_h, extra_in, wrapped_title, wrapped_subtitle


def _finish_chart(
    path: Path, fig, ax, fig_h: float, extra_in: float, wrapped_title: str, wrapped_subtitle: str, legend_ncol: int,
    theme: dict,
) -> None:
    """Draws the title (bold, top), the legend (one row, centered, below the
    title and above the subtitle), and the subtitle (grey, directly above
    the axes) at fixed inch offsets from the top of the figure, so none of
    the three ever overlaps no matter how tall or short the plot body is,
    or how many lines the wrapped title or subtitle need."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="center", bbox_to_anchor=(0.5, 1 - (_LEGEND_Y_IN + extra_in) / fig_h),
            ncol=legend_ncol, frameon=False, labelcolor=theme["ink2"], fontsize=9,
        )
    ax.set_title(wrapped_subtitle, color=theme["ink2"], fontsize=8.5, loc="left", pad=8)
    fig.suptitle(
        wrapped_title, x=0.015, ha="left", y=1 - _TITLE_Y_IN / fig_h, fontsize=12, fontweight="bold",
        color=theme["ink1"], linespacing=1.3,
    )
    fig.savefig(path)
    plt.close(fig)


def _style_value_axis(ax, groups: List[str], theme: dict) -> None:
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels(groups)
    ax.invert_yaxis()
    ax.set_xticks([])
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(theme["grid"])
    ax.tick_params(colors=theme["ink2"], length=0)


def _hbar_chart(
    path: Path,
    groups: List[str],
    series: Dict[str, List[float]],
    colors: Dict[str, str],
    title: str,
    subtitle: str,
    theme: dict,
    errors: Dict[str, List[Any]] | None = None,
    value_fmt=None,
    base_rate: float | None = None,
    base_rate_label: str = "",
) -> None:
    """Horizontal grouped bar chart: one cluster of bars per `groups` entry
    (e.g. "Top 1%"), one bar per series. The title states the finding; the
    subtitle (smaller, grey, below the title) names what was measured. Bars
    carry their own value as a direct end label, placed past the error
    whisker so the whisker never crosses the text. `errors[name]` is an
    optional list of `(lo, hi)` pairs (or `None` per group), giving an
    asymmetric error bar from a 5-seed range or a bootstrap interval.
    `base_rate`, if given, is drawn as a dashed line instead of a fourth
    bar. Legend sits above the plot, never over a bar. `colors[name]` is a
    theme role key ("model"/"rule"/"random"), resolved against `theme`."""
    fmt = value_fmt or _fmt_count
    n_groups, n_series = len(groups), len(series)
    height = 0.8 / n_series
    y = np.arange(n_groups)
    fig, ax, fig_h, extra_in, wrapped_title, wrapped_subtitle = _new_chart_figure(
        n_groups, groups, title, subtitle, theme,
    )
    max_v = max((v for values in series.values() for v in values), default=1.0) or 1.0
    max_extent = max_v
    for i, (name, values) in enumerate(series.items()):
        offset = ((n_series - 1) / 2 - i) * height
        errs = errors.get(name) if errors else None
        xerr = None
        if errs and any(e is not None for e in errs):
            lo = [v - (e[0] if e is not None else v) for v, e in zip(values, errs)]
            hi = [(e[1] if e is not None else v) - v for v, e in zip(values, errs)]
            xerr = [lo, hi]
        bars = ax.barh(
            y + offset, values, height=height * 0.85, label=name,
            color=theme[colors[name]], edgecolor="none",
            xerr=xerr, ecolor=theme["ink2"], capsize=2,
            error_kw={"linewidth": 1, "alpha": 0.8},
        )
        for j, (b, v) in enumerate(zip(bars, values)):
            hi_extent = errs[j][1] if errs and errs[j] is not None else v
            max_extent = max(max_extent, hi_extent)
            ax.text(
                max(b.get_width(), hi_extent) + max_v * 0.03, b.get_y() + b.get_height() / 2,
                fmt(v), ha="left", va="center", fontsize=9, color=theme["ink1"],
            )
    if base_rate is not None:
        max_extent = max(max_extent, base_rate)
        ax.axvline(base_rate, color=theme["random"], linestyle="--", linewidth=1.4, zorder=0)
        if base_rate_label:
            ax.text(
                base_rate, -0.5, base_rate_label, color=theme["ink2"], fontsize=8.5,
                ha="left", va="bottom",
            )
    _style_value_axis(ax, groups, theme)
    ax.set_xlim(0, max_extent * 1.30)
    if base_rate_label:
        # The reference-line label sits just above row 0; without extra
        # headroom it overlaps that row's bar instead of floating above it.
        ax.set_ylim(n_groups - 0.5, -0.9)
    _finish_chart(path, fig, ax, fig_h, extra_in, wrapped_title, wrapped_subtitle, n_series, theme)


def _neartie_dot_chart(
    path: Path,
    groups: List[str],
    series: Dict[str, List[float]],
    colors: Dict[str, str],
    base_rate: float,
    title: str,
    subtitle: str,
    theme: dict,
) -> None:
    """For a comparison where model, rule and random are all close together
    (e.g. a file where almost everyone lapses): dots instead of bars, a
    zoomed value axis, and the random/base rate drawn as a dashed line
    rather than a third bar, since a bar chart of 96 vs 97 vs 95 makes three
    identical-looking columns."""
    n_groups = len(groups)
    y = np.arange(n_groups)
    all_v = [v for values in series.values() for v in values] + [base_rate]
    lo_v, hi_v = min(all_v), max(all_v)
    pad = max(1.5, (hi_v - lo_v) * 0.9)
    fig, ax, fig_h, extra_in, wrapped_title, wrapped_subtitle = _new_chart_figure(
        n_groups, groups, title, subtitle, theme,
    )
    ax.axvline(
        base_rate, color=theme["random"], linestyle="--", linewidth=1.4, zorder=1,
        label=f"Picking at random: {base_rate:.0f} of 100",
    )
    offsets = np.linspace(-0.16, 0.16, len(series))
    for off, (name, values) in zip(offsets, series.items()):
        ax.scatter(values, y + off, color=theme[colors[name]], s=60, zorder=3, label=name)
        for v, yy in zip(values, y + off):
            ax.text(v, yy, f"  {v:.0f}", va="center", ha="left", fontsize=9, color=theme["ink1"])
    _style_value_axis(ax, groups, theme)
    ax.set_xlim(lo_v - pad, hi_v + pad)
    _finish_chart(path, fig, ax, fig_h, extra_in, wrapped_title, wrapped_subtitle, len(series) + 1, theme)


def _profit_curve_chart(
    path: Path,
    mailed: List[int],
    net_revenue: List[float],
    stop_k: int,
    stop_net: float,
    everyone_k: int,
    everyone_net: float,
    title: str,
    subtitle: str,
    theme: dict,
) -> None:
    """Net revenue (y) against how many donors are mailed (x), most likely
    to respond first, replacing the two-bar who-to-mail chart (E.13c). Marks
    the model's own "mail if E[gift]>cost" stopping point and the
    mail-everyone endpoint on the same curve, so the whole decision -
    including what a higher or lower cost per piece would do to it - is in
    one picture."""
    wrapped_title = _wrap_title(title)
    wrapped_subtitle = _wrap_subtitle(subtitle)
    extra_in = _TITLE_LINE_IN * wrapped_title.count("\n") + _SUBTITLE_LINE_IN * wrapped_subtitle.count("\n")
    body_in = 2.6
    fig_h = _HEADER_IN + extra_in + body_in
    fig, ax = plt.subplots(figsize=(_FIG_W, fig_h), dpi=150)
    ax.set_facecolor(theme["surface"])
    fig.patch.set_facecolor(theme["surface"])
    fig.subplots_adjust(top=body_in / fig_h, left=0.16, right=0.97, bottom=0.65 / fig_h)

    ax.plot(mailed, net_revenue, color=theme["model"], linewidth=2, zorder=2)
    ax.scatter(
        [stop_k], [stop_net], color=theme["model"], s=70, zorder=3, edgecolor=theme["surface"], linewidth=1,
        label=f"Model's stopping point: {stop_k:,} mailed, {_fmt_money(stop_net)}",
    )
    ax.scatter(
        [everyone_k], [everyone_net], color=theme["rule"], s=70, zorder=3, marker="s",
        edgecolor=theme["surface"], linewidth=1,
        label=f"Mail everyone: {everyone_k:,} mailed, {_fmt_money(everyone_net)}",
    )
    ax.set_xlabel("Donors mailed, ranked most to least likely to respond", color=theme["ink2"], fontsize=9)
    ax.set_ylabel("Net revenue", color=theme["ink2"], fontsize=9)
    ax.xaxis.set_major_formatter(lambda v, _pos: f"{v:,.0f}")
    ax.yaxis.set_major_formatter(lambda v, _pos: _fmt_money(v))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(theme["grid"])
    ax.spines["bottom"].set_color(theme["grid"])
    ax.tick_params(colors=theme["ink2"])
    ax.grid(axis="y", color=theme["grid"], linewidth=0.6, zorder=0)
    _finish_chart(path, fig, ax, fig_h, extra_in, wrapped_title, wrapped_subtitle, 2, theme)


_SCOREBOARD_X = {"loses": 0.15, "modest": 0.5, "wins": 0.85}
_SCOREBOARD_MARKERS = {"Sample data": "o", "KDD Cup 1998": "s", "cup98VAL": "^"}


def _scoreboard_chart(path: Path, rows: List[tuple], theme: dict) -> None:
    """One row per question, one dot per dataset, placed left (loses to the
    simple rule), centre (about the same) or right (beats the rule); a
    dataset with no rule to compare against (planned giving) gets a hollow
    grey marker at centre labelled untested instead. Different marker shapes
    (not just colour) tell datasets apart for colour-blind readers. Replaces
    a table as the first thing a reader sees (E.13c)."""
    n = len(rows)
    row_in = 0.5
    fig_h = _HEADER_IN + row_in * n + 0.7
    fig, ax = plt.subplots(figsize=(_FIG_W, fig_h), dpi=150)
    ax.set_facecolor(theme["surface"])
    fig.patch.set_facecolor(theme["surface"])
    body_in = row_in * n + 0.7
    fig.subplots_adjust(top=body_in / fig_h, left=0.24, right=0.98, bottom=0.55 / fig_h)

    seen_datasets: List[str] = []
    n_wins = n_total = 0
    for i, (_label, dots) in enumerate(rows):
        y = n - 1 - i
        y_jitter = np.linspace(-0.12, 0.12, len(dots)) if len(dots) > 1 else [0.0]
        for (dataset, verdict), dy in zip(dots, y_jitter):
            if dataset not in seen_datasets:
                seen_datasets.append(dataset)
            marker = _SCOREBOARD_MARKERS.get(dataset, "o")
            if verdict is None:
                ax.scatter(
                    [0.5], [y + dy], marker="o", s=90, facecolor="none", edgecolor=theme["random"],
                    linewidth=1.4, zorder=3,
                )
                continue
            n_total += 1
            n_wins += verdict == "wins"
            ax.scatter(
                [_SCOREBOARD_X[verdict]], [y + dy], marker=marker, s=90, color=theme["model"],
                edgecolor=theme["surface"], linewidth=1, zorder=3,
            )
    for xv in _SCOREBOARD_X.values():
        ax.axvline(xv, color=theme["grid"], linewidth=0.8, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_xticks(list(_SCOREBOARD_X.values()))
    ax.set_xticklabels(["Worse than\nthe rule", "About the\nsame", "Beats\nthe rule"], fontsize=8.5, color=theme["ink2"])
    ax.set_yticks(range(n))
    ax.set_yticklabels([label for label, _ in reversed(rows)])
    ax.set_ylim(-0.6, n - 0.4)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=theme["ink2"], length=0)

    legend_handles = [
        plt.Line2D([0], [0], marker=_SCOREBOARD_MARKERS.get(ds, "o"), color="w", markerfacecolor=theme["model"],
                   markeredgecolor=theme["surface"], markersize=9, label=ds)
        for ds in seen_datasets
    ] + [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor=theme["random"],
                   markersize=9, label="Not yet tested")
    ]

    title = f"{n_wins} of {n_total} real-and-sample-data comparisons beat the simple rule your shop already uses"
    wrapped_title = _wrap_title(title)
    extra_in = _TITLE_LINE_IN * wrapped_title.count("\n")
    fig.legend(
        handles=legend_handles, loc="center", bbox_to_anchor=(0.5, 1 - (_LEGEND_Y_IN + extra_in) / fig_h),
        ncol=len(legend_handles), frameon=False, labelcolor=theme["ink2"], fontsize=9,
    )
    fig.suptitle(
        wrapped_title, x=0.015, ha="left", y=1 - _TITLE_Y_IN / fig_h, fontsize=12, fontweight="bold",
        color=theme["ink1"], linespacing=1.3,
    )
    fig.savefig(path)
    plt.close(fig)


def _row(rows, model, metric):
    for r in rows:
        if r.model == model and r.metric == metric:
            return r
    return None


def _ci(row) -> tuple[float, float] | None:
    """A row's `(lo, hi)` interval in percentage points: a 5-seed min/max
    range for synthetic rows, a bootstrap 95% interval for KDD98 single-split
    rows on the metrics `bootstrap=True` was requested for, or `None` when
    neither applies (E.11a rule 4)."""
    if row is None or row.lo is None or row.hi is None:
        return None
    return (row.lo * 100, row.hi * 100)


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
            snaps = bm.build_leadership_snapshots(
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
# $1K upgrade worked example: score_leadership_prospects on a fixed seeded panel
# --------------------------------------------------------------------------- #
def upgrade_worked_example(seed: int) -> Dict[str, Any]:
    """One seed's worth of the public `score_leadership_prospects` API, for the
    donor-count narrative on the Upgrade page. Uses the first of `SEEDS`
    (the same seeds `bench_upgrade`'s 5-seed average uses) rather than an
    unrelated fixed seed, so the worked example is traceable to the same
    population the top-of-page chart summarises, even though it is a
    different pipeline (the shipped function, not the benchmark's own
    feature set) and so a different single number."""
    panel = make_donor_panel(n_donors=N_DONORS, n_years=N_YEARS, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, report = score_leadership_prospects(panel["gifts"], random_state=seed)
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


def upgrade_decile_average(seeds) -> Dict[str, Any]:
    """Averages `score_leadership_prospects`'s per-decile upgrade rate across
    all of `seeds` (min/max kept as a range), instead of reading the decile
    breakdown off a single seed the way the page used to. A one-seed decile
    chart can show a step that is just that seed's noise (E.13a finding 5);
    averaging is the same fix `bench_upgrade`'s topn numbers already get."""
    per_decile: Dict[int, list] = {}
    overall_rates = []
    for seed in seeds:
        panel = make_donor_panel(n_donors=N_DONORS, n_years=N_YEARS, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _, report = score_leadership_prospects(panel["gifts"], random_state=seed)
        overall_rates.append(report["overall_upgrade_rate"] * 100)
        for d in report["deciles"]:
            if d["actual_rate"] is not None:
                per_decile.setdefault(d["decile"], []).append(d["actual_rate"] * 100)
    deciles = sorted(per_decile)
    return {
        "decile": deciles,
        "mean": [float(np.mean(per_decile[d])) for d in deciles],
        "lo": [float(min(per_decile[d])) for d in deciles],
        "hi": [float(max(per_decile[d])) for d in deciles],
        "overall_mean": float(np.mean(overall_rates)),
        "n_seeds": len(seeds),
    }


# --------------------------------------------------------------------------- #
# "What the model looks at": plain-language feature groups, labels and
# driver computation for the second tab group on every results page
# (E.11i; the feature-level schema is new, everything else about the page
# follows the same reader/writing/chart rules as the rest of this script).
# --------------------------------------------------------------------------- #
GROUP_ORDER = ("giving_history", "recency", "momentum", "engagement", "wealth", "mailing")
GROUP_LABELS = {
    "giving_history": "Giving history",
    "recency": "Recency",
    # Labeled "Year-over-year change", not "Momentum": the columns in this
    # group (fy_trend, streak, consecutive_years_given) are a one-year
    # difference and a years-given count, not the multi-year trailing-slope
    # "momentum" features from philanthropy.utils._momentum, which no
    # benchmark here uses - "Momentum" is reserved for those *_slope_*/
    # *_rel_slope_* columns, should a benchmark ever use them.
    "momentum": "Year-over-year change",
    "engagement": "Engagement",
    "wealth": "Wealth & demographics",
    "mailing": "Mailing history",
}
GROUP_MEANINGS = {
    "giving_history": "how much and how often they have given in total",
    "recency": "how recently they gave",
    "momentum": "whether this year's giving is up or down from last year",
    "engagement": "events, volunteering and other non-gift contact",
    "wealth": "wealth screening and demographic data",
    "mailing": "how they have responded to past mailings",
}

# Groups each dataset's underlying file could plausibly report, regardless of
# whether the benchmarked model is actually given them (E.11i's "What the
# model looks at" must not claim a file lacks data it has but a given model
# simply isn't fed; see _render_group_table). "momentum" is deliberately
# absent from every value here: it isn't a record type a file "has" or
# "lacks" the way wealth or mailing history is, it's a computed column, so it
# is excluded from that sentence entirely rather than asserted either way.
DATASET_GROUPS_AVAILABLE = {
    # Every synthetic driver function below fits on make_donor_panel's
    # output (via _period_panel), never generate_synthetic_donor_data's, so
    # "engagement" (that other generator's event_attendance_count) really is
    # absent here. make_donor_panel's donors frame does carry
    # wealth_estimate, unused by every synthetic benchmark, so "wealth" is
    # available-but-unused, not absent. Its gifts frame's "appeal" records
    # which campaign an actual gift came from, not a solicitation/response
    # history across mailed and non-responding donors, so it isn't "mailing
    # history" in the sense KDD98's TIMELAG/promotion-history columns are;
    # mailing stays absent for synthetic data.
    "synthetic": frozenset({"giving_history", "recency", "wealth"}),
    "kdd98": frozenset({"giving_history", "recency", "wealth", "mailing"}),
    "cup98val": frozenset({"giving_history", "recency", "wealth", "mailing"}),
}


def _dataset_category(key: str) -> str:
    for suffix in ("_synthetic", "_cup98val", "_kdd98"):
        if key.endswith(suffix):
            return suffix[1:]
    raise ValueError(f"unrecognized dataset key: {key!r}")


def _join_english(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"

# Column -> (plain label, group). Every feature any bench_* function in
# scripts/benchmark_models_vs_baselines.py actually feeds a model, across
# every dataset this script scores. A column missing here is a bug, not a
# silent default: the renderer raises on an unmapped column.
FEATURE_INFO = {
    # synthetic as-of panel (_period_panel / PARITY_FEATURES)
    "total": ("lifetime giving", "giving_history"),
    "n": ("number of gifts", "giving_history"),
    "recent": ("this year's gift", "recency"),
    "streak": ("giving streak", "momentum"),
    "years_since_last": ("years since last gift", "recency"),
    "prev_recent": ("last year's gift", "recency"),
    "max_gift": ("biggest single gift", "giving_history"),
    "tenure": ("years as a donor", "giving_history"),
    # synthetic upgrade snapshots (build_leadership_snapshots)
    "fiscal_year": ("which fiscal year", "giving_history"),
    "fy_total": ("this year's giving", "recency"),
    "fy_total_prior1": ("last year's giving", "recency"),
    "fy_total_prior2": ("giving two years ago", "recency"),
    "fy_trend": ("giving trend, this year vs last", "momentum"),
    "largest_gift": ("largest gift this year", "giving_history"),
    "gift_count": ("number of gifts this year", "giving_history"),
    "consecutive_years_given": ("consecutive years given", "momentum"),
    "months_since_last_gift": ("months since last gift", "recency"),
    # KDD Cup 1998 (_kdd_feature_frame)
    "AGE": ("donor's age", "wealth"),
    "INCOME": ("household income bracket", "wealth"),
    "WEALTH1": ("wealth rating (source 1)", "wealth"),
    "WEALTH2": ("wealth rating (source 2)", "wealth"),
    "NUMCHLD": ("number of children", "wealth"),
    "HOMEOWNER": ("owns their home", "wealth"),
    "RAMNTALL": ("lifetime giving", "giving_history"),
    "NGIFTALL": ("lifetime number of gifts", "giving_history"),
    "LASTGIFT": ("last gift amount", "recency"),
    "AVGGIFT": ("average gift amount", "giving_history"),
    "MAXRAMNT": ("largest gift ever", "giving_history"),
    "MINRAMNT": ("smallest gift ever", "giving_history"),
    "TIMELAG": ("days between past mailings and gifts", "mailing"),
    "rfm_recency": ("months since last gift", "recency"),
    "rfm_frequency": ("number of gifts", "giving_history"),
    "rfm_monetary": ("lifetime giving", "giving_history"),
    "rfm_tenure": ("years as a donor", "giving_history"),
    # KDD Cup 1998 response model (_kdd_response_frame: unprefixed RFM)
    "recency": ("months since last gift", "recency"),
    "frequency": ("number of gifts", "giving_history"),
    "monetary": ("lifetime giving", "giving_history"),
}

# _kdd_feature_frame's column set (AskAmountRecommender, cost-aware mailing):
# _KDD_BASE_COLS plus the engineered HOMEOWNER flag and the rfm_-prefixed
# recency/frequency/monetary/tenure columns.
KDD_FEATURE_COLS = tuple(bm._KDD_BASE_COLS) + ("HOMEOWNER", "rfm_recency", "rfm_frequency", "rfm_monetary", "rfm_tenure")

FEATURE_SET_LABELS = {
    "current": "only what the model uses today",
    "full": "plus giving streak, time since last gift, last year's gift, biggest gift and tenure",
}

_DIRECTION_SYMBOL = {"+": "▲", "-": "▼", "mixed": "●"}


def _direction_phrase(direction: str, target_label: str) -> str:
    """'score' for a classifier's probability, but a regressor (e.g. the ask
    amount) doesn't have a 'score' a reader would recognize; callers pass the
    actual predicted quantity's name instead."""
    if direction == "+":
        return f"raises the {target_label}"
    if direction == "-":
        return f"lowers the {target_label}"
    return "depends"


def _pd_direction(estimator, X, feature_idx: int) -> str:
    """Overall trend of the partial-dependence curve for one feature: '+' if
    the model's score rises as the feature rises, '-' if it falls, 'mixed' if
    neither trend is consistent (E.11i item 6). Uses the curve's Spearman
    rank correlation against the grid order rather than requiring every step
    to point the same way: a calibrated/boosted estimator's curve almost
    never is perfectly monotone even when the real trend is one-directional
    (isotonic calibration and a coarse grid both add small local wiggles), so
    a strict sign-of-every-diff check calls nearly everything "mixed"."""
    try:
        # grid_resolution=10 (not sklearn's default 100): only the overall
        # trend is used, and a coarser grid is an order of magnitude cheaper
        # on the gradient-boosted/calibrated estimators this runs against (a
        # 5-seed x many-feature loop would otherwise be the single slowest
        # part of this script).
        result = partial_dependence(estimator, X, [feature_idx], kind="average", grid_resolution=10)
    except Exception:
        return "mixed"
    avg = np.asarray(result["average"]).reshape(-1)
    if len(avg) < 2:
        return "mixed"
    with warnings.catch_warnings():
        # A perfectly flat curve (the model is fully saturated over this
        # feature's observed range) makes scipy's constant-input warning
        # fire; corr is NaN either way, already handled below.
        warnings.simplefilter("ignore")
        corr = pandas.Series(avg).corr(pandas.Series(np.arange(len(avg))), method="spearman")
    if corr is None or np.isnan(corr):
        return "mixed"
    if corr >= 0.5:
        return "+"
    if corr <= -0.5:
        return "-"
    return "mixed"


def _top_drivers(
    make_model, seeds, build_fn, feature_cols: Sequence[str], scoring: str, top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Fits ``make_model(seed)`` (an unfitted estimator or pipeline) on
    ``build_fn(seed)`` (returning ``(Xtr, ytr, Xte, yte)``) for each seed, and
    averages permutation importance and partial-dependence direction across
    seeds (a 5-seed range, the same convention the rest of this script uses
    for synthetic rows). A single-seed ``build_fn`` (one fit) still works;
    the range then collapses to a single point (``lo``/``hi`` are ``None``)."""
    importances: Dict[str, List[float]] = {f: [] for f in feature_cols}
    directions: Dict[str, List[str]] = {f: [] for f in feature_cols}
    for seed in seeds:
        Xtr, ytr, Xte, yte = build_fn(seed)
        model = make_model(seed).fit(Xtr, ytr)
        fi = donor_feature_importance(
            model, Xte, yte, feature_names=list(feature_cols), random_state=seed, scoring=scoring,
        )
        for _, row in fi.iterrows():
            importances[row["feature"]].append(float(row["importance_mean"]))
        for i, f in enumerate(feature_cols):
            directions[f].append(_pd_direction(model, Xte, i))

    drivers = []
    for f in feature_cols:
        vals = importances[f]
        label, group = FEATURE_INFO[f]
        # Majority vote across seeds, not unanimous agreement: one seed's
        # curve landing just under the +/- correlation threshold should not
        # by itself override four seeds that agree.
        direction, _n_votes = Counter(directions[f]).most_common(1)[0]
        drivers.append(
            {
                "column": f,
                "label": label,
                "group": group,
                "importance": float(np.mean(vals)),
                "lo": float(min(vals)) if len(vals) > 1 else None,
                "hi": float(max(vals)) if len(vals) > 1 else None,
                "direction": direction,
            }
        )
    drivers.sort(key=lambda d: d["importance"], reverse=True)
    return drivers[:top_k]


def _features_entry(
    feature_cols: Sequence[str], drivers: List[Dict[str, Any]], scoring: str, split: str,
    extra_sets: List[Dict[str, Any]] | None = None, target_label: str = "score",
) -> Dict[str, Any]:
    """Builds the ``"features"`` block the schema in E.11i's pasted spec
    describes, with one ``"current"`` feature-set row (what the benchmarked
    model actually saw) plus any additional ablation rows the caller
    computed (e.g. the full-parity feature set). ``scoring`` and ``split``
    are recorded for the page's collapsed analyst note, not shown above it."""
    sets = [
        {
            "id": "current", "label": FEATURE_SET_LABELS["current"],
            "columns": list(feature_cols), "chosen": True,
        }
    ] + (extra_sets or [])
    return {
        "sets": sets, "drivers": drivers, "scoring": scoring, "split": split,
        "n_configs": 1, "git_sha": _git_sha(), "target_label": target_label,
    }


def _reused_features(features: Dict[str, Any], note: str) -> Dict[str, Any]:
    """Points a dataset tab at another tab's already-computed drivers (e.g.
    cup98VAL reusing the KDD Cup 1998 tab's model and columns) instead of
    re-fitting on a second large download for the same feature set."""
    return {**features, "reused_from_note": note}


def _topn_pct_block(rows, model_name: str) -> Dict[str, Dict[str, float]]:
    out = {}
    for p in (1, 5, 10):
        row = _row(rows, model_name, f"top{p}pct_hit_rate")
        out[f"top{p}pct"] = {"value": row.value * 100, "lo": _ci(row)[0] if _ci(row) else None, "hi": _ci(row)[1] if _ci(row) else None}
    return out


def response_drivers_synthetic(seeds, n_donors, n_years) -> Dict[str, Any]:
    feature_cols = ("total", "n", "recent")

    def build(seed):
        train, test = bm._train_test_periods(bm._period_panel(n_donors, n_years, seed))
        return (
            train[list(feature_cols)].to_numpy(), train["y_response"].to_numpy(),
            test[list(feature_cols)].to_numpy(), test["y_response"].to_numpy(),
        )

    drivers = _top_drivers(lambda seed: bm.MajorGiftClassifier(random_state=seed), seeds, build, feature_cols, scoring="roc_auc")
    full_rows = bm.bench_response(seeds, n_donors, n_years, feature_cols=bm.PARITY_FEATURES)
    full_set = {
        "id": "full", "label": FEATURE_SET_LABELS["full"], "columns": list(bm.PARITY_FEATURES), "chosen": False,
        **_topn_pct_block(full_rows, "MajorGiftClassifier"),
    }
    return _features_entry(feature_cols, drivers, scoring="roc_auc", split=bm.SYNTHETIC_SPLIT, extra_sets=[full_set])


def lapse_drivers_synthetic(seeds, n_donors, n_years) -> Dict[str, Any]:
    feature_cols = ("total", "n", "recent")

    def build(seed):
        train, test = bm._train_test_periods(bm._period_panel(n_donors, n_years, seed))
        return (
            train[list(feature_cols)].to_numpy(), 1 - train["y_response"].to_numpy(),
            test[list(feature_cols)].to_numpy(), 1 - test["y_response"].to_numpy(),
        )

    drivers = _top_drivers(lambda seed: bm.LapsePredictor(random_state=seed), seeds, build, feature_cols, scoring="roc_auc")
    full_rows = bm.bench_lapse(seeds, n_donors, n_years, feature_cols=bm.PARITY_FEATURES)
    full_set = {
        "id": "full", "label": FEATURE_SET_LABELS["full"], "columns": list(bm.PARITY_FEATURES), "chosen": False,
        **_topn_pct_block(full_rows, "LapsePredictor"),
    }
    return _features_entry(feature_cols, drivers, scoring="roc_auc", split=bm.SYNTHETIC_SPLIT, extra_sets=[full_set])


def ask_drivers_synthetic(seeds, n_donors, n_years) -> Dict[str, Any]:
    feature_cols = ("total", "n", "recent")

    def build(seed):
        train, test = bm._train_test_periods(bm._period_panel(n_donors, n_years, seed))
        train_resp, test_resp = train[train["y_response"] == 1], test[test["y_response"] == 1]
        return (
            train_resp[list(feature_cols)].to_numpy(), train_resp["y_amount"].to_numpy(),
            test_resp[list(feature_cols)].to_numpy(), test_resp["y_amount"].to_numpy(),
        )

    drivers = _top_drivers(lambda seed: bm.AskAmountRecommender(random_state=seed), seeds, build, feature_cols, scoring="neg_mean_absolute_error")
    return _features_entry(
        feature_cols, drivers, scoring="neg_mean_absolute_error", split=bm.SYNTHETIC_SPLIT, target_label="suggested ask",
    )


# build_leadership_snapshots always returns this fixed numeric column set (no
# activities/donors frame is passed in bench_upgrade, so no engagement or
# wealth columns ever appear here; see _features_entry's "not used" note).
UPGRADE_FEATURE_COLS = (
    "fiscal_year", "fy_total", "fy_total_prior1", "fy_total_prior2", "fy_trend",
    "largest_gift", "gift_count", "consecutive_years_given", "months_since_last_gift",
)


def upgrade_drivers_synthetic(seeds, n_donors, n_years, threshold=1000.0, band=(100.0, 999.0)) -> Dict[str, Any]:
    feature_cols = UPGRADE_FEATURE_COLS[1:]  # bench_upgrade drops "fiscal_year" too (#283)

    def build(seed):
        panel = make_donor_panel(n_donors=n_donors, n_years=n_years, random_state=seed)
        years = sorted(panel["gifts"]["fiscal_year"].unique())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            snaps = bm.build_leadership_snapshots(panel["gifts"], fiscal_years=years[:-1], threshold=threshold, band=band)
        X = snaps[list(feature_cols)].to_numpy(dtype="float64")
        y = snaps["target"].to_numpy()
        fy = snaps["fiscal_year"].to_numpy()
        splitter = bm.FiscalYearGroupedSplitter(n_splits=1, drop_repeat_donors=False)
        train_idx, test_idx = list(splitter.split(X, groups=fy))[-1]
        return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    drivers = _top_drivers(lambda seed: bm.MajorGiftClassifier(random_state=seed), seeds, build, feature_cols, scoring="roc_auc")
    return _features_entry(feature_cols, drivers, scoring="roc_auc", split="walk-forward (FiscalYearGroupedSplitter, last split)")


def planned_giving_drivers_synthetic(seeds, n_donors, n_years) -> Dict[str, Any]:
    feature_cols = ("total", "n", "recent")

    def build(seed):
        train, test = bm._train_test_periods(bm._period_panel(n_donors, n_years, seed))
        return (
            train[list(feature_cols)].to_numpy(), train["y_response"].to_numpy(),
            test[list(feature_cols)].to_numpy(), test["y_response"].to_numpy(),
        )

    drivers = _top_drivers(lambda seed: PlannedGivingIntentScorer(random_state=seed), seeds, build, feature_cols, scoring="roc_auc")
    return _features_entry(feature_cols, drivers, scoring="roc_auc", split=bm.SYNTHETIC_SPLIT)


# --------------------------------------------------------------------------- #
# KDD Cup 1998 driver functions: same `_top_drivers` helper, one real fitted
# model (KDD_SEED, no 5-seed range: one real file has no second draw), the
# same feature construction each bench_kdd_* function above already uses.
# --------------------------------------------------------------------------- #
def response_drivers_kdd98(seed) -> Dict[str, Any]:
    feature_cols = ("recency", "frequency", "monetary", "tenure")

    def build(_seed):
        X_rfm, donors_idx = bm._kdd_response_frame(bm.fetch_kdd98_donors())
        y = donors_idx["TARGET_B"].to_numpy()
        idx_train, _idx_val, idx_test = bm._split_55_15_30(len(y), y, seed)
        Xtr, Xte = X_rfm.iloc[idx_train][list(feature_cols)], X_rfm.iloc[idx_test][list(feature_cols)]
        return Xtr.to_numpy(), y[idx_train], Xte.to_numpy(), y[idx_test]

    drivers = _top_drivers(lambda s: bm.MajorGiftClassifier(random_state=s), [seed], build, feature_cols, scoring="roc_auc")
    return _features_entry(feature_cols, drivers, scoring="roc_auc", split=bm.KDD_SPLIT)


def lapse_drivers_kdd98(seed) -> Dict[str, Any]:
    feature_cols = ("total", "n", "recent")

    def build(_seed):
        panel = bm._kdd_lapse_panel(bm.fetch_kdd98_donors())
        last_period = panel["period"].max()
        val_period = last_period - 1
        train_p = panel[panel["period"] < val_period]
        test_p = panel[panel["period"] == last_period]
        return (
            train_p[list(feature_cols)].to_numpy(), train_p["lapsed"].to_numpy(),
            test_p[list(feature_cols)].to_numpy(), test_p["lapsed"].to_numpy(),
        )

    drivers = _top_drivers(
        lambda s: bm.LapsePredictor(n_estimators=100, max_depth=10, random_state=s),
        [seed], build, feature_cols, scoring="roc_auc",
    )
    return _features_entry(feature_cols, drivers, scoring="roc_auc", split="walk-forward across KDD98 promotion periods (train < period N-1, val=N-1, test=N)")


def ask_drivers_kdd98(seed) -> Dict[str, Any]:
    feature_cols = KDD_FEATURE_COLS

    def build(_seed):
        donors = bm.fetch_kdd98_donors()
        rfm = bm._kdd_rfm(bm._kdd_gift_log(donors))
        Xd_train, _Xd_val, Xd_test, yd_train, _yd_val, yd_test = bm._kdd_ask_design(donors, rfm, seed)
        resp_train, resp_test = yd_train > 0, yd_test > 0
        # Kept as a DataFrame (not `.to_numpy()`): WealthScreeningImputer
        # matches `wealth_cols` by column name and silently skips them on a
        # bare array, which would leave WEALTH1/WEALTH2/INCOME unimputed.
        return (
            Xd_train.loc[resp_train, list(feature_cols)].astype("float64"), yd_train[resp_train].to_numpy(),
            Xd_test.loc[resp_test, list(feature_cols)].astype("float64"), yd_test[resp_test].to_numpy(),
        )

    make_model = lambda s: bm.make_pipeline(  # noqa: E731
        bm.WealthScreeningImputer(wealth_cols=["WEALTH1", "WEALTH2", "INCOME"]), bm.AskAmountRecommender(random_state=s),
    )
    drivers = _top_drivers(make_model, [seed], build, feature_cols, scoring="neg_mean_absolute_error")
    return _features_entry(
        feature_cols, drivers, scoring="neg_mean_absolute_error", split=bm.KDD_SPLIT, target_label="suggested ask",
    )


def upgrade_drivers_kdd98(seed, threshold=50.0, band=(5.0, 49.0)) -> Dict[str, Any]:
    feature_cols = UPGRADE_FEATURE_COLS[1:]  # bench_kdd_upgrade drops "fiscal_year"

    def build(_seed):
        donors = bm.fetch_kdd98_donors()
        gifts = bm._kdd_gift_log(donors)
        train_fy, test_fy = 1994, 1995
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            snaps = bm.build_leadership_snapshots(
                gifts, fiscal_years=[train_fy, test_fy], threshold=threshold, band=band, fiscal_year_start=7,
            )
        train, test = snaps[snaps["fiscal_year"] == train_fy], snaps[snaps["fiscal_year"] == test_fy]
        return (
            train[list(feature_cols)].to_numpy(dtype="float64"), train["target"].to_numpy(),
            test[list(feature_cols)].to_numpy(dtype="float64"), test["target"].to_numpy(),
        )

    drivers = _top_drivers(lambda s: bm.MajorGiftClassifier(random_state=s), [seed], build, feature_cols, scoring="roc_auc")
    return _features_entry(feature_cols, drivers, scoring="roc_auc", split="walk-forward by fiscal year (train FY1994, test FY1995)")


def who_to_mail_drivers_kdd98(seed) -> Dict[str, Any]:
    """Drivers for the response half of the mail/no-mail decision only: "mail
    if E[gift] > cost" multiplies a response probability by a suggested gift
    (:data:`ask_drivers_kdd98` has that half's own drivers), so there is no
    single feature-scored ranking to attribute as one model."""
    feature_cols = KDD_FEATURE_COLS

    def build(_seed):
        donors = bm.fetch_kdd98_donors()
        rfm = bm._kdd_rfm(bm._kdd_gift_log(donors))
        Xd_train, _Xd_val, Xd_test, yd_train, _yd_val, yd_test = bm._kdd_ask_design(donors, rfm, seed)
        ytr, yte = (yd_train > 0).astype(int).to_numpy(), (yd_test > 0).astype(int).to_numpy()
        # Kept as a DataFrame, same reason as ask_drivers_kdd98 above.
        return Xd_train[list(feature_cols)].astype("float64"), ytr, Xd_test[list(feature_cols)].astype("float64"), yte

    make_model = lambda s: bm.make_pipeline(  # noqa: E731
        bm.WealthScreeningImputer(wealth_cols=["WEALTH1", "WEALTH2", "INCOME"]), bm.MajorGiftClassifier(random_state=s),
    )
    drivers = _top_drivers(make_model, [seed], build, feature_cols, scoring="roc_auc")
    return _features_entry(feature_cols, drivers, scoring="roc_auc", split=bm.KDD_SPLIT)


# --------------------------------------------------------------------------- #
# "What the model looks at" tab group: one markdown snippet per model,
# included into its results page with pymdownx.snippets (E.11i pasted spec).
# One tab per dataset the model has a "features" entry for, in the order
# each page already presents that dataset.
# --------------------------------------------------------------------------- #
MODEL_DATASET_TABS = {
    "upgrade": [
        ("Sample data", "upgrade_synthetic"),
        ("KDD Cup 1998 (real donor file)", "upgrade_kdd98"),
    ],
    "response": [
        ("KDD Cup 1998 (real donor file)", "response_kdd98"),
        ("cup98VAL (real donor file, never seen by the model)", "response_cup98val"),
        ("Sample data (checks the code runs, not that the model works)", "response_synthetic"),
    ],
    "lapse": [
        ("KDD Cup 1998 (real donor file)", "lapse_kdd98"),
    ],
    "ask": [
        ("KDD Cup 1998 (real donor file)", "ask_kdd98"),
        ("Sample data", "ask_synthetic"),
    ],
    "planned_giving": [
        ("Sample data", "planned_giving_synthetic"),
    ],
    "who_to_mail": [
        ("KDD Cup 1998 (real donor file)", "who_to_mail_kdd98"),
        ("cup98VAL (real donor file, never seen by the model)", "who_to_mail_cup98val"),
    ],
}


def _fmt_pct_range(block: Dict[str, Any]) -> str:
    v = block["value"] if "value" in block else block
    lo, hi = block.get("lo"), block.get("hi")
    base = f"{v:.0f} of 100"
    if lo is not None and hi is not None:
        base += f" (between {lo:.0f} and {hi:.0f})"
    return base


def _render_group_table(columns: Sequence[str], dataset_category: str) -> List[str]:
    present = sorted({FEATURE_INFO[c][1] for c in columns}, key=GROUP_ORDER.index)
    lines = ["    | What it knows | What that means |", "    |---|---|"]
    for g in present:
        lines.append(f"    | {GROUP_LABELS[g]} | {GROUP_MEANINGS[g]} |")
    available = DATASET_GROUPS_AVAILABLE[dataset_category]
    # "momentum" is a computed year-over-year comparison, not a record type a
    # file has or lacks (and the KDD98 upgrade tab already uses it on the
    # same file other KDD98 tabs would otherwise call "absent"), so it is
    # never part of this file-coverage sentence either way.
    candidates = [g for g in GROUP_ORDER if g not in present and g != "momentum"]
    truly_absent = [g for g in candidates if g not in available]
    available_unused = [g for g in candidates if g in available]
    # Two different, non-overlapping claims: a group this dataset's file
    # never has at all (truly_absent) versus one the file does have but this
    # particular benchmarked model isn't fed (available_unused) - conflating
    # them would say, falsely, that e.g. KDD98 has no wealth data.
    if truly_absent or available_unused:
        lines.append("")
    if truly_absent:
        names = _join_english([GROUP_LABELS[g].lower() for g in truly_absent])
        verb = "it is" if len(truly_absent) == 1 else "they are"
        lines.append(f"    This file has no {names} records, so {verb} not used here.")
    if available_unused:
        names = _join_english([GROUP_LABELS[g].lower() for g in available_unused])
        verb, pron = ("is", "it") if len(available_unused) == 1 else ("are", "them")
        lines.append(f"    {names.capitalize()} {verb} in this file, but this model is not given {pron} here.")
    return lines


def _render_driver_table(drivers: List[Dict[str, Any]], target_label: str) -> Tuple[List[str], str | None]:
    # Only features with a measurable effect (importance, or for a multi-seed
    # range its lower bound, strictly above zero) are shown as "drivers" - a
    # 0.000 permutation importance is noise, not something that "raises or
    # lowers" anything, and showing it as a ranked driver overstates it. The
    # full, unfiltered list (including zero-effect features) stays in the
    # analyst note.
    visible = [d for d in drivers if (d["lo"] if d["lo"] is not None else d["importance"]) > 0]
    if not visible:
        return (
            ["    No single feature here had a measurable effect on its own; see the analyst note below for the full list."],
            None,
        )
    lines = [f"    | What raises or lowers the {target_label} | |", "    |---|---|"]
    for d in visible:
        symbol = _DIRECTION_SYMBOL[d["direction"]]
        phrase = _direction_phrase(d["direction"], target_label)
        lines.append(f"    | {d['label']} | {symbol} {phrase} |")
    note = None
    if len(visible) < len(drivers):
        noun = "feature" if len(visible) == 1 else "features"
        note = f"    Only {len(visible)} {noun} had a measurable effect; the rest are in the analyst note below."
    return lines, note


def _render_ablation_table(entry: Dict[str, Any], result_block: Dict[str, Any]) -> List[str] | None:
    extra = [s for s in entry["sets"] if not s["chosen"]]
    if not extra:
        return None
    lines = [
        "    ### What adding information did", "",
        "    | What the model saw | Top 1% | Top 5% | Top 10% |", "    |---|---|---|---|",
        f"    | {FEATURE_SET_LABELS['current']} | "
        + " | ".join(f"{result_block[f'top{p}pct']['model']:.0f} of 100" for p in (1, 5, 10)) + " |",
    ]
    for s in extra:
        lines.append(f"    | {s['label']} | " + " | ".join(_fmt_pct_range(s[f"top{p}pct"]) for p in (1, 5, 10)) + " |")
    lines.append(
        "    | the best simple rule (for comparison) | "
        + " | ".join(f"{result_block[f'top{p}pct']['rule']:.0f} of 100" for p in (1, 5, 10)) + " |"
    )
    cur10, full10 = result_block["top10pct"]["model"], extra[0]["top10pct"]["value"]
    gap = full10 - cur10
    if abs(gap) < 1.5:
        takeaway = "The extra signals made about the same difference at the top of the list."
    elif gap > 0:
        takeaway = f"The extra signals raised the top-10% hit rate from {cur10:.0f} of 100 to {full10:.0f} of 100."
    else:
        takeaway = f"The extra signals lowered the top-10% hit rate, from {cur10:.0f} of 100 to {full10:.0f} of 100."
    lines += ["", f"    {takeaway}"]
    return lines


def _render_analyst_note(entry: Dict[str, Any]) -> List[str]:
    current = entry["sets"][0]
    lines = ['    ??? note "For analysts"']
    lines.append(f"        Columns: {', '.join(f'`{c}`' for c in current['columns'])}")
    lines.append("")
    lines.append("        | Column | Importance | Direction |")
    lines.append("        |---|---|---|")
    for d in entry["drivers"]:
        imp = f"{d['importance']:.3f}"
        if d["lo"] is not None and d["hi"] is not None:
            imp += f" (range {d['lo']:.3f} to {d['hi']:.3f})"
        lines.append(f"        | `{d['column']}` | {imp} | {d['direction']} |")
    lines.append("")
    note_parts = [
        f"Method: permutation importance (`{entry['scoring']}`), partial-dependence sign for direction "
        '("mixed" if it changes sign).',
        f"Split: {entry['split']}.",
    ]
    if entry["n_configs"] != 1:
        note_parts.append(f"Configurations compared: {entry['n_configs']}.")
    note_parts.append(f"Git SHA: `{entry['git_sha']}`.")
    lines.append("        " + " ".join(note_parts))
    if "reused_from_note" in entry:
        lines.append(f"        {entry['reused_from_note']}")
    return lines


def render_feature_snippets(results: Dict[str, Any], out_dir: Path) -> None:
    """Writes one self-contained tab block per (model, dataset) to
    ``docs/results/_features/<model>__<dataset_key>.md``, split per dataset
    rather than one file per model, so the page's tab group drops in a whole
    number of tabs: a fast synthetic-only regeneration (no KDD download)
    never has to touch, and so can never accidentally drop, an already
    real-data tab it did not recompute. Each model's results page lists its
    tabs' files with consecutive ``--8<--`` includes, in the same order as
    :data:`MODEL_DATASET_TABS`, which pymdownx.tabbed reads as one group."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for model, tabs in MODEL_DATASET_TABS.items():
        for label, key in tabs:
            if key not in results or "features" not in results[key]:
                continue
            result_block, entry = results[key], results[key]["features"]
            n = len(entry["sets"][0]["columns"])
            dataset_category = _dataset_category(key)
            lines = [f'=== "{label}"', ""]
            lines.append(f"    On this file the model looks at {n} things about each donor.")
            lines.append("")
            lines += _render_group_table(entry["sets"][0]["columns"], dataset_category)
            lines.append("")
            lines.append("    These matter most:")
            lines.append("")
            # Fall back to the model name when re-rendering an already-stored
            # results.json entry that predates the "target_label" field (a
            # snippets-only regeneration reuses the committed numbers as-is).
            target_label = entry.get("target_label") or ("suggested ask" if model == "ask" else "score")
            driver_lines, driver_note = _render_driver_table(entry["drivers"], target_label)
            lines += driver_lines
            if driver_note:
                lines.append("")
                lines.append(driver_note)
            ablation = _render_ablation_table(entry, result_block)
            if ablation:
                lines.append("")
                lines += ablation
            lines.append("")
            lines += _render_analyst_note(entry)
            lines.append("")
            lines.append("    Results on your own file will differ.")
            (out_dir / f"{model}__{key}.md").write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-kdd98", action="store_true", help="Also run the KDD Cup 1998 section (downloads ~36MB).")
    parser.add_argument(
        "--with-cup98val", action="store_true",
        help="Also score who-to-mail on KDD98's own held-out cup98VAL+valtargt file "
        "(a second ~37MB download; opt-in). No effect without --with-kdd98.",
    )
    args = parser.parse_args()

    results: Dict[str, Any] = {}

    # --- synthetic: response / major gift -----------------------------------
    resp_rows = bm.bench_response(SEEDS, N_DONORS, N_YEARS)
    rand_resp = random_rate_response() * 100
    resp_row_by_p = {p: _row(resp_rows, "MajorGiftClassifier", f"top{p}pct_hit_rate") for p in (1, 5, 10)}
    results["response_synthetic"] = {
        f"top{p}pct": {
            "model": resp_row_by_p[p].value * 100,
            "rule": resp_row_by_p[p].baseline * 100,
            "random": rand_resp,
        }
        for p in (1, 5, 10)
    }
    results["response_synthetic"]["verdict"] = resp_row_by_p[10].verdict
    results["response_synthetic"]["features"] = response_drivers_synthetic(SEEDS, N_DONORS, N_YEARS)
    r10 = results["response_synthetic"]["top10pct"]
    _render_themed(
        _hbar_chart,
        OUT_DIR / "response.png",
        ["Top 1%", "Top 5%", "Top 10%"],
        {
            "Model": [results["response_synthetic"][f"top{p}pct"]["model"] for p in (1, 5, 10)],
            "Best simple rule": [results["response_synthetic"][f"top{p}pct"]["rule"] for p in (1, 5, 10)],
            "Random pick": [results["response_synthetic"][f"top{p}pct"]["random"] for p in (1, 5, 10)],
        },
        {"Model": "model", "Best simple rule": "rule", "Random pick": "random"},
        title=_takeaway(r10["model"], r10["rule"], "The model", "ranking by past giving"),
        subtitle="Sample donor panel, 5 random draws averaged. Who gave again next year, out of every 100 picked.",
        errors={
            "Model": [_ci(resp_row_by_p[p]) for p in (1, 5, 10)],
            "Best simple rule": [None, None, None],
            "Random pick": [None, None, None],
        },
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
    results["lapse_synthetic"]["features"] = lapse_drivers_synthetic(SEEDS, N_DONORS, N_YEARS)

    # --- synthetic: ask -------------------------------------------------
    ask_rows = bm.bench_ask(SEEDS, N_DONORS, N_YEARS)
    within = _row(ask_rows, "AskAmountRecommender", "within25pct")
    results["ask_synthetic"] = {
        "within25pct_model": within.value * 100,
        "within25pct_last_gift": within.baseline * 100,
        "verdict": within.verdict,
        "features": ask_drivers_synthetic(SEEDS, N_DONORS, N_YEARS),
    }

    # --- synthetic: $1K upgrade (bench_upgrade) ------------------------------
    upgrade_rows = bm.bench_upgrade(SEEDS, N_DONORS, N_YEARS)
    rand_upgrade = random_rate_upgrade() * 100
    upg_row_by_p = {
        p: _row(upgrade_rows, "upgrade_model (MajorGiftClassifier)", f"top{p}pct_hit_rate") for p in (1, 5, 10)
    }
    results["upgrade_synthetic"] = {
        f"top{p}pct": {
            "model": upg_row_by_p[p].value * 100,
            "rule": upg_row_by_p[p].baseline * 100,
            "random": rand_upgrade,
        }
        for p in (1, 5, 10)
    }
    results["upgrade_synthetic"]["verdict"] = upg_row_by_p[10].verdict
    results["upgrade_synthetic"]["features"] = upgrade_drivers_synthetic(SEEDS, N_DONORS, N_YEARS)
    u10 = results["upgrade_synthetic"]["top10pct"]
    _render_themed(
        _hbar_chart,
        OUT_DIR / "upgrade_topn.png",
        ["Top 1%", "Top 5%", "Top 10%"],
        {
            "Model": [results["upgrade_synthetic"][f"top{p}pct"]["model"] for p in (1, 5, 10)],
            "Best simple rule": [results["upgrade_synthetic"][f"top{p}pct"]["rule"] for p in (1, 5, 10)],
            "Random pick": [results["upgrade_synthetic"][f"top{p}pct"]["random"] for p in (1, 5, 10)],
        },
        {"Model": "model", "Best simple rule": "rule", "Random pick": "random"},
        title=_takeaway(u10["model"], u10["rule"], "The model", "ranking by this year's giving"),
        subtitle="Sample donor panel, 5 random draws averaged. Who crossed $1,000 next year, out of every 100 picked.",
        errors={
            "Model": [_ci(upg_row_by_p[p]) for p in (1, 5, 10)],
            "Best simple rule": [None, None, None],
            "Random pick": [None, None, None],
        },
    )

    # --- synthetic: planned giving (coverage vs. chance) ---------------------
    # No chart: this scores a stand-in label (giving-response), not real
    # bequest intent, so a bar chart of it would misrepresent the model as
    # tested on planned giving. See docs/results/planned_giving.md.
    results["planned_giving_synthetic"] = planned_giving_hit_rates()
    results["planned_giving_synthetic"]["features"] = planned_giving_drivers_synthetic(SEEDS, N_DONORS, N_YEARS)

    # --- $1K upgrade worked example ------------------------------------------
    ex = upgrade_worked_example(SEEDS[0])
    results["upgrade_worked_example"] = ex

    davg = upgrade_decile_average(SEEDS)
    results["upgrade_deciles_avg"] = davg
    overall_rate = davg["overall_mean"]
    top_decile_rate = davg["mean"][0]
    _render_themed(
        _hbar_chart,
        OUT_DIR / "upgrade_deciles.png",
        [f"D{d}" for d in davg["decile"]],
        {"Upgrade rate": davg["mean"]},
        {"Upgrade rate": "model"},
        title=f"The top decile (D1) upgrades at {top_decile_rate:.0f} of 100, against {overall_rate:.0f} of 100 overall",
        subtitle=f"{davg['n_seeds']} random draws averaged (D1 = the 10% the model liked most, D10 = the 10% it liked least). Dashed line: the overall rate.",
        value_fmt=lambda v: f"{v:.0f}",
        base_rate=overall_rate,
        base_rate_label=f"overall: {overall_rate:.0f} of 100",
        errors={"Upgrade rate": list(zip(davg["lo"], davg["hi"]))},
    )

    # --- KDD98 (opt-in) --------------------------------------------------
    if args.with_kdd98:
        seed = bm.KDD_SEED
        kdd_resp = bm.bench_kdd_response(seed)
        kdd_lapse = bm.bench_kdd_lapse(seed)
        kdd_ask = bm.bench_kdd_ask(seed)
        kdd_cost = bm.bench_kdd_cost_aware(seed)

        resp_kdd_row_by_p = {p: _row(kdd_resp, "MajorGiftClassifier", f"top{p}pct_hit_rate") for p in (1, 5, 10)}
        results["response_kdd98"] = {
            f"top{p}pct": {"model": resp_kdd_row_by_p[p].value * 100, "rule": resp_kdd_row_by_p[p].baseline * 100}
            for p in (1, 5, 10)
        }
        results["response_kdd98"]["verdict"] = resp_kdd_row_by_p[10].verdict
        results["response_kdd98"]["features"] = response_drivers_kdd98(seed)
        rk10 = results["response_kdd98"]["top10pct"]
        _render_themed(
            _hbar_chart,
            OUT_DIR / "response_kdd98.png",
            ["Top 1%", "Top 5%", "Top 10%"],
            {
                "Model": [results["response_kdd98"][f"top{p}pct"]["model"] for p in (1, 5, 10)],
                "Best simple rule": [results["response_kdd98"][f"top{p}pct"]["rule"] for p in (1, 5, 10)],
            },
            {"Model": "model", "Best simple rule": "rule"},
            title=_takeaway(rk10["model"], rk10["rule"], "The model", "the best of lifetime giving, RFM and RFA_2"),
            subtitle="KDD Cup 1998, held-out 30% of the file. Who gave again, out of every 100 picked.",
            errors={
                "Model": [_ci(resp_kdd_row_by_p[p]) for p in (1, 5, 10)],
                "Best simple rule": [None, None, None],
            },
        )

        # --- $1K upgrade on KDD98 (rescaled $50/$5-49 threshold) ------------
        kdd_upgrade = bm.bench_kdd_upgrade(seed)
        upg_kdd_row_by_p = {
            p: _row(kdd_upgrade, "upgrade_model (MajorGiftClassifier)", f"top{p}pct_hit_rate") for p in (1, 5, 10)
        }
        results["upgrade_kdd98"] = {
            f"top{p}pct": {"model": upg_kdd_row_by_p[p].value * 100, "rule": upg_kdd_row_by_p[p].baseline * 100}
            for p in (1, 5, 10)
        }
        results["upgrade_kdd98"]["verdict"] = upg_kdd_row_by_p[10].verdict
        results["upgrade_kdd98"]["features"] = upgrade_drivers_kdd98(seed)
        uk10 = results["upgrade_kdd98"]["top10pct"]
        _render_themed(
            _hbar_chart,
            OUT_DIR / "upgrade_kdd98.png",
            ["Top 1%", "Top 5%", "Top 10%"],
            {
                "Model": [results["upgrade_kdd98"][f"top{p}pct"]["model"] for p in (1, 5, 10)],
                "Best simple rule": [results["upgrade_kdd98"][f"top{p}pct"]["rule"] for p in (1, 5, 10)],
            },
            {"Model": "model", "Best simple rule": "rule"},
            title=_takeaway(uk10["model"], uk10["rule"], "The model", "the largest single gift in the band"),
            subtitle="KDD Cup 1998, threshold rescaled to $50 (this file's gifts are far smaller than a major-gift program's). Crossed $50 next year, out of every 100 picked.",
            errors={
                "Model": [_ci(upg_kdd_row_by_p[p]) for p in (1, 5, 10)],
                "Best simple rule": [None, None, None],
            },
        )

        lapse_top = {p: _row(kdd_lapse, "LapsePredictor", f"top{p}pct_hit_rate") for p in (1, 5, 10)}
        donors = bm.fetch_kdd98_donors()
        base_rate_lapse_kdd = float((donors["TARGET_B"].to_numpy() == 0).mean()) * 100
        results["lapse_kdd98"] = {
            "base_rate_pct": base_rate_lapse_kdd,
            "verdict": lapse_top[10].verdict,
            **{f"top{p}pct": {"model": r.value * 100, "rule": r.baseline * 100} for p, r in lapse_top.items()},
        }
        results["lapse_kdd98"]["features"] = lapse_drivers_kdd98(seed)
        _render_themed(
            _neartie_dot_chart,
            OUT_DIR / "lapse_kdd98.png",
            ["Top 1%", "Top 5%", "Top 10%"],
            {
                "Model": [results["lapse_kdd98"][f"top{p}pct"]["model"] for p in (1, 5, 10)],
                "Best simple rule": [results["lapse_kdd98"][f"top{p}pct"]["rule"] for p in (1, 5, 10)],
            },
            {"Model": "model", "Best simple rule": "rule"},
            base_rate=base_rate_lapse_kdd,
            title="Almost everyone lapses here, so no list beats picking at random by much",
            subtitle="KDD Cup 1998. Lapsed next period, out of every 100 picked.",
        )

        # --- the retention read: the bottom decile by lapse score -----------
        kdd_retention = bm.bench_kdd_lapse_retention(seed)
        ret_row_by_p = {p: _row(kdd_retention, "LapsePredictor", f"top{p}pct_hit_rate") for p in (1, 5, 10)}
        retention_base_rate = 100.0 - base_rate_lapse_kdd
        results["lapse_kdd98_retention"] = {
            f"top{p}pct": {"model": ret_row_by_p[p].value * 100, "rule": ret_row_by_p[p].baseline * 100}
            for p in (1, 5, 10)
        }
        results["lapse_kdd98_retention"]["verdict"] = ret_row_by_p[10].verdict
        rt10 = results["lapse_kdd98_retention"]["top10pct"]
        rt_gap = abs(rt10["model"] - rt10["rule"])
        if rt_gap < 1.5:
            rt_title = "The model and the rule find about the same retained group here"
        elif rt10["model"] > rt10["rule"]:
            rt_title = f"The model's least-likely-to-lapse 10% retains better: {rt10['model']:.0f} of 100 vs {rt10['rule']:.0f} of 100"
        else:
            rt_title = f"Years since last gift still finds a better group: {rt10['rule']:.0f} of 100 vs {rt10['model']:.0f} of 100"
        _render_themed(
            _hbar_chart,
            OUT_DIR / "lapse_kdd98_retention.png",
            ["Top 1%", "Top 5%", "Top 10%"],
            {
                "Model": [results["lapse_kdd98_retention"][f"top{p}pct"]["model"] for p in (1, 5, 10)],
                "Best simple rule": [results["lapse_kdd98_retention"][f"top{p}pct"]["rule"] for p in (1, 5, 10)],
            },
            {"Model": "model", "Best simple rule": "rule"},
            title=rt_title,
            subtitle="KDD Cup 1998, the 10% least likely to lapse by model score. Gave again, out of every 100 in that group.",
            base_rate=retention_base_rate,
            base_rate_label=f"everyone: {retention_base_rate:.0f} of 100",
            errors={
                "Model": [_ci(ret_row_by_p[p]) for p in (1, 5, 10)],
                "Best simple rule": [None, None, None],
            },
        )

        ask_row = _row(kdd_ask, "AskAmountRecommender", "within25pct")
        results["ask_kdd98"] = {
            "within25pct_model": ask_row.value * 100,
            "within25pct_last_gift": ask_row.baseline * 100,
            "verdict": ask_row.verdict,
            "features": ask_drivers_kdd98(seed),
        }
        _render_themed(
            _hbar_chart,
            OUT_DIR / "ask_kdd98.png",
            ["Suggested ask"],
            {
                "Model": [results["ask_kdd98"]["within25pct_model"]],
                "Best simple rule": [results["ask_kdd98"]["within25pct_last_gift"]],
            },
            {"Model": "model", "Best simple rule": "rule"},
            title=_takeaway(
                results["ask_kdd98"]["within25pct_model"], results["ask_kdd98"]["within25pct_last_gift"],
                "The model", "the higher of last gift and average gift",
            ),
            subtitle="KDD Cup 1998. Suggested amounts landing within 25% of what the donor actually gave.",
        )

        net_row = _row(kdd_cost, "cost_aware_selection", "net_revenue")
        results["who_to_mail_kdd98"] = {
            "net_revenue_model": net_row.value,
            "net_revenue_mail_everyone": net_row.baseline,
            "verdict": net_row.verdict,
            "note": net_row.note,
            "features": who_to_mail_drivers_kdd98(seed),
        }
        curve = bm.kdd_mail_profit_curve(seed)
        results["who_to_mail_kdd98"]["curve"] = curve
        gain = curve["stop_net_revenue"] - curve["everyone_net_revenue"]
        skip = curve["n_total"] - curve["stop_k"]
        _render_themed(
            _profit_curve_chart,
            OUT_DIR / "who_to_mail.png",
            curve["mailed"], curve["net_revenue"],
            curve["stop_k"], curve["stop_net_revenue"],
            curve["n_total"], curve["everyone_net_revenue"],
            title=f"Mailing only likely responders raised ${gain:,.0f} more, skipping {skip:,} of {curve['n_total']:,} letters",
            subtitle="KDD Cup 1998. Net revenue after mailing cost, mail if expected gift beats the $0.68 cost.",
        )

        # --- who to mail, scored on KDD98's own held-out validation file ----
        if args.with_cup98val:
            kdd_cost_val = bm.bench_kdd_cost_aware_val(seed)
            net_row_val = _row(kdd_cost_val, "cost_aware_selection", "net_revenue")
            results["who_to_mail_cup98val"] = {
                "net_revenue_model": net_row_val.value,
                "net_revenue_mail_everyone": net_row_val.baseline,
                "verdict": net_row_val.verdict,
                "note": net_row_val.note,
                "features": _reused_features(
                    results["who_to_mail_kdd98"]["features"],
                    "Same response model and features as the KDD Cup 1998 tab; cup98VAL supplies new test "
                    "donors on the same columns, not new columns.",
                ),
            }
            curve_val = bm.kdd_mail_profit_curve_val(seed)
            results["who_to_mail_cup98val"]["curve"] = curve_val
            gain_val = curve_val["stop_net_revenue"] - curve_val["everyone_net_revenue"]
            skip_val = curve_val["n_total"] - curve_val["stop_k"]
            _render_themed(
                _profit_curve_chart,
                OUT_DIR / "who_to_mail_cup98val.png",
                curve_val["mailed"], curve_val["net_revenue"],
                curve_val["stop_k"], curve_val["stop_net_revenue"],
                curve_val["n_total"], curve_val["everyone_net_revenue"],
                title=(
                    f"On a file the model never saw, mailing smarter still raised ${gain_val:,.0f} more, "
                    f"skipping {skip_val:,} of {curve_val['n_total']:,} letters"
                ),
                subtitle="cup98VAL, KDD Cup 1998's own held-out file: 96,367 donors never touched during fitting.",
            )

            # --- response, scored on the same held-out file -----------------
            kdd_val = bm.bench_kdd_val_models(seed)
            val_row_by_p = {p: _row(kdd_val, "MajorGiftClassifier", f"top{p}pct_hit_rate") for p in (1, 5, 10)}
            results["response_cup98val"] = {
                f"top{p}pct": {"model": val_row_by_p[p].value * 100, "rule": val_row_by_p[p].baseline * 100}
                for p in (1, 5, 10)
            }
            results["response_cup98val"]["verdict"] = val_row_by_p[10].verdict
            results["response_cup98val"]["features"] = _reused_features(
                results["response_kdd98"]["features"],
                "Same model and features as the KDD Cup 1998 tab; cup98VAL supplies new test donors on the "
                "same columns, not new columns.",
            )
            rv10 = results["response_cup98val"]["top10pct"]
            _render_themed(
                _hbar_chart,
                OUT_DIR / "response_cup98val.png",
                ["Top 1%", "Top 5%", "Top 10%"],
                {
                    "Model": [results["response_cup98val"][f"top{p}pct"]["model"] for p in (1, 5, 10)],
                    "Best simple rule": [results["response_cup98val"][f"top{p}pct"]["rule"] for p in (1, 5, 10)],
                },
                {"Model": "model", "Best simple rule": "rule"},
                title=_takeaway(rv10["model"], rv10["rule"], "The model", "the best simple rule"),
                subtitle="cup98VAL, 96,367 donors never touched during fitting. Who gave again, out of every 100 picked.",
                errors={
                    "Model": [_ci(val_row_by_p[p]) for p in (1, 5, 10)],
                    "Best simple rule": [None, None, None],
                },
            )

    # --- scoreboard: one row per question, one dot per dataset -------------
    def _dots(*pairs):
        return [(label, results[key]["verdict"]) for label, key in pairs if key in results]

    scoreboard_rows = [
        ("Leadership upgrade", _dots(("Sample data", "upgrade_synthetic"), ("KDD Cup 1998", "upgrade_kdd98"))),
        (
            "Response", _dots(
                ("Sample data", "response_synthetic"), ("KDD Cup 1998", "response_kdd98"),
                ("cup98VAL", "response_cup98val"),
            ),
        ),
        ("Lapse", _dots(("Sample data", "lapse_synthetic"), ("KDD Cup 1998", "lapse_kdd98"))),
        ("Lapse (retention read)", _dots(("KDD Cup 1998", "lapse_kdd98_retention"))),
        ("Suggested ask", _dots(("Sample data", "ask_synthetic"), ("KDD Cup 1998", "ask_kdd98"))),
        ("Who to mail", _dots(("KDD Cup 1998", "who_to_mail_kdd98"), ("cup98VAL", "who_to_mail_cup98val"))),
        ("Planned giving", [("Sample data", None)]),
    ]
    _render_themed(_scoreboard_chart, OUT_DIR / "scoreboard.png", scoreboard_rows)

    results["_env"] = {
        "git_sha": _git_sha(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
    }
    with open(OUT_DIR / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    render_feature_snippets(results, ROOT / "docs" / "results" / "_features")
    print(f"Wrote {OUT_DIR / 'results.json'} and PNGs to {OUT_DIR}")


if __name__ == "__main__":
    main()
