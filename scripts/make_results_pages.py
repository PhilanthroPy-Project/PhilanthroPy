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
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas  # noqa: E402
import sklearn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_models_vs_baselines as bm  # noqa: E402

from philanthropy.datasets import make_donor_panel  # noqa: E402
from philanthropy.models import PlannedGivingIntentScorer, score_upgrade_prospects  # noqa: E402


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
def upgrade_worked_example(seed: int) -> Dict[str, Any]:
    """One seed's worth of the public `score_upgrade_prospects` API, for the
    donor-count narrative on the Upgrade page. Uses the first of `SEEDS`
    (the same seeds `bench_upgrade`'s 5-seed average uses) rather than an
    unrelated fixed seed, so the worked example is traceable to the same
    population the top-of-page chart summarises, even though it is a
    different pipeline (the shipped function, not the benchmark's own
    feature set) and so a different single number."""
    panel = make_donor_panel(n_donors=N_DONORS, n_years=N_YEARS, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, report = score_upgrade_prospects(panel["gifts"], random_state=seed)
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
    """Averages `score_upgrade_prospects`'s per-decile upgrade rate across
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
            _, report = score_upgrade_prospects(panel["gifts"], random_state=seed)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-kdd98", action="store_true", help="Also run the KDD Cup 1998 section (downloads ~36MB).")
    parser.add_argument(
        "--with-cup98val", action="store_true",
        help="Also score who-to-mail on KDD98's own held-out cup98VAL+valtargt file "
        "(a second ~37MB download; opt-in). No effect without --with-kdd98.",
    )
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

    # --- DonorsChoose / PSID (opt-in, local files; numbers only, no charts
    # or nav wiring here, those are handled separately) ----------------------
    def _classifier_entry(rows: List, model: str, momentum: bool, extra_meta: Dict[str, Any]):
        row_by_p = {p: _row(rows, model, f"top{p}pct_hit_rate") for p in (1, 5, 10)}
        if row_by_p[10] is None:
            return None
        entry = {
            f"top{p}pct": {"model": row_by_p[p].value * 100, "rule": row_by_p[p].baseline * 100}
            for p in (1, 5, 10)
        }
        entry["verdict"] = row_by_p[10].verdict
        entry["roc_auc"] = _row(rows, model, "roc_auc").value
        entry["metadata"] = {"momentum": momentum, **extra_meta}
        return entry

    def _ask_entry(rows: List, model: str, momentum: bool, extra_meta: Dict[str, Any]):
        mae_row = _row(rows, model, "mae")
        if mae_row is None:
            return None
        within_row = _row(rows, model, "within25pct")
        return {
            "mae_model": mae_row.value, "mae_rule": mae_row.baseline,
            "within25pct_model": within_row.value * 100, "within25pct_rule": within_row.baseline * 100,
            "verdict": mae_row.verdict,
            "metadata": {"momentum": momentum, **extra_meta},
        }

    def _donorschoose_fold_meta(kind: str, threshold: float = 1000.0, band: tuple = (100.0, 999.0)) -> Dict[str, Any]:
        gifts = bm._donorschoose_gift_log(args.donorschoose_path, bm.DONORSCHOOSE_SUBSAMPLE, bm.DONORSCHOOSE_SEED)
        snap = bm._gift_log_period_snapshots(gifts, 7, kind, False, threshold, band)
        if snap.empty:
            return {"subsample": bm.DONORSCHOOSE_SUBSAMPLE, "seed": bm.DONORSCHOOSE_SEED, "fold_years": [], "n_per_fold": [], "base_rate_pct": None}
        test_years = bm._walk_forward_test_periods(snap, "fiscal_year", bm.DONORSCHOOSE_N_FOLDS)
        test = snap[snap["fiscal_year"].isin(test_years)]
        return {
            "subsample": bm.DONORSCHOOSE_SUBSAMPLE, "seed": bm.DONORSCHOOSE_SEED,
            "fold_years": test_years, "n_per_fold": [int((test["fiscal_year"] == t).sum()) for t in test_years],
            "base_rate_pct": float(test["target"].mean()) * 100 if kind != "ask" else None,
        }

    def _psid_fold_meta(kind: str, threshold: float = 1000.0, band: tuple = (100.0, 999.0)) -> Dict[str, Any]:
        snap = bm._psid_wave_period_snapshots(args.psid_data, args.psid_do, kind, False, threshold, band)
        if snap.empty:
            return {"seed": bm.PSID_SEED, "fold_waves": [], "n_per_fold": [], "base_rate_pct": None}
        test_waves = bm._walk_forward_test_periods(snap, "wave", bm.PSID_N_FOLDS)
        test = snap[snap["wave"].isin(test_waves)]
        return {
            "seed": bm.PSID_SEED, "fold_waves": test_waves,
            "n_per_fold": [int((test["wave"] == w).sum()) for w in test_waves],
            "base_rate_pct": float(test["target"].mean()) * 100 if kind != "ask" else None,
        }

    if args.donorschoose_path:
        upgrade_meta = _donorschoose_fold_meta("upgrade")
        lapse_meta = _donorschoose_fold_meta("lapse")
        ask_meta = _donorschoose_fold_meta("ask")
        for momentum in (False, True):
            suffix = "_momentum" if momentum else ""
            up = bm.bench_upgrade_donorschoose(args.donorschoose_path, include_momentum=momentum)
            entry = _classifier_entry(up, "upgrade_model (MajorGiftClassifier)", momentum, upgrade_meta)
            if entry:
                results[f"upgrade_donorschoose{suffix}"] = entry

            lap = bm.bench_lapse_donorschoose(args.donorschoose_path, include_momentum=momentum)
            entry = _classifier_entry(lap, "LapsePredictor", momentum, {**lapse_meta, "note": "84% base rate; see the retention read for the useful list"})
            if entry:
                results[f"lapse_donorschoose{suffix}"] = entry

            ret = bm.bench_lapse_donorschoose_retention(args.donorschoose_path, include_momentum=momentum)
            entry = _classifier_entry(ret, "LapsePredictor", momentum, lapse_meta)
            if entry:
                results[f"lapse_donorschoose_retention{suffix}"] = entry

            ask = bm.bench_ask_donorschoose(args.donorschoose_path, include_momentum=momentum)
            entry = _ask_entry(ask, "AskAmountRecommender", momentum, {
                **ask_meta, "target": "next fiscal-year total, given they give again",
            })
            if entry:
                results[f"ask_donorschoose{suffix}"] = entry

    if args.psid_data and args.psid_do:
        upgrade_meta = _psid_fold_meta("upgrade")
        lapse_meta = _psid_fold_meta("lapse")
        ask_meta = _psid_fold_meta("ask")
        for momentum in (False, True):
            suffix = "_momentum" if momentum else ""
            up = bm.bench_upgrade_psid(args.psid_data, args.psid_do, include_momentum=momentum)
            entry = _classifier_entry(up, "upgrade_model (MajorGiftClassifier)", momentum, upgrade_meta)
            if entry:
                results[f"upgrade_psid{suffix}"] = entry

            lap = bm.bench_lapse_psid(args.psid_data, args.psid_do, include_momentum=momentum)
            entry = _classifier_entry(lap, "LapsePredictor", momentum, lapse_meta)
            if entry:
                results[f"lapse_psid{suffix}"] = entry

            ret = bm.bench_lapse_psid_retention(args.psid_data, args.psid_do, include_momentum=momentum)
            entry = _classifier_entry(ret, "LapsePredictor", momentum, lapse_meta)
            if entry:
                results[f"lapse_psid_retention{suffix}"] = entry

            ask = bm.bench_ask_psid(args.psid_data, args.psid_do, include_momentum=momentum)
            entry = _ask_entry(ask, "AskAmountRecommender", momentum, {
                **ask_meta, "target": "next-wave total, given the household gives again",
            })
            if entry:
                results[f"ask_psid{suffix}"] = entry

    # Models this benchmark cannot honestly answer on either real dataset:
    # neither file has mailing-cost or planned-giving/bequest data.
    results["response_donorschoose_note"] = "No mailing/appeal log in this file, so a response model has nothing to predict response to."
    results["response_psid_note"] = "No mailing/appeal log in this extract, so a response model has nothing to predict response to."
    results["who_to_mail_donorschoose_note"] = "No per-contact mailing cost in this file, so cost-aware selection has no cost side to weigh."
    results["who_to_mail_psid_note"] = "No per-contact mailing cost in this extract, so cost-aware selection has no cost side to weigh."
    results["planned_giving_donorschoose_note"] = "No bequest/estate-intent signal in this file."
    results["planned_giving_psid_note"] = "No bequest/estate-intent signal in this extract."

    # --- scoreboard: one row per question, one dot per dataset -------------
    def _dots(*pairs):
        return [(label, results[key]["verdict"]) for label, key in pairs if key in results]

    scoreboard_rows = [
        ("$1K upgrade", _dots(("Sample data", "upgrade_synthetic"), ("KDD Cup 1998", "upgrade_kdd98"))),
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
    print(f"Wrote {OUT_DIR / 'results.json'} and PNGs to {OUT_DIR}")


if __name__ == "__main__":
    main()
