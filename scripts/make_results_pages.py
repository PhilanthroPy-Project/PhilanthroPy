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


_SCOREBOARD_MARKERS = {"KDD Cup 1998": "s", "cup98VAL": "^", "DonorsChoose": "D", "PSID": "P", "Karlan and List": "v"}
_SCOREBOARD_XLIM = (0.8, 1.6)


def _scoreboard_chart(path: Path, rows: List[tuple], notes: List[str], theme: dict) -> None:
    """One row per question, one dot per real file, placed by lift ratio:
    the model's top-10% result divided by the best simple rule's (1.0 = the
    same; for the ask, the share of suggestions within 25%; for who to mail,
    net revenue against mailing everyone). A thin line through each dot is
    the paired model-minus-rule interval on the same ratio scale; the dot is
    filled in the model colour when the verdict is "beats the rule", in the
    rule colour when it loses, and hollow when the two are about the same.
    Sample data never appears: it is a code check, not a result. Questions
    no real file can test, and any row that mixes reads, are explained in
    the caption lines in ``notes``, not drawn. Marker
    shapes (not just colour) tell files apart for colour-blind readers."""
    n = len(rows)
    row_in = 0.5
    fig_h = _HEADER_IN + row_in * n + 1.25
    fig, ax = plt.subplots(figsize=(_FIG_W, fig_h), dpi=150)
    ax.set_facecolor(theme["surface"])
    fig.patch.set_facecolor(theme["surface"])
    body_in = row_in * n + 1.25
    fig.subplots_adjust(top=body_in / fig_h, left=0.24, right=0.95, bottom=1.0 / fig_h)

    seen_datasets: List[str] = []
    n_wins = n_total = 0
    for i, (_label, dots) in enumerate(rows):
        y = n - 1 - i
        y_jitter = np.linspace(-0.15, 0.15, len(dots)) if len(dots) > 1 else [0.0]
        for (dataset, verdict, ratio, ratio_lo, ratio_hi), dy in zip(dots, y_jitter):
            if dataset not in seen_datasets:
                seen_datasets.append(dataset)
            n_total += 1
            n_wins += verdict == "wins"
            clip = lambda v: float(np.clip(v, *_SCOREBOARD_XLIM))  # noqa: E731
            if ratio_lo is not None and ratio_hi is not None:
                ax.plot([clip(ratio_lo), clip(ratio_hi)], [y + dy, y + dy], color=theme["ink2"], linewidth=1, zorder=2)
            color = {"wins": theme["model"], "loses": theme["rule"]}.get(verdict)
            ax.scatter(
                [clip(ratio)], [y + dy], marker=_SCOREBOARD_MARKERS.get(dataset, "o"), s=80,
                facecolor=color if color else theme["surface"], edgecolor=color or theme["ink2"],
                linewidth=1.3, zorder=3,
            )
    ax.axvline(1.0, color=theme["ink2"], linewidth=1, zorder=1)
    ax.set_xlim(*_SCOREBOARD_XLIM)
    ticks = [0.8, 1.0, 1.2, 1.4, 1.6]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0.8x", "1.0x = the rule", "1.2x", "1.4x", "1.6x+"], fontsize=8.5, color=theme["ink2"])
    ax.grid(axis="x", color=theme["grid"], linewidth=0.8, zorder=0)
    ax.set_yticks(range(n))
    ax.set_yticklabels([label for label, _ in reversed(rows)])
    ax.set_ylim(-0.6, n - 0.4)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=theme["ink2"], length=0)

    legend_handles = [
        plt.Line2D([0], [0], marker=_SCOREBOARD_MARKERS.get(ds, "o"), linestyle="none", markerfacecolor=theme["ink2"],
                   markeredgecolor=theme["ink2"], markersize=8, label=ds)
        for ds in seen_datasets
    ] + [
        plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=theme["model"],
                   markeredgecolor=theme["model"], markersize=8, label="beats the rule"),
        plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=theme["surface"],
                   markeredgecolor=theme["ink2"], markersize=8, label="about the same"),
        plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=theme["rule"],
                   markeredgecolor=theme["rule"], markersize=8, label="loses"),
    ]

    title = f"{n_wins} of {n_total} real-file comparisons beat the simple rule your shop already uses"
    wrapped_title = _wrap_title(title)
    extra_in = _TITLE_LINE_IN * wrapped_title.count("\n")
    # Two legend rows, files then verdicts, so five files still fit the width.
    n_files = len(seen_datasets)
    for row_i, handles in enumerate((legend_handles[:n_files], legend_handles[n_files:])):
        fig.legend(
            handles=handles, loc="center",
            bbox_to_anchor=(0.5, 1 - (_LEGEND_Y_IN - 0.1 + 0.22 * row_i + extra_in) / fig_h),
            ncol=len(handles), frameon=False, labelcolor=theme["ink2"], fontsize=8,
            columnspacing=0.8, handletextpad=0.2,
        )
    fig.suptitle(
        wrapped_title, x=0.015, ha="left", y=1 - _TITLE_Y_IN / fig_h, fontsize=12, fontweight="bold",
        color=theme["ink1"], linespacing=1.3,
    )
    caption = " ".join(
        ["Dot: model result divided by the best simple rule's, top 10% of the list. "
         "Line: range across redraws and test years."] + notes
    )
    fig.text(0.015, 0.25 / fig_h, _wrap_subtitle(caption), fontsize=8, color=theme["ink2"], ha="left", va="bottom")
    fig.savefig(path)
    plt.close(fig)


def _row(rows, model, metric):
    for r in rows:
        if r.model == model and r.metric == metric:
            return r
    return None


MOMENTUM_METHOD_UPGRADE = (
    "momentum = build_leadership_snapshots(include_momentum=True): trailing 3y/5y OLS "
    "slope and relative slope per base series (fy_total, gift_count, largest_gift), "
    "plus fy_total_growth_ratio. The shipped, opt-in feature set score_leadership_prospects "
    "itself can use."
)
MOMENTUM_METHOD_PANEL = (
    "momentum = trailing slope of yearly giving, added to the benchmark's own feature "
    "panel (philanthropy.utils.trailing_slope_features applied to this script's own "
    "per-donor annual series, not the shipped RFMTransformer/activities_to_features path)."
)


def _momentum_summary(rows, model: str, method: str) -> Dict[str, Any]:
    """Top1/5/10pct model hit rate (in percentage points) plus verdict, for a
    bench run with ``include_momentum=True``. No baseline/random columns: this
    is a comparison against the model's own default-feature row (``_row``
    callers diff the two "model" numbers), not a new baseline. Carries each
    row's own ``lo``/``hi`` (a 5-seed min/max range for synthetic rows, a
    bootstrap interval for KDD98) and ``n_seeds``, plus a plain-language
    ``method`` string the docs can quote directly."""
    row_by_p = {p: _row(rows, model, f"top{p}pct_hit_rate") for p in (1, 5, 10)}
    out = {
        f"top{p}pct": {
            "model": row_by_p[p].value * 100,
            "lo": row_by_p[p].lo * 100 if row_by_p[p].lo is not None else None,
            "hi": row_by_p[p].hi * 100 if row_by_p[p].hi is not None else None,
            "n_seeds": row_by_p[p].n_seeds,
        }
        for p in (1, 5, 10)
    }
    out.update(_verdict_fields(row_by_p[10]))
    out["method"] = method
    return out


def _verdict_fields(row, scale: float = 100.0) -> Dict[str, Any]:
    """The row's verdict plus the paired model-minus-rule interval it was
    read from (``diff_lo``/``diff_hi``, in percentage points for rates, or
    in dollars with ``scale=1``), so every verdict in ``results.json`` can be
    checked against its own interval."""
    def _scaled(v):
        return None if v is None or v != v else v * scale

    return {"verdict": row.verdict, "diff_lo": _scaled(row.diff_lo), "diff_hi": _scaled(row.diff_hi)}


def _diff_pp(row) -> Dict[str, Any]:
    """``diff_lo``/``diff_hi`` of one top-N row, in percentage points."""
    fields = _verdict_fields(row)
    return {"diff_lo": fields["diff_lo"], "diff_hi": fields["diff_hi"]}


def _revenue_fields(rows, model: str = "AskAmountRecommender") -> Dict[str, Any]:
    """Share of next-gift dollars (out of 100) from the top 10% of donors
    ranked by the model's predicted amount and by the best rule's, with the
    paired interval when the run has one."""
    row = _row(rows, model, "revenue_top10pct")
    if row is None:
        return {}
    fields = _verdict_fields(row)
    return {
        "revenue_top10pct_model": row.value * 100, "revenue_top10pct_rule": row.baseline * 100,
        "revenue_top10pct_diff_lo": fields["diff_lo"], "revenue_top10pct_diff_hi": fields["diff_hi"],
        "revenue_top10pct_verdict": fields["verdict"],
    }


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
    # DonorsChoose's Donations file (load_donorschoose) has only donor_id,
    # gift date/amount, donor_type and payment-method flags: no wealth
    # screening, demographics, or mailing/solicitation history at all.
    "donorschoose": frozenset({"giving_history", "recency"}),
    # PSID (load_psid_philanthropy): household giving by cause, family
    # income, wealth with and without home equity, and volunteer hours; no
    # mailing/solicitation history.
    "psid": frozenset({"giving_history", "recency", "wealth", "engagement"}),
    # Karlan and List (load_karlan_list): giving history before the letter,
    # gender / couple flags and the 2004 presidential vote of the donor's
    # state and county; no wealth screen, no event or mailing history.
    "karlan_list": frozenset({"giving_history", "recency", "wealth"}),
}

# Who one row of each dataset is, for "the model looks at N things about each
# <noun>"; datasets not listed here are donor files.
DATASET_SUBJECT_NOUN = {"psid": "household"}

# Per-dataset replacements for GROUP_MEANINGS where the generic wording would
# be wrong for that file (PSID's wealth columns are survey answers, not a
# wealth screen).
DATASET_GROUP_MEANINGS = {
    "psid": {"wealth": "self-reported household income and wealth (survey answers, not a wealth screen)"},
    "karlan_list": {"wealth": "gender, couple and the 2004 vote where the donor lives (no wealth screen)"},
}


def _dataset_category(key: str) -> str:
    for suffix in ("_synthetic", "_cup98val", "_kdd98", "_donorschoose", "_psid", "_karlan_list"):
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
    # shared snapshot builder (philanthropy.ingest.build_snapshots): a period
    # is a fiscal year on a gift log and a survey wave on PSID
    "period_total": ("giving this year (this wave on PSID)", "recency"),
    "period_total_prior1": ("giving the year before (the wave before on PSID)", "recency"),
    "period_total_prior2": ("giving two years before (two waves before on PSID)", "recency"),
    "period_trend": ("giving trend, this year vs the one before", "momentum"),
    "consecutive_periods_given": ("consecutive years (or waves) given", "momentum"),
    "gave_prior1": ("gave the year (or wave) before", "recency"),
    "gave_prior2": ("gave two years (or waves) before", "recency"),
    "periods_since_first_gift": ("years (or waves) since first gift", "giving_history"),
    # PSID household x wave snapshots (bm._psid_wave_period_snapshots)
    "total_giving": ("this wave's giving", "recency"),
    "prior_wave_total": ("last wave's giving", "recency"),
    "trend": ("giving trend, this wave vs last", "momentum"),
    "waves_given_streak": ("consecutive waves given", "momentum"),
    "largest_giving_category": ("largest single cause this wave", "giving_history"),
    "itemized_charitable_contrib_amount": ("charitable deduction claimed on taxes", "giving_history"),
    "giving_checkpoint_other_2001": ("giving to other causes (2001 only)", "giving_history"),
    "giving_combo": ("giving to combined-purpose charities", "giving_history"),
    "giving_community": ("giving to community causes", "giving_history"),
    "giving_cultural": ("giving to arts and culture", "giving_history"),
    "giving_education": ("giving to education", "giving_history"),
    "giving_environment": ("giving to the environment", "giving_history"),
    "giving_health": ("giving to health causes", "giving_history"),
    "giving_international": ("giving to international aid", "giving_history"),
    "giving_needy": ("giving to help people in need", "giving_history"),
    "giving_other": ("giving to other causes", "giving_history"),
    "giving_religious": ("giving to religious causes", "giving_history"),
    "giving_youth": ("giving to youth causes", "giving_history"),
    "family_income": ("household income", "wealth"),
    "wealth1": ("household wealth, not counting home equity", "wealth"),
    "wealth2": ("household wealth, counting home equity", "wealth"),
    "head_volunteer_hours_annual": ("head's volunteer hours last year", "engagement"),
    "spouse_volunteer_hours_annual": ("spouse's volunteer hours last year", "engagement"),
    "household_volunteer_hours_regular": ("household's regular volunteer hours", "engagement"),
    "head_volunteer_hours_typical_week": ("head's volunteer hours in a typical week", "engagement"),
    "spouse_volunteer_hours_typical_week": ("spouse's volunteer hours in a typical week", "engagement"),
    # Karlan and List matching-grant experiment (bm.KARLAN_LIST_FEATURES;
    # months_since_last_gift is shared with the upgrade snapshots above)
    "prior_gifts": ("number of past gifts", "giving_history"),
    "highest_previous_amount": ("largest past gift", "giving_history"),
    "years_since_first_gift": ("years as a donor", "giving_history"),
    "female": ("donor is a woman", "wealth"),
    "couple": ("donor record is a couple", "wealth"),
    "red_state": ("state voted Republican in 2004", "wealth"),
    "red_county": ("county voted Republican in 2004", "wealth"),
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


def upgrade_drivers_kdd98(seed, threshold="p92", band=(5.0, 49.0)) -> Dict[str, Any]:
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
# DonorsChoose driver functions: same columns as the synthetic/KDD98 upgrade
# snapshots (_gift_log_period_snapshots reuses build_leadership_snapshots'
# internals), one real fitted model on the last walk-forward fold (no 5-seed
# range: one real file has no second draw, same convention as KDD98 above).
# --------------------------------------------------------------------------- #
def _drivers_last_fold(
    snap, period_col: str, n_folds: int, feature_cols: Sequence[str], seed: int, kind: str,
    make_model, scoring: str, split: str, target_label: str = "score",
) -> Dict[str, Any]:
    """Shared body for the real-file (DonorsChoose, PSID) driver functions:
    one fit on the last walk-forward fold of an already-built snapshot
    table, differing only in ``kind``, the estimator, and the scoring metric."""
    target_dtype = "float64" if kind == "ask" else None

    def build(_seed):
        t = bm._walk_forward_test_periods(snap, period_col, n_folds)[-1]
        train, test = snap[snap[period_col] < t], snap[snap[period_col] == t]
        return (
            train[list(feature_cols)].to_numpy("float64"), train["target"].to_numpy(target_dtype),
            test[list(feature_cols)].to_numpy("float64"), test["target"].to_numpy(target_dtype),
        )

    drivers = _top_drivers(make_model, [seed], build, feature_cols, scoring=scoring)
    return _features_entry(feature_cols, drivers, scoring=scoring, split=split, target_label=target_label)


def _drivers_donorschoose(path: str, seed: int, kind: str, make_model, scoring: str, target_label: str = "score") -> Dict[str, Any]:
    """Same columns as the synthetic/KDD98 upgrade snapshots
    (``_gift_log_period_snapshots`` reuses ``build_leadership_snapshots``'
    internals)."""
    gifts = bm._donorschoose_gift_log(path, bm.DONORSCHOOSE_SUBSAMPLE, bm.DONORSCHOOSE_SEED)
    if kind == "lapse":
        snap = bm._donorschoose_lapse_snapshots(gifts, False)
        period, cols = "period", bm._snapshot_feature_cols(snap, "donor_id")
    else:
        snap = bm._gift_log_period_snapshots(gifts, 7, kind, False, 1000.0, (100.0, 999.0))
        period, cols = "fiscal_year", UPGRADE_FEATURE_COLS[1:]
    return _drivers_last_fold(
        snap, period, bm.DONORSCHOOSE_N_FOLDS, cols, seed, kind, make_model, scoring,
        split=f"walk-forward (subsample={bm.DONORSCHOOSE_SUBSAMPLE}, seed={bm.DONORSCHOOSE_SEED}, last fiscal-year fold)",
        target_label=target_label,
    )


def upgrade_drivers_donorschoose(path: str, seed: int) -> Dict[str, Any]:
    return _drivers_donorschoose(path, seed, "upgrade", lambda s: bm.MajorGiftClassifier(random_state=s), scoring="roc_auc")


def lapse_drivers_donorschoose(path: str, seed: int) -> Dict[str, Any]:
    return _drivers_donorschoose(path, seed, "lapse", lambda s: bm.LapsePredictor(random_state=s), scoring="roc_auc")


def ask_drivers_donorschoose(path: str, seed: int) -> Dict[str, Any]:
    return _drivers_donorschoose(
        path, seed, "ask", lambda s: bm.AskAmountRecommender(random_state=s),
        scoring="neg_mean_absolute_error", target_label="suggested ask",
    )


def drivers_psid(data_path: str, do_path: str, seed: int, kind: str) -> Dict[str, Any]:
    """PSID counterpart of the DonorsChoose drivers above: the household x
    wave columns bench_*_psid actually feeds the model (giving by cause,
    income, wealth, volunteer hours), last walk-forward wave fold."""
    make_model, scoring, target_label = {
        "upgrade": (lambda s: bm.MajorGiftClassifier(random_state=s), "roc_auc", "score"),
        "lapse": (lambda s: bm.LapsePredictor(random_state=s), "roc_auc", "score"),
        "ask": (lambda s: bm.AskAmountRecommender(random_state=s), "neg_mean_absolute_error", "suggested ask"),
    }[kind]
    if kind == "lapse":
        snap, period = bm._psid_lapse_snapshots(data_path, do_path, False), "period"
    else:
        snap, period = bm._psid_wave_period_snapshots(data_path, do_path, kind, False, 1000.0, (100.0, 999.0)), "wave"
    return _drivers_last_fold(
        snap, period, bm.PSID_N_FOLDS, bm._snapshot_feature_cols(snap, "household_key"), seed, kind, make_model, scoring,
        split=f"walk-forward (seed={bm.PSID_SEED}, last wave fold)", target_label=target_label,
    )


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
        ("DonorsChoose (real donor file)", "upgrade_donorschoose"),
        ("PSID (household survey)", "upgrade_psid"),
    ],
    "response": [
        ("KDD Cup 1998 (real donor file)", "response_kdd98"),
        ("cup98VAL (real donor file, never seen by the model)", "response_cup98val"),
        ("Karlan and List (real donor file)", "response_karlan_list"),
        ("Sample data (checks the code runs, not that the model works)", "response_synthetic"),
    ],
    "lapse": [
        ("KDD Cup 1998 (real donor file)", "lapse_kdd98"),
        ("DonorsChoose (real donor file)", "lapse_donorschoose"),
        ("PSID (household survey)", "lapse_psid"),
    ],
    "ask": [
        ("KDD Cup 1998 (real donor file)", "ask_kdd98"),
        ("Sample data", "ask_synthetic"),
        ("DonorsChoose (real donor file)", "ask_donorschoose"),
        ("PSID (household survey)", "ask_psid"),
        ("Karlan and List (real donor file)", "ask_karlan_list"),
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
    meanings = {**GROUP_MEANINGS, **DATASET_GROUP_MEANINGS.get(dataset_category, {})}
    for g in present:
        lines.append(f"    | {GROUP_LABELS[g]} | {meanings[g]} |")
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
            noun = DATASET_SUBJECT_NOUN.get(dataset_category, "donor")
            lines.append(f"    On this file the model looks at {n} things about each {noun}.")
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


# --------------------------------------------------------------------------- #
# The index table and the scoreboard, both computed from results.json so no
# verdict on the site is hand-written (E.15.5.1). Sample data never appears
# in either: it is the "does the code run" tab, not a result.
# --------------------------------------------------------------------------- #
INDEX_QUESTIONS = (
    ("upgrade", "[Leadership upgrade ($1,000+)](leadership.md)", "Leadership upgrade",
     "Which mid-level donors are about to become $1,000+ donors?"),
    ("response", "[Response](response.md)", "Response", "Who is most likely to give again next year?"),
    ("lapse", "[Lapse](lapse.md)", "Lapse", "Which donors are about to stop giving?"),
    ("ask", "[Suggested ask](ask.md)", "Suggested ask", "What will this donor give next?"),
    ("planned_giving", "[Planned giving](planned_giving.md)", "Planned giving", "Which donors look like bequest prospects?"),
    ("who_to_mail", "[Who to mail](who_to_mail.md)", "Who to mail", "Is it worth mailing this donor at all?"),
)
INDEX_FILES = (
    ("KDD Cup 1998", "kdd98"), ("cup98VAL", "cup98val"), ("DonorsChoose", "donorschoose"), ("PSID", "psid"),
    ("Karlan and List", "karlan_list"),
)
# Pairs with no "<question>_<file>_note" key in results.json that still
# cannot be tested; anything else missing reads "not run".
INDEX_CANT_TEST = {("planned_giving", "kdd98"): "no bequest-intent label"}
# Which organisation's donors each file holds: KDD Cup 1998 and cup98VAL are
# two halves of one charity's 1997 mailing, so they are not independent.
INDEX_ORGS = {
    "kdd98": "PVA", "cup98val": "PVA", "donorschoose": "DonorsChoose", "psid": "PSID",
    "karlan_list": "Karlan and List",
}
# Cells shown but not counted in the bottom line, recorded before the run
# that produced them: KDD98 gifts are too small for the $1,000 question, so
# its upgrade cell answers a $50 proxy (E.15: "KDD98 is not an upgrade
# benchmark; PSID is").
INDEX_PROXY = {("upgrade", "kdd98"): "Proxy question ($50 threshold), not counted"}
HIGH_BASE_RATE_PCT = 80.0
VERDICT_WORDS = {"wins": "Beats the rule", "modest": "About the same as the rule", "loses": "Loses to the rule"}


def _lapse_base_rate(entry: Dict[str, Any]) -> float | None:
    return entry.get("base_rate_pct", entry.get("metadata", {}).get("base_rate_pct"))


def _n_folds(entry: Dict[str, Any]) -> int:
    meta = entry.get("metadata", {})
    return len(meta.get("fold_years") or meta.get("fold_waves") or []) or 1


def _of100(x: float) -> str:
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}"


def _ratio_block(model: float, rule: float, diff_lo: float | None, diff_hi: float | None) -> Dict[str, Any]:
    if not rule:
        return {"ratio": None, "ratio_lo": None, "ratio_hi": None}
    return {
        "ratio": model / rule,
        "ratio_lo": None if diff_lo is None else (rule + diff_lo) / rule,
        "ratio_hi": None if diff_hi is None else (rule + diff_hi) / rule,
    }


def index_cell(results: Dict[str, Any], question: str, ds: str) -> Dict[str, Any]:
    cell = _index_cell(results, question, ds)
    cell["org"] = INDEX_ORGS[ds]
    cell["counted"] = cell["verdict"] is not None and (question, ds) not in INDEX_PROXY
    if (question, ds) in INDEX_PROXY and cell["verdict"] is not None:
        short = {"wins": "wins", "modest": "about the same", "loses": "loses"}[cell["verdict"]]
        cell["text"] = f"{INDEX_PROXY[(question, ds)]}: {short} {cell['text'].split(': ', 1)[1]}"
    return cell


def _index_cell(results: Dict[str, Any], question: str, ds: str) -> Dict[str, Any]:
    """One index-table cell: its text, the verdict it carries (``None`` for
    "can't test" / "not run"), and the lift ratio the scoreboard plots. A
    lapse cell on a file where more than 80 in 100 donors lapse reads the
    retention list instead (E.15.3 item 4): "who will lapse" has no useful
    answer there, so the cell says so and reports the "who keeps giving" list
    and its lift over random."""
    key = f"{question}_{ds}"
    if key not in results:
        note = results.get(f"{key}_note")
        reason = INDEX_CANT_TEST.get((question, ds))
        if note:
            reason = note.rstrip(".").split(", so ")[0].split(" in this ")[0]
            reason = reason[0].lower() + reason[1:]
        return {"text": f"can't test ({reason})" if reason else "not run", "verdict": None}
    entry = results[key]
    if question == "lapse":
        base = _lapse_base_rate(entry)
        ret = results.get(f"lapse_{ds}_retention")
        if base is not None and base > HIGH_BASE_RATE_PCT and ret is not None:
            t = ret["top10pct"]
            lift = t["model"] / (100.0 - base)
            return {
                "text": (
                    f"Nearly everyone lapses here ({base:.0f} of 100). Who keeps giving: "
                    f"{VERDICT_WORDS[ret['verdict']].lower()}, {_of100(t['model'])} vs {_of100(t['rule'])} of 100 "
                    f"({lift:.1f}x random)"
                ),
                "verdict": ret["verdict"], "retention": True, "folds": _n_folds(ret),
                "top_slice_win": False,
                **_ratio_block(t["model"], t["rule"], t.get("diff_lo"), t.get("diff_hi")),
            }
    if question == "ask":
        m, r = entry["within25pct_model"], entry["within25pct_last_gift"]
        text = f"{VERDICT_WORDS[entry['verdict']]}: {_of100(m)} vs {_of100(r)} of 100 within 25%"
        return {"text": text, "verdict": entry["verdict"], "folds": _n_folds(entry), "top_slice_win": False,
                **_ratio_block(m, r, entry.get("diff_lo"), entry.get("diff_hi"))}
    if question == "who_to_mail":
        m, r = entry["net_revenue_model"], entry["net_revenue_mail_everyone"]
        word = {"wins": "Beats mailing everyone", "loses": "Loses to mailing everyone"}.get(
            entry["verdict"], "About the same as mailing everyone"
        )
        return {"text": f"{word}: ${m:,.0f} vs ${r:,.0f} net", "verdict": entry["verdict"], "folds": _n_folds(entry),
                "top_slice_win": False, **_ratio_block(m, r, entry.get("diff_lo"), entry.get("diff_hi"))}
    t = entry["top10pct"]
    top_slice_win = any((entry[f"top{p}pct"].get("diff_lo") or 0) > 0 for p in (1, 5))
    text = f"{VERDICT_WORDS[entry['verdict']]}: {_of100(t['model'])} vs {_of100(t['rule'])} of 100"
    if entry["verdict"] != "wins" and top_slice_win:
        text += ", ahead in the top 1% or 5%"
    return {"text": text, "verdict": entry["verdict"], "folds": _n_folds(entry), "top_slice_win": top_slice_win,
            **_ratio_block(t["model"], t["rule"], t.get("diff_lo"), t.get("diff_hi"))}


def bottom_line(cells: List[Dict[str, Any]]) -> str:
    """One of the six fixed bottom lines from a question's counted cells
    (real files, minus proxy questions; a "nearly everyone lapses" cell
    counts through its retention read), by a rule fixed before the run
    (E.11f rule 5):

    - Use the model: wins from two independent sources (different
      organisations, or one source across several test years) and no loss.
    - Use the model (tested on one organisation so far): no loss, and every
      win comes from one organisation, in single splits.
    - Use the model for your top slice only: no loss, no win, and an "about
      the same" that is ahead at the top 1% or 5%.
    - Use the retention list: as "Use the model", when every counted cell is
      a retention read.
    - Can't tell yet: a win and a loss, or nothing counted.
    - Use the simple rule: everything else."""
    counted = [c for c in cells if c.get("counted", c["verdict"] is not None)]
    wins = [c for c in counted if c["verdict"] == "wins"]
    losses = [c for c in counted if c["verdict"] == "loses"]
    if not counted or (wins and losses):
        return "Can't tell yet"
    if losses:
        return "Use the simple rule"
    if len({c["org"] for c in wins}) >= 2 or any(c["folds"] >= 2 for c in wins):
        if all(c.get("retention") for c in counted):
            return "Use the retention list"
        return "Use the model"
    if wins:
        return "Use the model (tested on one organisation so far)"
    if any(c["top_slice_win"] for c in counted):
        return "Use the model for your top slice only"
    return "Use the simple rule"


def build_index(results: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for question, _link, _label, _ask in INDEX_QUESTIONS:
        cells = {ds: index_cell(results, question, ds) for _name, ds in INDEX_FILES}
        out[question] = {"cells": cells, "bottom_line": bottom_line(list(cells.values()))}
    return out


def render_index_table(index: Dict[str, Any], path: Path) -> None:
    lines = [
        "| Model | Question it answers | " + " | ".join(name for name, _ in INDEX_FILES) + " | Bottom line |",
        "|---|---|" + "---|" * len(INDEX_FILES) + "---|",
    ]
    for question, link, _label, ask in INDEX_QUESTIONS:
        row = index[question]
        cells = " | ".join(row["cells"][ds]["text"] for _name, ds in INDEX_FILES)
        lines.append(f"| {link} | {ask} | {cells} | **{row['bottom_line']}** |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def render_cost_sweep_table(sweep: List[Dict[str, Any]], path: Path) -> None:
    """Who to mail at each cost per letter, as a table snippet for one tab.
    The range is the paired bootstrap interval on the difference."""
    lines = [
        "| Cost per letter | Letters sent | Raised after costs | Mailing everyone | Difference (range) |",
        "|---|---|---|---|---|",
    ]

    def money(v: float, sign: str = "") -> str:
        return f"{'-' if v < 0 else ('+' if sign and v > 0 else '')}${abs(v):,.0f}"

    for r in sweep:
        diff = r["net_revenue_model"] - r["net_revenue_mail_everyone"]
        lines.append(
            f"| ${r['cost']:.2f} | {r['mailed']:,} of {r['n_total']:,} | {money(r['net_revenue_model'])} | "
            f"{money(r['net_revenue_mail_everyone'])} | {money(diff, '+')} "
            f"({money(r['diff_lo'], '+')} to {money(r['diff_hi'], '+')}) |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def scoreboard_rows(index: Dict[str, Any]) -> Tuple[List[tuple], List[str]]:
    rows, untested, notes = [], [], []
    for question, _link, label, _ask in INDEX_QUESTIONS:
        cells = [(name, index[question]["cells"][ds]) for name, ds in INDEX_FILES]
        dots = [(name, c["verdict"], c["ratio"], c["ratio_lo"], c["ratio_hi"])
                for name, c in cells if c["counted"] and c.get("ratio") is not None]
        if not dots:
            untested.append(label.lower())
            continue
        rows.append((label, dots))
        notes += [f"{label} on {name}: {INDEX_PROXY[(question, ds)].lower()}, so not drawn."
                  for name, ds in INDEX_FILES if (question, ds) in INDEX_PROXY]
        retention = [name for name, c in cells if c.get("retention")]
        if retention:
            notes.append(f"{label} on {' and '.join(retention)}: nearly everyone lapses, so the dot is the "
                         "who-keeps-giving list.")
    if untested:
        notes.append(f"Not yet testable on any real file: {', '.join(untested)}.")
    return rows, notes


def karlan_list_drivers(path: str, kind: str) -> Dict[str, Any]:
    """Drivers for the Karlan and List response ("response") or amount
    ("ask") model, on the same split and features as the benchmark rows."""
    feature_cols = tuple(bm.KARLAN_LIST_FEATURES)

    def build(_seed):
        df, idx_train, _idx_val, idx_test = bm._karlan_list_split(path)
        if kind == "ask":
            gave = df["gave"].to_numpy() == 1
            tr, te = idx_train[gave[idx_train]], idx_test[gave[idx_test]]
            X, y = df[list(feature_cols)], df["amount"].to_numpy()
        else:
            tr, te = idx_train, idx_test
            X, y = bm._karlan_list_design(df, idx_train), df["gave"].to_numpy()
        return X.iloc[tr].to_numpy(), y[tr], X.iloc[te].to_numpy(), y[te]

    seed = bm.KARLAN_LIST_SEED
    if kind == "ask":
        drivers = _top_drivers(
            lambda s: bm.AskAmountRecommender(random_state=s), [seed], build, feature_cols,
            scoring="neg_mean_absolute_error",
        )
        return _features_entry(
            feature_cols, drivers, scoring="neg_mean_absolute_error", split=bm.KDD_SPLIT, target_label="suggested ask",
        )
    drivers = _top_drivers(lambda s: bm.MajorGiftClassifier(random_state=s), [seed], build, feature_cols, scoring="roc_auc")
    return _features_entry(feature_cols, drivers, scoring="roc_auc", split=bm.KDD_SPLIT)


def _karlan_list_section(results: Dict[str, Any], path: str) -> None:
    """Response, amount and matching-grant uplift on the Karlan and List
    experiment (aggregates only; data CC BY 4.0, copyright AEA 2007)."""
    df = bm.load_karlan_list(path)
    meta = {"n_donors": int(len(df)), "base_rate_pct": float(df["gave"].mean()) * 100, "split": bm.KDD_SPLIT}

    rows = bm.bench_response_karlan_list(path)
    by_p = {p: _row(rows, "MajorGiftClassifier", f"top{p}pct_hit_rate") for p in (1, 5, 10)}
    entry = {
        f"top{p}pct": {"model": by_p[p].value * 100, "rule": by_p[p].baseline * 100, **_diff_pp(by_p[p])}
        for p in (1, 5, 10)
    }
    entry.update(_verdict_fields(by_p[10]))
    entry["roc_auc"] = _row(rows, "MajorGiftClassifier", "roc_auc").value
    entry["metadata"] = meta
    entry["features"] = karlan_list_drivers(path, "response")
    results["response_karlan_list"] = entry
    r10 = entry["top10pct"]
    _render_themed(
        _hbar_chart, OUT_DIR / "response_karlan_list.png", ["Top 1%", "Top 5%", "Top 10%"],
        {
            "Model": [entry[f"top{p}pct"]["model"] for p in (1, 5, 10)],
            "Best simple rule": [entry[f"top{p}pct"]["rule"] for p in (1, 5, 10)],
        },
        {"Model": "model", "Best simple rule": "rule"},
        title=_takeaway(r10["model"], r10["rule"], "The model", "the best simple rule"),
        subtitle="Karlan and List, one 2005 fundraising letter, held-out 30% of donors. Gave, out of every 100 picked.",
        errors={"Model": [_ci(by_p[p]) for p in (1, 5, 10)], "Best simple rule": [None, None, None]},
        base_rate=meta["base_rate_pct"], base_rate_label=f"everyone: {meta['base_rate_pct']:.1f} of 100",
    )

    rows = bm.bench_ask_karlan_list(path)
    within, mae = _row(rows, "AskAmountRecommender", "within25pct"), _row(rows, "AskAmountRecommender", "mae")
    results["ask_karlan_list"] = {
        "within25pct_model": within.value * 100, "within25pct_last_gift": within.baseline * 100,
        "mae_model": mae.value, "mae_rule": mae.baseline,
        "mae_diff_lo": mae.diff_lo, "mae_diff_hi": mae.diff_hi,
        **_verdict_fields(within),
        **_revenue_fields(rows),
        "metadata": {
            **meta, "target": "amount given, among donors who gave", "rule": "highest previous gift",
            "n_test": int(within.note.rsplit("n=", 1)[1]),
        },
        "features": karlan_list_drivers(path, "ask"),
    }
    a = results["ask_karlan_list"]
    _render_themed(
        _hbar_chart, OUT_DIR / "ask_karlan_list.png", ["Suggested ask"],
        {"Model": [a["within25pct_model"]], "Best simple rule": [a["within25pct_last_gift"]]},
        {"Model": "model", "Best simple rule": "rule"},
        title=_takeaway(a["within25pct_model"], a["within25pct_last_gift"], "The model", "the best simple rule"),
        subtitle="Karlan and List, donors who gave to the 2005 letter. Predicted amounts landing within 25% of the actual gift.",
    )

    rows = bm.bench_uplift_karlan_list(path)
    up = {}
    for p in (10, 30):
        r = _row(rows, "UpliftTLearner", f"uplift_top{p}pct")
        up[f"top{p}pct"] = {"model": r.value * 100, "rule": r.baseline * 100, **_diff_pp(r), "verdict": r.verdict}
    note = _row(rows, "UpliftTLearner", "uplift_top30pct").note
    up["everyone"] = float(note.rsplit("everyone=", 1)[1]) * 100
    up["rule"] = note.split("rule_set=", 1)[1].split(" (", 1)[0]
    up["metadata"] = meta
    results["uplift_karlan_list"] = up


INTERVAL_FILES = (
    ("kdd98", "KDD Cup 1998, held-out 30% of donors who gave"),
    ("donorschoose", "DonorsChoose, last 4 fiscal years, one at a time"),
    ("psid", "PSID household survey, last 4 waves, one at a time"),
    ("karlan_list", "Karlan and List, held-out 30% of donors who gave"),
    ("synthetic", "Sample data, 5 seeds"),
)


def _interval_entry(rows) -> Dict[str, Any]:
    """Requested vs attained range coverage (out of 100) and median width
    (dollars) per level, with the fold or seed range where there is one."""
    levels = {}
    for level in bm.INTERVAL_LEVELS:
        cov = _row(rows, "GiftIntervalCalibrator", f"empirical_coverage(target={level:.2f})")
        width = _row(rows, "GiftIntervalCalibrator", f"median_width(target={level:.2f})")
        levels[f"{level * 100:.0f}"] = {
            "requested": level * 100, "attained": cov.value * 100,
            "attained_lo": cov.lo * 100 if cov.lo is not None else None,
            "attained_hi": cov.hi * 100 if cov.hi is not None else None,
            "median_width": width.value if width else None,
            "verdict": cov.verdict,
        }
    return {"levels": levels, "metadata": {"note": rows[0].note}}


def _interval_section(results: Dict[str, Any], args) -> None:
    """GiftIntervalCalibrator around each file's ask model: asked-for range
    coverage against what the ranges actually held, one chart per file."""
    runs = {"synthetic": lambda: [
        r for level in bm.INTERVAL_LEVELS
        for r in bm.bench_gift_interval(SEEDS, N_DONORS, N_YEARS, alpha=round(1 - level, 2))
    ]}
    if args.with_kdd98:
        runs["kdd98"] = bm.bench_gift_interval_kdd98
    if args.donorschoose_path:
        runs["donorschoose"] = lambda: bm.bench_gift_interval_donorschoose(args.donorschoose_path)
    if args.psid_data and args.psid_do:
        runs["psid"] = lambda: bm.bench_gift_interval_psid(args.psid_data, args.psid_do)
    if args.karlan_list_path:
        runs["karlan_list"] = lambda: bm.bench_gift_interval_karlan_list(args.karlan_list_path)
    for ds, subtitle in INTERVAL_FILES:
        if ds not in runs:
            continue
        entry = _interval_entry(runs[ds]())
        if ds == "synthetic":
            for v in entry["levels"].values():
                v.pop("verdict")
        results[f"interval_{ds}"] = entry
        lv = entry["levels"]
        _render_themed(
            _hbar_chart, OUT_DIR / f"interval_{ds}.png", [f"{k}% range" for k in lv],
            {"Asked for": [v["requested"] for v in lv.values()], "Held the actual gift": [v["attained"] for v in lv.values()]},
            {"Asked for": "random", "Held the actual gift": "model"},
            title=f"Asked for 90%, the range held the actual gift {lv['90']['attained']:.0f} times in 100",
            subtitle=f"{subtitle}. Out of every 100 gifts.",
            errors={
                "Asked for": [None] * len(lv),
                "Held the actual gift": [
                    (v["attained_lo"], v["attained_hi"]) if v["attained_lo"] != v["attained_hi"] else None
                    for v in lv.values()
                ],
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-kdd98", action="store_true", help="Also run the KDD Cup 1998 section (downloads ~36MB).")
    parser.add_argument(
        "--with-cup98val", action="store_true",
        help="Also score who-to-mail on KDD98's own held-out cup98VAL+valtargt file "
        "(a second ~37MB download; opt-in). No effect without --with-kdd98.",
    )
    parser.add_argument(
        "--with-momentum", action="store_true",
        help="Also run each synthetic model (and KDD98 upgrade, with --with-kdd98) with "
        "include_momentum=True, writing a '<key>_momentum' results.json entry next to the "
        "default-feature one. Does not change any existing key or chart.",
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
    parser.add_argument(
        "--karlan-list-path", type=str, default=None,
        help="Path to a user-obtained AERtables1-5.dta (openICPSR 113224, Karlan and List 2007); "
        "skipped entirely when not given.",
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
    results["response_synthetic"].update(_verdict_fields(resp_row_by_p[10]))
    if args.with_momentum:
        resp_mom_rows = bm.bench_response(SEEDS, N_DONORS, N_YEARS, include_momentum=True)
        results["response_synthetic_momentum"] = _momentum_summary(resp_mom_rows, "MajorGiftClassifier", MOMENTUM_METHOD_PANEL)
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
    results["lapse_synthetic"].update(_verdict_fields(_row(lapse_rows, "LapsePredictor", "top10pct_hit_rate")))
    if args.with_momentum:
        lapse_mom_rows = bm.bench_lapse(SEEDS, N_DONORS, N_YEARS, include_momentum=True)
        results["lapse_synthetic_momentum"] = _momentum_summary(lapse_mom_rows, "LapsePredictor", MOMENTUM_METHOD_PANEL)
    results["lapse_synthetic"]["features"] = lapse_drivers_synthetic(SEEDS, N_DONORS, N_YEARS)

    # --- synthetic: ask -------------------------------------------------
    ask_rows = bm.bench_ask(SEEDS, N_DONORS, N_YEARS)
    within = _row(ask_rows, "AskAmountRecommender", "within25pct")
    results["ask_synthetic"] = {
        "within25pct_model": within.value * 100,
        "within25pct_last_gift": within.baseline * 100,
        **_verdict_fields(within),
        **_revenue_fields(ask_rows),
        "features": ask_drivers_synthetic(SEEDS, N_DONORS, N_YEARS),
    }
    if args.with_momentum:
        ask_mom_rows = bm.bench_ask(SEEDS, N_DONORS, N_YEARS, include_momentum=True)
        within_mom = _row(ask_mom_rows, "AskAmountRecommender", "within25pct")
        results["ask_synthetic_momentum"] = {
            "within25pct_model": within_mom.value * 100,
            **_verdict_fields(within_mom),
            "method": MOMENTUM_METHOD_PANEL,
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
    results["upgrade_synthetic"].update(_verdict_fields(upg_row_by_p[10]))
    if args.with_momentum:
        upg_mom_rows = bm.bench_upgrade(SEEDS, N_DONORS, N_YEARS, include_momentum=True)
        results["upgrade_synthetic_momentum"] = _momentum_summary(upg_mom_rows, "upgrade_model (MajorGiftClassifier)", MOMENTUM_METHOD_UPGRADE)
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
            f"top{p}pct": {"model": resp_kdd_row_by_p[p].value * 100, "rule": resp_kdd_row_by_p[p].baseline * 100, **_diff_pp(resp_kdd_row_by_p[p])}
            for p in (1, 5, 10)
        }
        results["response_kdd98"].update(_verdict_fields(resp_kdd_row_by_p[10]))
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
            f"top{p}pct": {"model": upg_kdd_row_by_p[p].value * 100, "rule": upg_kdd_row_by_p[p].baseline * 100, **_diff_pp(upg_kdd_row_by_p[p])}
            for p in (1, 5, 10)
        }
        results["upgrade_kdd98"].update(_verdict_fields(upg_kdd_row_by_p[10]))
        if args.with_momentum:
            kdd_upg_mom_rows = bm.bench_kdd_upgrade(seed, include_momentum=True)
            results["upgrade_kdd98_momentum"] = _momentum_summary(
                kdd_upg_mom_rows, "upgrade_model (MajorGiftClassifier)", MOMENTUM_METHOD_UPGRADE
            )
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
            **_verdict_fields(lapse_top[10]),
            **{f"top{p}pct": {"model": r.value * 100, "rule": r.baseline * 100, **_diff_pp(r)} for p, r in lapse_top.items()},
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
            f"top{p}pct": {"model": ret_row_by_p[p].value * 100, "rule": ret_row_by_p[p].baseline * 100, **_diff_pp(ret_row_by_p[p])}
            for p in (1, 5, 10)
        }
        results["lapse_kdd98_retention"].update(_verdict_fields(ret_row_by_p[10]))
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
            **_verdict_fields(ask_row),
            **_revenue_fields(kdd_ask),
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
            **_verdict_fields(net_row, scale=1.0),
            "note": net_row.note,
            "features": who_to_mail_drivers_kdd98(seed),
        }
        curve = bm.kdd_mail_profit_curve(seed)
        results["who_to_mail_kdd98"]["curve"] = curve
        results["who_to_mail_kdd98"]["cost_sweep"] = bm.kdd_mail_cost_sweep(seed, held_out_file=False)
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
                **_verdict_fields(net_row_val, scale=1.0),
                "note": net_row_val.note,
                "features": _reused_features(
                    results["who_to_mail_kdd98"]["features"],
                    "Same response model and features as the KDD Cup 1998 tab; cup98VAL supplies new test "
                    "donors on the same columns, not new columns.",
                ),
            }
            curve_val = bm.kdd_mail_profit_curve_val(seed)
            results["who_to_mail_cup98val"]["curve"] = curve_val
            results["who_to_mail_cup98val"]["cost_sweep"] = bm.kdd_mail_cost_sweep(seed, held_out_file=True)
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
                f"top{p}pct": {"model": val_row_by_p[p].value * 100, "rule": val_row_by_p[p].baseline * 100, **_diff_pp(val_row_by_p[p])}
                for p in (1, 5, 10)
            }
            results["response_cup98val"].update(_verdict_fields(val_row_by_p[10]))
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

    # --- DonorsChoose / PSID (opt-in, local files; aggregates only) ----------
    def _classifier_entry(rows: List, model: str, momentum: bool, extra_meta: Dict[str, Any]):
        row_by_p = {p: _row(rows, model, f"top{p}pct_hit_rate") for p in (1, 5, 10)}
        if row_by_p[10] is None:
            return None
        entry = {
            f"top{p}pct": {"model": row_by_p[p].value * 100, "rule": row_by_p[p].baseline * 100, **_diff_pp(row_by_p[p])}
            for p in (1, 5, 10)
        }
        entry.update(_verdict_fields(row_by_p[10]))
        entry["roc_auc"] = _row(rows, model, "roc_auc").value
        entry["metadata"] = {"momentum": momentum, **extra_meta}
        return entry

    def _ask_entry(rows: List, model: str, momentum: bool, extra_meta: Dict[str, Any]):
        within_row = _row(rows, model, "within25pct")
        if within_row is None:
            return None
        mae_row = _row(rows, model, "mae")
        return {
            "within25pct_model": within_row.value * 100, "within25pct_last_gift": within_row.baseline * 100,
            "mae_model": mae_row.value, "mae_rule": mae_row.baseline,
            **_verdict_fields(within_row),
            **_revenue_fields(rows, model),
            "metadata": {"momentum": momentum, **extra_meta},
        }

    def _donorschoose_fold_meta(kind: str, threshold: float = 1000.0, band: tuple = (100.0, 999.0)) -> Dict[str, Any]:
        gifts = bm._donorschoose_gift_log(args.donorschoose_path, bm.DONORSCHOOSE_SUBSAMPLE, bm.DONORSCHOOSE_SEED)
        if kind == "lapse":
            snap, period = bm._donorschoose_lapse_snapshots(gifts, False), "period"
        else:
            snap, period = bm._gift_log_period_snapshots(gifts, 7, kind, False, threshold, band), "fiscal_year"
        if snap.empty:
            return {"subsample": bm.DONORSCHOOSE_SUBSAMPLE, "seed": bm.DONORSCHOOSE_SEED, "fold_years": [], "n_per_fold": [], "base_rate_pct": None}
        test_years = bm._walk_forward_test_periods(snap, period, bm.DONORSCHOOSE_N_FOLDS)
        test = snap[snap[period].isin(test_years)]
        return {
            "subsample": bm.DONORSCHOOSE_SUBSAMPLE, "seed": bm.DONORSCHOOSE_SEED,
            "fold_years": test_years, "n_per_fold": [int((test[period] == t).sum()) for t in test_years],
            "base_rate_pct": float(test["target"].mean()) * 100 if kind != "ask" else None,
        }

    def _psid_fold_meta(kind: str, threshold: float = 1000.0, band: tuple = (100.0, 999.0)) -> Dict[str, Any]:
        if kind == "lapse":
            snap, period = bm._psid_lapse_snapshots(args.psid_data, args.psid_do, False), "period"
        else:
            snap, period = bm._psid_wave_period_snapshots(args.psid_data, args.psid_do, kind, False, threshold, band), "wave"
        if snap.empty:
            return {"seed": bm.PSID_SEED, "fold_waves": [], "n_per_fold": [], "base_rate_pct": None}
        test_waves = bm._walk_forward_test_periods(snap, period, bm.PSID_N_FOLDS)
        test = snap[snap[period].isin(test_waves)]
        return {
            "seed": bm.PSID_SEED, "fold_waves": test_waves,
            "n_per_fold": [int((test[period] == w).sum()) for w in test_waves],
            "base_rate_pct": float(test["target"].mean()) * 100 if kind != "ask" else None,
        }

    def _real_file_charts(
        ds: str, upgrade_subtitle: str, lapse_subtitle: str, lapse_base_rate: float,
        retention_subtitle: str, ask_subtitle: str, lapse_title: str | None = None,
    ) -> None:
        """Upgrade/lapse/retention/ask charts for one opt-in real file
        (``ds`` is the results-key infix, e.g. "donorschoose"). A
        ``lapse_title`` means lapse is the norm on that file, drawn as a
        near-tie dot chart under that fixed title; otherwise lapse gets the
        same bar chart and computed title as the other reads."""
        def series(key):
            return {
                "Model": [results[key][f"top{p}pct"]["model"] for p in (1, 5, 10)],
                "Best simple rule": [results[key][f"top{p}pct"]["rule"] for p in (1, 5, 10)],
            }

        colors = {"Model": "model", "Best simple rule": "rule"}
        groups = ["Top 1%", "Top 5%", "Top 10%"]
        key = f"upgrade_{ds}"
        if key in results:
            u10 = results[key]["top10pct"]
            _render_themed(
                _hbar_chart, OUT_DIR / f"{key}.png", groups, series(key), colors,
                title=_takeaway(u10["model"], u10["rule"], "The model", "the best simple rule"),
                subtitle=upgrade_subtitle,
            )

        key = f"lapse_{ds}"
        if key in results:
            if lapse_title:
                _render_themed(
                    _neartie_dot_chart, OUT_DIR / f"{key}.png", groups, series(key), colors,
                    base_rate=lapse_base_rate, title=lapse_title, subtitle=lapse_subtitle,
                )
            else:
                l10 = results[key]["top10pct"]
                _render_themed(
                    _hbar_chart, OUT_DIR / f"{key}.png", groups, series(key), colors,
                    title=_takeaway(l10["model"], l10["rule"], "The model", "the best simple rule"),
                    subtitle=lapse_subtitle, base_rate=lapse_base_rate,
                    base_rate_label=f"everyone: {lapse_base_rate:.0f} of 100",
                )

        key = f"lapse_{ds}_retention"
        if key in results:
            r10 = results[key]["top10pct"]
            retention_base_rate = 100.0 - lapse_base_rate
            if abs(r10["model"] - r10["rule"]) < 1.5:
                rt_title = "The model and the rule find about the same retained group here"
            elif r10["model"] > r10["rule"]:
                rt_title = f"The model's least-likely-to-lapse 10% retains better: {r10['model']:.0f} of 100 vs {r10['rule']:.0f} of 100"
            else:
                rt_title = f"The rule still finds a better group: {r10['rule']:.0f} of 100 vs {r10['model']:.0f} of 100"
            _render_themed(
                _hbar_chart, OUT_DIR / f"{key}.png", groups, series(key), colors,
                title=rt_title, subtitle=retention_subtitle, base_rate=retention_base_rate,
                base_rate_label=f"everyone: {retention_base_rate:.0f} of 100",
            )

        key = f"ask_{ds}"
        if key in results:
            a = results[key]
            _render_themed(
                _hbar_chart, OUT_DIR / f"{key}.png", ["Suggested ask"],
                {"Model": [a["within25pct_model"]], "Best simple rule": [a["within25pct_last_gift"]]}, colors,
                title=_takeaway(a["within25pct_model"], a["within25pct_last_gift"], "The model", "the best simple rule"),
                subtitle=ask_subtitle,
            )

    if args.donorschoose_path:
        upgrade_meta = _donorschoose_fold_meta("upgrade")
        lapse_meta = _donorschoose_fold_meta("lapse")
        ask_meta = _donorschoose_fold_meta("ask")
        for momentum in (False, True):
            suffix = "_momentum" if momentum else ""
            up = bm.bench_upgrade_donorschoose(args.donorschoose_path, include_momentum=momentum)
            entry = _classifier_entry(up, "upgrade_model (MajorGiftClassifier)", momentum, upgrade_meta)
            if entry:
                if not momentum:
                    entry["features"] = upgrade_drivers_donorschoose(args.donorschoose_path, bm.DONORSCHOOSE_SEED)
                results[f"upgrade_donorschoose{suffix}"] = entry

            lap = bm.bench_lapse_donorschoose(args.donorschoose_path, include_momentum=momentum)
            entry = _classifier_entry(lap, "LapsePredictor", momentum, lapse_meta)
            if entry:
                if not momentum:
                    entry["features"] = lapse_drivers_donorschoose(args.donorschoose_path, bm.DONORSCHOOSE_SEED)
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
                if not momentum:
                    entry["features"] = ask_drivers_donorschoose(args.donorschoose_path, bm.DONORSCHOOSE_SEED)
                results[f"ask_donorschoose{suffix}"] = entry

        _real_file_charts(
            "donorschoose",
            upgrade_subtitle="DonorsChoose, a 10% random sample of citizen donors, test fiscal years "
            f"{upgrade_meta['fold_years'][0]}-{upgrade_meta['fold_years'][-1]}.",
            lapse_subtitle="DonorsChoose, a 10% random sample of citizen donors. Lapsed next fiscal year, out of every 100 picked.",
            lapse_base_rate=lapse_meta["base_rate_pct"],
            lapse_title="Most donors here give once, so lapsing is the norm, not a signal",
            retention_subtitle="DonorsChoose, the 10% least likely to lapse by model score. Gave again, out of every 100 in that group.",
            ask_subtitle="DonorsChoose, next fiscal-year total given they give again. Suggested amounts landing within 25% of what the donor actually gave.",
        )

    if args.psid_data and args.psid_do:
        upgrade_meta = _psid_fold_meta("upgrade")
        lapse_meta = _psid_fold_meta("lapse")
        ask_meta = _psid_fold_meta("ask")
        for momentum in (False, True):
            suffix = "_momentum" if momentum else ""
            up = bm.bench_upgrade_psid(args.psid_data, args.psid_do, include_momentum=momentum)
            entry = _classifier_entry(up, "upgrade_model (MajorGiftClassifier)", momentum, upgrade_meta)
            if entry:
                if not momentum:
                    entry["features"] = drivers_psid(args.psid_data, args.psid_do, bm.PSID_SEED, "upgrade")
                results[f"upgrade_psid{suffix}"] = entry

            lap = bm.bench_lapse_psid(args.psid_data, args.psid_do, include_momentum=momentum)
            entry = _classifier_entry(lap, "LapsePredictor", momentum, lapse_meta)
            if entry:
                if not momentum:
                    entry["features"] = drivers_psid(args.psid_data, args.psid_do, bm.PSID_SEED, "lapse")
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
                if not momentum:
                    entry["features"] = drivers_psid(args.psid_data, args.psid_do, bm.PSID_SEED, "ask")
                results[f"ask_psid{suffix}"] = entry

        waves = f"{upgrade_meta['fold_waves'][0]}-{upgrade_meta['fold_waves'][-1]}" if upgrade_meta["fold_waves"] else ""
        _real_file_charts(
            "psid",
            upgrade_subtitle=f"PSID, household heads giving $100-$999 to charity in a survey wave, test waves {waves}.",
            lapse_subtitle="PSID, household heads who gave in a survey wave. Gave nothing by the next wave, out of every 100 picked.",
            lapse_base_rate=lapse_meta["base_rate_pct"],
            retention_subtitle="PSID, the 10% least likely to lapse by model score. Gave again, out of every 100 in that group.",
            ask_subtitle="PSID, next-wave total given the household gives again. Suggested amounts landing within 25% of what the household actually gave.",
        )

    if args.donorschoose_path and (args.psid_data or args.with_kdd98):
        transfer = bm.bench_upgrade_transfer(
            args.donorschoose_path, args.psid_data, args.psid_do, with_kdd98=args.with_kdd98,
        )
        for target, rows in transfer.items():
            entry = _classifier_entry(rows, "upgrade_model transfer (fit on DonorsChoose)", False, {
                "fit_on": "donorschoose", "features": list(bm.TRANSFER_COLS),
            })
            if entry:
                results[f"upgrade_transfer_{target}"] = entry

    # Models this benchmark cannot honestly answer on either real dataset:
    # neither file has mailing-cost or planned-giving/bequest data.
    results["response_donorschoose_note"] = "No mailing/appeal log in this file, so a response model has nothing to predict response to."
    results["response_psid_note"] = "No mailing/appeal log in this extract, so a response model has nothing to predict response to."
    results["who_to_mail_donorschoose_note"] = "No per-contact mailing cost in this file, so cost-aware selection has no cost side to weigh."
    results["who_to_mail_psid_note"] = "No per-contact mailing cost in this extract, so cost-aware selection has no cost side to weigh."
    results["planned_giving_donorschoose_note"] = "No bequest/estate-intent signal in this file."
    results["planned_giving_psid_note"] = "No bequest/estate-intent signal in this extract."
    one_letter = "One letter and no later giving in this file, so there is no next year to predict."
    results["upgrade_karlan_list_note"] = one_letter
    results["lapse_karlan_list_note"] = one_letter
    results["who_to_mail_karlan_list_note"] = "No per-contact mailing cost in this file, so cost-aware selection has no cost side to weigh."
    results["planned_giving_karlan_list_note"] = "No bequest/estate-intent signal in this file."

    if args.karlan_list_path:
        _karlan_list_section(results, args.karlan_list_path)

    _interval_section(results, args)

    # --- verdicts off sample data; index table and scoreboard ----------------
    for key in [k for k in results if "_synthetic" in k]:
        if isinstance(results[key], dict):
            for field in ("verdict", "diff_lo", "diff_hi", "revenue_top10pct_verdict",
                          "revenue_top10pct_diff_lo", "revenue_top10pct_diff_hi"):
                results[key].pop(field, None)
    for _name, ds in INDEX_FILES:
        lapse, ret = results.get(f"lapse_{ds}"), results.get(f"lapse_{ds}_retention")
        if lapse is None or ret is None or _lapse_base_rate(lapse) is None:
            continue
        retention_base = 100.0 - _lapse_base_rate(lapse)
        ret["retention_base_rate_pct"] = retention_base
        ret["lift_over_random"] = ret["top10pct"]["model"] / retention_base
        ret["rule_lift_over_random"] = ret["top10pct"]["rule"] / retention_base
        lapse["nearly_everyone_lapses"] = _lapse_base_rate(lapse) > HIGH_BASE_RATE_PCT
    index = build_index(results)
    results["_index"] = {q: {"bottom_line": v["bottom_line"], "cells": {ds: c["text"] for ds, c in v["cells"].items()}}
                         for q, v in index.items()}
    render_index_table(index, ROOT / "docs" / "results" / "_verdicts" / "index_table.md")
    for key in ("who_to_mail_kdd98", "who_to_mail_cup98val"):
        if "cost_sweep" in results.get(key, {}):
            render_cost_sweep_table(results[key]["cost_sweep"], ROOT / "docs" / "results" / "_verdicts" / f"{key}_cost_sweep.md")
    sb_rows, sb_notes = scoreboard_rows(index)
    _render_themed(_scoreboard_chart, OUT_DIR / "scoreboard.png", sb_rows, sb_notes)

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
