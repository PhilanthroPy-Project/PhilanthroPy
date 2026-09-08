"""Plot the leakage measurement: walk-forward AUC, as-of vs whole-history.

The figure companion to :mod:`scripts.leakage_experiment` and
:mod:`scripts.real_data_leakage_experiment`. Those two scripts print five-seed
*means*; this one re-runs the identical cross-validation and keeps the per-fold
scores, so you can see the shape of the walk-forward backtest rather than one
number per condition. The fold means it reports reproduce the figures in
``docs/explanation/benchmarks.md`` and in the preprint: synthetic 0.625 as-of
against 0.750 whole-history, real 0.482 against 0.858.

Two panels, synthetic and real, one line per feature-construction rule, with the
seed spread shaded. The palette is the same two hues in both panels and was
checked for colourblind separation rather than chosen by eye.

The image deliberately carries no library name, logo or URL, so it can be posted
where promotional images are removed. Attribution belongs in the accompanying
text, not burned into the figure.

Running the real panel takes several minutes and downloads the KDD Cup 1998
archive (about 35 MB) on first use, via the same opt-in
:func:`philanthropy.datasets.fetch_kdd98_donors` the experiment script uses.
Nothing else here touches the network.

Usage:
    python scripts/render_leakage_chart.py --seeds 42        # quick look
    python scripts/render_leakage_chart.py                   # all five seeds
    python scripts/render_leakage_chart.py --cached          # re-plot, no refit
    python scripts/render_leakage_chart.py --out /tmp/figs   # choose a directory
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from sklearn.model_selection import cross_val_score

import leakage_experiment as synth
import real_data_leakage_experiment as real
from philanthropy.model_selection import FiscalYearGroupedSplitter

N_SPLITS = 3
AS_OF_COLOR = "#2a78d6"     # categorical slot 1
WHOLE_COLOR = "#eb6834"     # categorical slot 2
INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"


def per_fold(df, clf):
    """The committed ``_cv`` call with the per-fold array kept, not averaged.

    Mirrors ``_cv`` in both experiment scripts exactly: drop the final period
    (it is the held-out estimand, not CV data), score walk-forward ROC-AUC with
    the fiscal year as the group.
    """
    df = df[df["fy"] < df["fy"].max()]
    features = df[synth.FEATURES].to_numpy()
    labels = df["y"].to_numpy()
    fy = df["fy"].to_numpy()
    return cross_val_score(
        clf,
        features,
        labels,
        cv=FiscalYearGroupedSplitter(n_splits=N_SPLITS, drop_repeat_donors=False),
        groups=fy,
        scoring="roc_auc",
    )


def synthetic_scores(seeds):
    as_of_runs, whole_runs = [], []
    for seed in seeds:
        as_of, whole = synth._panel(seed)
        as_of_runs.append(per_fold(as_of, synth._clf()))
        whole_runs.append(per_fold(whole, synth._clf()))
    return np.array(as_of_runs), np.array(whole_runs)


def real_scores(seeds):
    as_of, whole = real._panel()          # the real data has no seed
    as_of_runs, whole_runs = [], []
    for seed in seeds:
        as_of_runs.append(per_fold(as_of, real._clf(seed)))
        whole_runs.append(per_fold(whole, real._clf(seed)))
    return np.array(as_of_runs), np.array(whole_runs)


def draw(panel, ax, as_of_runs, whole_runs, title, subtitle, show_legend):
    folds = np.arange(1, as_of_runs.shape[1] + 1)
    as_of_mean = as_of_runs.mean(axis=0)
    whole_mean = whole_runs.mean(axis=0)

    ax.set_facecolor(SURFACE)
    ax.axhline(0.5, color="#c9c8c3", linewidth=1, linestyle=(0, (4, 4)), zorder=1)
    ax.text(
        folds[0] - 0.06, 0.5, "coin flip", fontsize=8, color=MUTED,
        va="bottom", ha="left",
    )

    for mean, runs, color, label in (
        (whole_mean, whole_runs, WHOLE_COLOR, "Features built from the whole file"),
        (as_of_mean, as_of_runs, AS_OF_COLOR, "Features built as of the split date"),
    ):
        if runs.shape[0] > 1:
            ax.fill_between(
                folds, runs.min(axis=0), runs.max(axis=0),
                color=color, alpha=0.15, linewidth=0, zorder=2,
            )
        ax.plot(folds, mean, color=color, linewidth=2, label=label, zorder=3)
        ax.plot(
            folds, mean, "o", color=color, markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=4,
        )

    gap = whole_mean.mean() - as_of_mean.mean()
    ax.annotate(
        f"average gap  +{gap:.3f} AUC",
        xy=(folds[-1], (whole_mean[-1] + as_of_mean[-1]) / 2),
        xytext=(-4, 0), textcoords="offset points",
        fontsize=9, color=INK, ha="right", va="center",
    )

    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=14)
    ax.text(
        0, 1.015, subtitle, transform=ax.transAxes,
        fontsize=9, color=MUTED, ha="left", va="bottom",
    )
    ax.set_xlabel("Walk-forward fold (each tests one later period)", fontsize=9, color=MUTED)
    if panel == 0:
        ax.set_ylabel("ROC-AUC on the held-out period", fontsize=9, color=MUTED)
    ax.set_xticks(folds)
    ax.tick_params(labelleft=True)   # both panels readable when cropped
    ax.set_ylim(0.35, 1.0)
    ax.grid(axis="y", color="#e7e6e1", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d5d4cf")
    ax.tick_params(colors=MUTED, labelsize=9)
    if show_legend:
        ax.legend(
            loc="upper left", bbox_to_anchor=(0, -0.16), frameon=False,
            fontsize=9, labelcolor=INK, ncol=1, handlelength=1.6,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=synth.SEEDS)
    parser.add_argument(
        "--cached", action="store_true",
        help="Re-plot the last run's scores instead of recomputing them. Refitting "
             "the real panel takes several minutes, so use this for styling changes.",
    )
    parser.add_argument(
        "--out", default=".",
        help="Directory to write leakage_chart.png/.svg, the numbers JSON and the "
             "score cache into. Defaults to the current directory.",
    )
    args = parser.parse_args()

    out = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(out, "leakage_chart_scores.npz")
    if args.cached:
        z = np.load(cache)
        s_as_of, s_whole, r_as_of, r_whole = (
            z["s_as_of"], z["s_whole"], z["r_as_of"], z["r_whole"]
        )
        args.seeds = z["seeds"].tolist()
    else:
        started = time.time()
        s_as_of, s_whole = synthetic_scores(args.seeds)
        print(f"synthetic done in {time.time() - started:.0f}s", flush=True)
        started = time.time()
        r_as_of, r_whole = real_scores(args.seeds)
        print(f"real done in {time.time() - started:.0f}s", flush=True)
        np.savez(
            cache, seeds=np.array(args.seeds), s_as_of=s_as_of,
            s_whole=s_whole, r_as_of=r_as_of, r_whole=r_whole,
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    draw(
        0, axes[0], s_as_of, s_whole,
        "Simulated donors",
        "3,000 donors, 6 fiscal years, generated with known behaviour",
        show_legend=True,
    )
    draw(
        1, axes[1], r_as_of, r_whole,
        "Real donors (KDD Cup 1998)",
        "95,412 donors, 22 direct-mail campaigns, a real charity's file",
        show_legend=False,
    )
    fig.suptitle(
        "Scoring a donor model on features that peeked at the future makes it "
        "look far better than it is",
        fontsize=14, color=INK, x=0.008, ha="left", y=0.985,
    )
    fig.text(
        0.008, 0.005,
        f"Random forest, {len(args.seeds)} random seeds; shading spans the seeds. "
        "Both panels use the same expanding-window backtest; only the moment "
        "the features are computed differs.",
        fontsize=8, color=MUTED, ha="left", va="bottom",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))

    png = os.path.join(out, "leakage_chart.png")
    fig.savefig(png, dpi=200, facecolor=SURFACE)
    fig.savefig(os.path.join(out, "leakage_chart.svg"), facecolor=SURFACE)

    summary = {
        "seeds": args.seeds,
        "n_splits": N_SPLITS,
        "synthetic": {
            "as_of_per_fold": s_as_of.mean(axis=0).round(4).tolist(),
            "whole_per_fold": s_whole.mean(axis=0).round(4).tolist(),
            "as_of_mean": round(float(s_as_of.mean()), 4),
            "whole_mean": round(float(s_whole.mean()), 4),
        },
        "real": {
            "as_of_per_fold": r_as_of.mean(axis=0).round(4).tolist(),
            "whole_per_fold": r_whole.mean(axis=0).round(4).tolist(),
            "as_of_mean": round(float(r_as_of.mean()), 4),
            "whole_mean": round(float(r_whole.mean()), 4),
        },
    }
    with open(os.path.join(out, "leakage_chart_numbers.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
