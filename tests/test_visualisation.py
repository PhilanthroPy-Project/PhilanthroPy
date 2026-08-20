"""
tests/test_visualisation.py
Headless test suite for philanthropy.visualisation.

These assert on the artists the functions actually draw: bar heights, patch
counts, axis limits, legend text. `isinstance(ax, plt.Axes)` passes for any
function that returns an Axes, including one that plotted nothing.
"""

import pytest

# Visualisation is an optional extra (`pip install philanthropy[viz]`); skip this
# whole module rather than error-collect when matplotlib/seaborn aren't installed.
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from philanthropy.visualisation import plot_affinity_distribution, plot_retention_waterfall


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close('all')


@pytest.fixture
def scores():
    return np.random.default_rng(0).uniform(0, 100, 100)


# ---------------------------------------------------------------------------
# plot_affinity_distribution
# ---------------------------------------------------------------------------

def test_affinity_histogram_draws_the_requested_bins(scores):
    ax = plot_affinity_distribution(scores)
    assert ax.get_title() == "Affinity Score Distribution"
    assert ax.get_xlabel() == "Affinity Score (0-100)"
    assert ax.get_ylabel() == "Frequency"
    # bins=20 in _plots.py: one patch per bin, and they must hold every score.
    assert len(ax.patches) == 20
    assert sum(p.get_height() for p in ax.patches) == pytest.approx(len(scores))


def test_affinity_histogram_x_limits_span_the_data(scores):
    ax = plot_affinity_distribution(scores)
    lo, hi = ax.get_xlim()
    assert lo <= scores.min()
    assert hi >= scores.max()


def test_affinity_histogram_labelled_variant_has_a_two_class_legend(scores):
    labels = np.random.default_rng(1).integers(0, 2, len(scores))
    ax = plot_affinity_distribution(scores, labels=labels)

    assert ax.get_title() == "Affinity Score Distribution by Major-Gift Label"
    legend = ax.get_legend()
    assert legend is not None
    assert {t.get_text() for t in legend.get_texts()} == {"Major", "Non-Major"}
    assert sum(p.get_height() for p in ax.patches) == pytest.approx(len(scores))


def test_affinity_histogram_single_class_labels_legend_has_one_entry():
    rng = np.random.default_rng(2)
    ax = plot_affinity_distribution(
        rng.uniform(50, 100, 30), labels=np.ones(30, dtype=int)
    )
    legend = ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["Major"]
    assert sum(p.get_height() for p in ax.patches) == pytest.approx(30)


@pytest.mark.parametrize("constant", [
    np.zeros(50), np.full(50, 100.0), np.array([50.0])
])
def test_affinity_histogram_degenerate_inputs_still_plot_every_row(constant):
    ax = plot_affinity_distribution(constant)
    assert sum(p.get_height() for p in ax.patches) == pytest.approx(len(constant))


# ---------------------------------------------------------------------------
# plot_retention_waterfall
# ---------------------------------------------------------------------------

def test_waterfall_bar_heights_and_bases_match_the_flows():
    ax = plot_retention_waterfall(
        starting_donors=1000, acquired=200, lapsed=150, recovered=50
    )
    assert ax.get_title() == "Donor Retention Waterfall"
    assert ax.get_ylabel() == "Number of Donors"

    heights = [p.get_height() for p in ax.patches]
    bases = [p.get_y() for p in ax.patches]
    # Starting, Acquired, Lapsed (negative flow), Recovered, Ending.
    assert heights == [1000, 200, -150, 50, 1100]
    assert bases == [0, 1000, 1200, 1050, 0]
    # The ending bar is the arithmetic sum of the flows before it.
    assert heights[4] == heights[0] + heights[1] + heights[2] + heights[3]


def test_waterfall_labels_every_bar_with_its_absolute_size():
    ax = plot_retention_waterfall(
        starting_donors=1000, acquired=200, lapsed=150, recovered=50
    )
    assert [t.get_text() for t in ax.texts] == ["1000", "200", "150", "50", "1100"]
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "Starting", "Acquired", "Lapsed", "Recovered", "Ending"
    ]


def test_waterfall_all_zero_period_draws_five_flat_bars():
    ax = plot_retention_waterfall(
        starting_donors=0, acquired=0, lapsed=0, recovered=0
    )
    assert [p.get_height() for p in ax.patches] == [0, 0, 0, 0, 0]


def test_waterfall_scales_to_large_files():
    ax = plot_retention_waterfall(
        starting_donors=100000, acquired=50000, lapsed=20000, recovered=10000
    )
    assert [p.get_height() for p in ax.patches] == [
        100000, 50000, -20000, 10000, 140000
    ]
