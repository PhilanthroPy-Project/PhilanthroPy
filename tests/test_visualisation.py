"""
tests/test_visualisation.py
Headless test suite for philanthropy.visualisation.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from philanthropy.visualisation import plot_affinity_distribution, plot_retention_waterfall


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close('all')


def test_plot_affinity_distribution_returns_axes():
    scores = np.random.uniform(0, 100, 100)
    ax = plot_affinity_distribution(scores)
    assert isinstance(ax, plt.Axes)


def test_plot_affinity_distribution_labels_none():
    scores = np.random.uniform(0, 100, 100)
    ax = plot_affinity_distribution(scores, labels=None)
    assert ax.get_title() == "Affinity Score Distribution"


def test_plot_affinity_distribution_with_labels():
    scores = np.random.uniform(0, 100, 100)
    labels = np.random.randint(0, 2, 100)
    ax = plot_affinity_distribution(scores, labels=labels)
    assert ax.get_title() == "Affinity Score Distribution by Major-Gift Label"


def test_plot_affinity_distribution_all_zeros():
    scores = np.zeros(50)
    ax = plot_affinity_distribution(scores)
    assert isinstance(ax, plt.Axes)


def test_plot_affinity_distribution_all_hundreds():
    scores = np.full(50, 100.0)
    ax = plot_affinity_distribution(scores)
    assert isinstance(ax, plt.Axes)


def test_plot_retention_waterfall_returns_axes():
    ax = plot_retention_waterfall(
        starting_donors=1000, acquired=200, lapsed=150, recovered=50
    )
    assert isinstance(ax, plt.Axes)


def test_plot_retention_waterfall_title_and_bars():
    ax = plot_retention_waterfall(
        starting_donors=1000, acquired=200, lapsed=150, recovered=50
    )
    assert ax.get_title() == "Donor Retention Waterfall"
    assert len(ax.patches) == 5


def test_plot_retention_waterfall_zero_values():
    ax = plot_retention_waterfall(
        starting_donors=0, acquired=0, lapsed=0, recovered=0
    )
    assert isinstance(ax, plt.Axes)


def test_plot_retention_waterfall_large_values():
    ax = plot_retention_waterfall(
        starting_donors=100000, acquired=50000, lapsed=20000, recovered=10000
    )
    assert isinstance(ax, plt.Axes)


def test_plot_affinity_distribution_single_value():
    scores = np.array([50.0])
    ax = plot_affinity_distribution(scores)
    assert isinstance(ax, plt.Axes)


def test_plot_affinity_distribution_with_labels_all_major():
    scores = np.random.uniform(50, 100, 30)
    labels = np.ones(30, dtype=int)
    ax = plot_affinity_distribution(scores, labels=labels)
    assert isinstance(ax, plt.Axes)


def test_plots_close_cleanly():
    plot_affinity_distribution(np.random.uniform(0, 100, 50))
    plot_retention_waterfall(100, 50, 25, 10)
    plt.close('all')
    assert True
