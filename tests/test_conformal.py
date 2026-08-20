"""Tests for the split-conformal p-value of a donor score."""

import numpy as np
import pytest

from philanthropy.metrics import conformal_pvalue


def test_exact_lattice_values():
    # n = 9 calibration scores 0..8. |{i : cal_i >= s}| is 0, 5 and 9.
    cal = np.arange(9, dtype=float)
    assert conformal_pvalue(cal, [8.5, 4.0, -1.0]) == pytest.approx([0.1, 0.6, 1.0])


def test_ties_are_counted():
    # A score equal to a calibration point counts that point as >= itself.
    cal = np.array([1.0, 2.0, 3.0])
    # {2.0, 3.0} are >= 2.0 → (1 + 2) / 4.
    assert conformal_pvalue(cal, [2.0])[0] == pytest.approx(0.75)


def test_never_zero_and_never_above_one():
    # The bug the n+1 denominator and the 1+ numerator exist to prevent:
    # a p-value of 0 for an unbeatable score, or one above 1 for a hopeless one.
    cal = np.linspace(0.0, 1.0, 50)
    p = conformal_pvalue(cal, [1e9, -1e9, 0.5])
    assert p.min() == pytest.approx(1.0 / 51.0)
    assert p.max() == pytest.approx(1.0)
    assert np.all(p > 0.0)
    assert np.all(p <= 1.0)


def test_marginal_validity_is_exact_on_the_lattice():
    # Leave-one-out over m+1 exchangeable, distinct scores: each rotation must
    # produce a distinct lattice point, so the p-values are exactly uniform on
    # {1/(m+1), ..., 1} and P(p <= alpha) <= alpha holds with no slack.
    values = np.arange(20, dtype=float) * 1.7
    p = np.array(
        [
            conformal_pvalue(np.delete(values, j), [values[j]])[0]
            for j in range(values.size)
        ]
    )
    assert sorted(p) == pytest.approx(sorted(np.arange(1, 21) / 20.0))
    for alpha in (0.05, 0.1, 0.25, 0.5):
        assert np.mean(p <= alpha) <= alpha + 1e-12


def test_nan_scores_propagate():
    p = conformal_pvalue([1.0, 2.0, 3.0], [2.0, np.nan])
    assert p[0] == pytest.approx(0.75)
    assert np.isnan(p[1])


def test_empty_scores_returns_empty():
    assert conformal_pvalue([1.0, 2.0], []).shape == (0,)


@pytest.mark.parametrize(
    "bad_calibration",
    [[], [[1.0, 2.0], [3.0, 4.0]], [1.0, np.nan], [1.0, np.inf]],
)
def test_invalid_calibration_raises(bad_calibration):
    with pytest.raises(ValueError):
        conformal_pvalue(bad_calibration, [1.0])
