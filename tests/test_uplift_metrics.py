"""
tests/test_uplift_metrics.py: Qini curve and Qini area.
"""

import numpy as np
import pytest

from philanthropy.metrics import qini_auc_score, qini_curve


def _rct(n=4000, seed=0):
    """Randomised appeal where only donors with ``x > 0`` respond to it."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    treatment = rng.integers(0, 2, size=n)
    true_uplift = np.where(x > 0, 0.4, 0.0)
    y = (rng.random(n) < 0.2 + treatment * true_uplift).astype(int)
    return y, x, treatment


def test_hand_computed_example():
    y = [1, 1, 0, 1, 0, 0]
    t = [1, 0, 1, 0, 1, 0]
    u = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    # k=1: 1 - 0          =  1   (no control yet)
    # k=2: 1 - 1 * 1/1    =  0
    # k=3: 1 - 1 * 2/1    = -1
    # k=4: 1 - 2 * 2/2    = -1
    # k=5: 1 - 2 * 3/2    = -2
    # k=6: 1 - 2 * 3/3    = -1
    x, gains = qini_curve(y, u, t)
    np.testing.assert_allclose(x, [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1])
    np.testing.assert_allclose(gains, [0, 1, 0, -1, -1, -2, -1])
    # Trapezoids over steps of 1/6: (0+1 + 1+0 + 0-1 + -1-1 + -1-2 + -2-1)/12
    # = -7/12; the random line to (1, -1) has area -1/2.
    assert qini_auc_score(y, u, t) == pytest.approx(-7 / 12 + 1 / 2)


def test_matches_brute_force_loop():
    y, _, t = _rct(n=300, seed=3)
    u = np.random.default_rng(4).random(300)
    x, gains = qini_curve(y, u, t)
    order = np.argsort(-u)
    expected = [0.0]
    for k in range(1, 301):
        top = order[:k]
        yt, yc = y[top][t[top] == 1].sum(), y[top][t[top] == 0].sum()
        nt, nc = (t[top] == 1).sum(), (t[top] == 0).sum()
        expected.append(yt - (yc * nt / nc if nc else 0.0))
    np.testing.assert_allclose(gains, expected)
    np.testing.assert_allclose(x, np.arange(301) / 300)


def test_tied_scores_are_one_cut_off_and_order_free():
    y = np.array([1, 0, 1, 0, 1, 1])
    t = np.array([1, 0, 0, 1, 1, 0])
    u = np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.1])
    x, gains = qini_curve(y, u, t)
    np.testing.assert_allclose(x, [0, 0.5, 5 / 6, 1])
    perm = np.array([2, 0, 1, 5, 4, 3])
    x2, gains2 = qini_curve(y[perm], u[perm], t[perm])
    np.testing.assert_allclose(x2, x)
    np.testing.assert_allclose(gains2, gains)


def test_perfect_ordering_beats_random_and_random_is_near_zero():
    y, x, t = _rct()
    oracle = qini_auc_score(y, np.where(x > 0, 0.4, 0.0), t)
    rng = np.random.default_rng(1)
    random_scores = [qini_auc_score(y, rng.random(len(y)), t) for _ in range(30)]
    assert oracle > 0
    assert oracle > max(random_scores)
    assert abs(np.mean(random_scores)) < 0.1 * oracle
    # Reversing the ranking targets the non-responders first: worse than random.
    assert qini_auc_score(y, -x, t) < 0


def test_amount_outcomes_are_accepted():
    y, x, t = _rct(n=500)
    amounts = y * 50.0
    assert qini_auc_score(amounts, x, t) == pytest.approx(
        50.0 * qini_auc_score(y, x, t)
    )


@pytest.mark.parametrize(
    "y, u, t, match",
    [
        ([1, 0], [0.1, 0.2], [1], "same length"),
        ([], [], [], "empty"),
        ([1, np.nan], [0.1, 0.2], [1, 0], "finite"),
        ([1, 0], [0.1, 0.2], [1, 2], "binary"),
        ([1, 0], [0.1, 0.2], [1, 1], "Both arms"),
    ],
)
def test_rejects_bad_input(y, u, t, match):
    with pytest.raises(ValueError, match=match):
        qini_curve(y, u, t)
