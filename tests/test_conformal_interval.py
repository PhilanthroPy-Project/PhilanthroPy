"""
tests/test_conformal_interval.py
================================
Checks for ``GiftIntervalCalibrator`` and the interval metrics.

Each test names the input that makes it fail in its own comment. The two that
exist because a plausible implementation gets them wrong:

* ``test_floor_reads_alpha_as_an_exact_rational`` -- ``Fraction(1, 3)`` and
  ``Fraction(1 / 3)`` are different numbers, and the second refuses a
  calibration set the first certifies.
* ``test_clipping_leaves_coverage_bit_identical_and_narrows_width`` -- coverage
  is asserted with ``array_equal``, not a tolerance, because intersecting with a
  bound the target respects cannot move a single row.
"""

from __future__ import annotations

import warnings
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from philanthropy.metrics import interval_report, interval_score
from philanthropy.models import AskAmountRecommender, GiftIntervalCalibrator
from philanthropy.models._conformal_interval import (
    _exact_alpha,
    _min_calibration_size,
    _rank,
)

# ---------------------------------------------------------------------------
# Fixtures: a fitted regressor plus rows it has never seen
# ---------------------------------------------------------------------------


def _panel(n=400, seed=0, noise=2_000.0, intercept=20_000.0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, (n, 3))
    y = intercept + 60_000.0 * X[:, 0] + rng.normal(0.0, noise, n)
    return X, np.maximum(y, 0.0)


def _fitted(X, y, n_train=200):
    return LinearRegression().fit(X[:n_train], y[:n_train])


def _calibrated(alpha=0.05, n_cal=60, seed=0, **kwargs):
    X, y = _panel(seed=seed)
    model = _fitted(X, y)
    cal = GiftIntervalCalibrator(model, alpha=alpha, **kwargs)
    cal.fit(X[200:200 + n_cal], y[200:200 + n_cal])
    return cal, X[300:], y[300:]


# ---------------------------------------------------------------------------
# 1. The floor, in exact rational arithmetic
# ---------------------------------------------------------------------------


def test_floor_reads_alpha_as_an_exact_rational():
    # One third has no finite binary form. Fraction(1, 3) is exactly one third,
    # so (n + 1)(1 - alpha) == n holds at n = 2 and two rows certify. Both
    # float-derived spellings are the double, strictly above one third, so the
    # product lands above 2 and the floor moves to 3. Failing input: the same
    # alpha written two ways.
    assert _min_calibration_size(Fraction(1, 3)) == 2
    assert _min_calibration_size(Fraction(1 / 3)) == 3
    assert _min_calibration_size(Fraction(str(1 / 3))) == 3
    assert _rank(2, Fraction(1, 3)) == 2
    assert _rank(2, Fraction(1 / 3)) == 3  # exceeds n = 2, so it is refused


def test_exact_alpha_accepts_the_two_spellings_it_documents():
    X, y = _panel(n=210)
    model = _fitted(X, y)
    exact = GiftIntervalCalibrator(model, alpha=Fraction(1, 3))
    exact.fit(X[200:202], y[200:202])
    assert exact.n_calibration_ == 2
    assert exact.rank_ == 2
    assert exact.attained_level_ == pytest.approx(2 / 3)

    binary = GiftIntervalCalibrator(model, alpha=Fraction(1 / 3))
    with pytest.raises(ValueError, match="floor is 3"):
        binary.fit(X[200:202], y[200:202])


def test_float_alpha_is_read_as_the_decimal_it_prints_as():
    assert _exact_alpha(0.05) == Fraction(1, 20)
    assert _exact_alpha(0.05) != Fraction(0.05)  # the binary value is larger
    assert _min_calibration_size(0.05) == 19
    assert _min_calibration_size(0.1) == 9


def test_floor_is_a_ceiling_not_a_truncation():
    # int(1 / alpha - 1) is the off-by-one: it truncates 13.2857 to 13, and 13
    # rows cannot certify 0.93. Failing input: alpha = 0.07 at n = 13.
    assert int(1 / 0.07 - 1) == 13
    assert _min_calibration_size(0.07) == 14
    assert _rank(13, Fraction(7, 100)) == 14  # one rank past the largest score

    X, y = _panel(n=220)
    model = _fitted(X, y)
    with pytest.raises(ValueError, match="floor is 14"):
        GiftIntervalCalibrator(model, alpha=0.07).fit(X[200:213], y[200:213])
    ok = GiftIntervalCalibrator(model, alpha=0.07).fit(X[200:214], y[200:214])
    assert ok.rank_ == 14
    assert ok.n_calibration_ == 14


@pytest.mark.parametrize("alpha", [0.5, 0.2, 0.1, 0.07, 0.05, 0.02, Fraction(1, 3)])
def test_the_floor_and_the_rank_check_are_the_same_fact(alpha):
    # The refusal is written as n < floor; the guarantee is r <= n. If those two
    # ever disagree the error message is describing a different check from the
    # one being run. Failing input: any n either side of the floor.
    exact = _exact_alpha(alpha)
    floor = _min_calibration_size(exact)
    for n in range(1, floor + 5):
        assert (n < floor) == (_rank(n, exact) > n), (alpha, n)


def test_refusal_message_carries_the_floor_and_the_level():
    X, y = _panel(n=210)
    model = _fitted(X, y)
    with pytest.raises(ValueError) as excinfo:
        GiftIntervalCalibrator(model, alpha=0.05).fit(X[200:205], y[200:205])
    message = str(excinfo.value)
    assert "5 calibration row(s)" in message
    assert "floor is 19" in message
    assert "0.95" in message


def test_no_parameter_switches_the_floor_off():
    # A flag is how a certification check becomes decoration. Failing input:
    # any future keyword named to bypass the refusal.
    params = set(GiftIntervalCalibrator(LinearRegression()).get_params(deep=False))
    assert params == {
        "estimator",
        "alpha",
        "score",
        "difficulty_estimator",
        "lower_bound",
    }


# ---------------------------------------------------------------------------
# 2. The attained level, not the requested one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_cal, expected_rank, expected_level",
    [(19, 19, 19 / 20), (20, 20, 20 / 21), (30, 30, 30 / 31), (39, 38, 38 / 40)],
)
def test_attained_level_is_reported_not_the_requested_one(
    n_cal, expected_rank, expected_level
):
    # A caller asking for 0.95 at n = 30 gets 0.9677. Failing input: reporting
    # 1 - alpha, which is right for none of these four.
    cal, X_test, _ = _calibrated(alpha=0.05, n_cal=n_cal)
    assert cal.rank_ == expected_rank
    assert cal.attained_level_ == pytest.approx(expected_level)
    assert cal.requested_level_ == 0.95

    interval = cal.predict_gift_interval(X_test)
    assert interval.requested_level == 0.95
    assert np.all(interval.rank == expected_rank)
    assert interval.attained_level == pytest.approx(expected_level)
    assert interval.rank.shape == interval.lower.shape == (len(X_test),)


def test_coverage_holds_on_average_over_calibration_draws():
    # The guarantee is marginal over the calibration draw, so one draw can dip.
    # Failing input: an off-by-one in the rank, which biases the average down
    # by roughly 1 / (n + 1) and shows up here even though no single seed proves
    # it.
    attained, covered = [], []
    for seed in range(40):
        cal, X_test, y_test = _calibrated(alpha=0.1, n_cal=40, seed=seed)
        interval = cal.predict_gift_interval(X_test)
        attained.append(cal.attained_level_)
        covered.append(
            np.mean((y_test >= interval.lower) & (y_test <= interval.upper))
        )
    assert np.mean(covered) >= np.mean(attained) - 0.02


# ---------------------------------------------------------------------------
# 3. Clipping to the support
# ---------------------------------------------------------------------------


def test_clipping_leaves_coverage_bit_identical_and_narrows_width():
    # Wide intervals on small amounts push the lower bound below zero. A gift
    # cannot be negative, so y >= 0 and y in [l, u] give y in [max(l, 0), u]:
    # the same rows are covered, bit for bit. Failing input: clipping the upper
    # bound too, or clipping at a positive bound, either of which moves rows.
    X, y = _panel(n=400, noise=25_000.0, intercept=8_000.0)
    model = _fitted(X, y)
    unclipped = GiftIntervalCalibrator(model, alpha=0.05, lower_bound=None)
    unclipped.fit(X[200:300], y[200:300])
    clipped = GiftIntervalCalibrator(model, alpha=0.05, lower_bound=0.0)
    clipped.fit(X[200:300], y[200:300])

    X_test, y_test = X[300:], y[300:]
    raw = unclipped.predict_gift_interval(X_test)
    cut = clipped.predict_gift_interval(X_test)

    assert (raw.lower < 0).any(), "fixture no longer exercises the clip"
    assert np.all(y_test >= 0.0)

    covered_raw = (y_test >= raw.lower) & (y_test <= raw.upper)
    covered_cut = (y_test >= cut.lower) & (y_test <= cut.upper)
    assert np.array_equal(covered_raw, covered_cut)

    assert np.median(cut.upper - cut.lower) < np.median(raw.upper - raw.lower)
    assert np.array_equal(cut.upper, raw.upper)
    assert np.array_equal(cut.lower, np.maximum(raw.lower, 0.0))


def test_calibration_targets_below_the_bound_are_refused():
    # The coverage identity above needs y >= lower_bound. A negative target is
    # evidence the bound is wrong, so it raises instead of silently voiding the
    # argument. Failing input: one calibration target of -1.0.
    X, y = _panel(n=300)
    model = _fitted(X, y)
    y_bad = y.copy()
    y_bad[250] = -1.0
    with pytest.raises(ValueError, match="lower_bound"):
        GiftIntervalCalibrator(model, alpha=0.05).fit(X[200:300], y_bad[200:300])

    ok = GiftIntervalCalibrator(model, alpha=0.05, lower_bound=None)
    ok.fit(X[200:300], y_bad[200:300])
    assert ok.n_calibration_ == 100


# ---------------------------------------------------------------------------
# 4. Calibrating within the segment
# ---------------------------------------------------------------------------


_SEGMENTS = {
    # Row counts deliberately imbalanced, the way a real file is: most rows are
    # annual donors, so a pooled calibration set is theirs whether anyone
    # intended that or not.
    "annual": (1_500, 400.0),
    "leadership": (300, 6_000.0),
    "principal": (200, 60_000.0),
}
_N_TRAIN, _N_CAL = 800, 400


def _segmented_panel(seed=0):
    """Three segments sharing a trend and differing in residual scale."""
    rng = np.random.default_rng(seed)
    frames = []
    for name, (n_rows, scale) in _SEGMENTS.items():
        X = rng.uniform(0.0, 1.0, (n_rows, 3))
        y = 40_000.0 + 60_000.0 * X[:, 0] + rng.normal(0.0, scale, n_rows)
        frames.append((X, np.maximum(y, 0.0), np.full(n_rows, name)))
    X = np.vstack([f[0] for f in frames])
    y = np.concatenate([f[1] for f in frames])
    seg = np.concatenate([f[2] for f in frames])
    order = rng.permutation(len(y))
    return X[order], y[order], seg[order]


def _split():
    """``(train, calibration, test)`` slices of the segmented panel."""
    cal_end = _N_TRAIN + _N_CAL
    return slice(0, _N_TRAIN), slice(_N_TRAIN, cal_end), slice(cal_end, None)


def test_segment_calibration_covers_the_segment_pooling_starves():
    # A pooled calibration set is dominated by whichever segment supplies most
    # of the rows, so a segment with a wider residual distribution is
    # under-covered however much marginal data is added. Failing input: pooling
    # 100 rows drawn from the base and reading the marginal coverage as if it
    # applied per segment.
    X, y, seg = _segmented_panel()
    train, cal, test = _split()
    model = _fitted(X, y, n_train=_N_TRAIN)
    X_cal, y_cal, seg_cal = X[cal], y[cal], seg[cal]
    X_test, y_test, seg_test = X[test], y[test], seg[test]

    pooled = GiftIntervalCalibrator(model, alpha=0.05).fit(X_cal, y_cal)
    grouped = GiftIntervalCalibrator(model, alpha=0.05).fit(
        X_cal, y_cal, groups=seg_cal
    )

    def by_segment(interval):
        return {
            name: float(
                np.mean(
                    (y_test[seg_test == name] >= interval.lower[seg_test == name])
                    & (y_test[seg_test == name] <= interval.upper[seg_test == name])
                )
            )
            for name in np.unique(seg_test)
        }

    pooled_cov = by_segment(pooled.predict_gift_interval(X_test))
    grouped_cov = by_segment(
        grouped.predict_gift_interval(X_test, groups=seg_test)
    )

    # Measured: principal covers at 0.41 pooled against 0.95 grouped, while the
    # marginal number over all three segments clears 0.95 either way.
    assert min(pooled_cov.values()) < 0.80, pooled_cov
    # 0.88 rather than 0.90: the worst segment has 68 test rows, so one row is
    # 0.015 of the estimate and the bound has to leave room for sampling noise
    # rather than tracking the measured 0.953 exactly.
    assert min(grouped_cov.values()) > 0.88, grouped_cov
    assert min(grouped_cov.values()) > min(pooled_cov.values()) + 0.3, (
        pooled_cov,
        grouped_cov,
    )

    # And the segment that pooling over-served gets a narrower interval, not
    # merely a differently-wrong one: measured $3.1k grouped against $78k
    # pooled.
    pooled_iv = pooled.predict_gift_interval(X_test)
    grouped_iv = grouped.predict_gift_interval(X_test, groups=seg_test)
    annual = seg_test == "annual"
    pooled_width = float(np.median((pooled_iv.upper - pooled_iv.lower)[annual]))
    grouped_width = float(np.median((grouped_iv.upper - grouped_iv.lower)[annual]))
    assert grouped_width < 0.5 * pooled_width, (grouped_width, pooled_width)

    # Each segment is calibrated on its own rows, so the ranks differ.
    assert set(grouped.n_calibration_) == set(_SEGMENTS)
    assert sum(grouped.n_calibration_.values()) == len(y_cal)
    assert len(set(grouped.rank_.values())) > 1, grouped.rank_


def test_more_pooled_rows_do_not_close_the_gap():
    # The pooled shortfall is a bias, not a sample-size problem. Failing input:
    # tripling the pooled calibration set, which leaves the worst segment where
    # it was.
    X, y, seg = _segmented_panel()
    _, _, test = _split()
    model = _fitted(X, y, n_train=_N_TRAIN)
    X_test, y_test, seg_test = X[test], y[test], seg[test]

    worst = []
    for n_cal in (100, 400, 1_200):
        cal = GiftIntervalCalibrator(model, alpha=0.05).fit(
            X[_N_TRAIN:_N_TRAIN + n_cal], y[_N_TRAIN:_N_TRAIN + n_cal]
        )
        interval = cal.predict_gift_interval(X_test)
        worst.append(
            min(
                float(
                    np.mean(
                        (y_test[seg_test == name] >= interval.lower[seg_test == name])
                        & (y_test[seg_test == name] <= interval.upper[seg_test == name])
                    )
                )
                for name in np.unique(seg_test)
            )
        )
    # Measured 0.15 at 100 rows and 0.41 at 400; twelve times the data does not
    # reach the level, because the shortfall is not a sample-size problem.
    assert max(worst) < 0.80, worst


def test_group_below_the_floor_is_refused_not_pooled():
    # Failing input: one segment with 5 rows at alpha = 0.05, which a pooling
    # implementation would silently calibrate at another segment's capacity
    # level.
    X, y, seg = _segmented_panel()
    _, cal, _ = _split()
    model = _fitted(X, y, n_train=_N_TRAIN)
    idx = np.concatenate([
        np.where(seg[cal] == "annual")[0][:60],
        np.where(seg[cal] == "principal")[0][:5],
    ]) + _N_TRAIN
    with pytest.raises(ValueError) as excinfo:
        GiftIntervalCalibrator(model, alpha=0.05).fit(
            X[idx], y[idx], groups=seg[idx]
        )
    message = str(excinfo.value)
    assert "'principal' (5 row(s))" in message
    assert "19 row(s) needed per group" in message
    assert "annual" not in message.split("Short:")[1].split(".")[0]


def test_unseen_group_at_predict_is_refused():
    # Failing input: a segment that had no calibration rows. Borrowing another
    # segment's quantile is exactly the pooling groups= exists to prevent.
    X, y, seg = _segmented_panel()
    _, cal_s, test_s = _split()
    model = _fitted(X, y, n_train=_N_TRAIN)
    keep = seg[cal_s] != "principal"
    cal = GiftIntervalCalibrator(model, alpha=0.05).fit(
        X[cal_s][keep], y[cal_s][keep], groups=seg[cal_s][keep]
    )
    with pytest.raises(ValueError, match="no calibration rows for group"):
        cal.predict_gift_interval(X[test_s], groups=seg[test_s])


def test_groups_must_be_supplied_consistently():
    cal, X_test, _ = _calibrated()
    with pytest.raises(ValueError, match="fitted without groups"):
        cal.predict_gift_interval(X_test, groups=np.zeros(len(X_test)))

    X, y, seg = _segmented_panel()
    _, cal_s, test_s = _split()
    model = _fitted(X, y, n_train=_N_TRAIN)
    grouped = GiftIntervalCalibrator(model, alpha=0.05).fit(
        X[cal_s], y[cal_s], groups=seg[cal_s]
    )
    with pytest.raises(ValueError, match="fitted with groups"):
        grouped.predict_gift_interval(X[test_s])


def test_groups_length_and_shape_are_checked():
    X, y, seg = _segmented_panel()
    _, cal_s, _ = _split()
    model = _fitted(X, y, n_train=_N_TRAIN)
    cal = GiftIntervalCalibrator(model, alpha=0.05)
    with pytest.raises(ValueError, match="label"):
        cal.fit(X[cal_s], y[cal_s], groups=seg[cal_s][:-1])
    with pytest.raises(ValueError, match="one-dimensional"):
        cal.fit(X[cal_s], y[cal_s], groups=seg[cal_s].reshape(-1, 1))


def test_donor_ids_passed_as_groups_are_refused():
    # The argument is the segment, not the donor. One row per group cannot
    # certify anything, and the floor says so instead of the class inventing a
    # per-donor guarantee. Failing input: groups = donor_id.
    X, y = _panel(n=300)
    model = _fitted(X, y)
    donor_id = np.arange(200, 300)
    with pytest.raises(ValueError, match="19 row\\(s\\) needed per group"):
        GiftIntervalCalibrator(model, alpha=0.05).fit(
            X[200:300], y[200:300], groups=donor_id
        )


# ---------------------------------------------------------------------------
# 5. The three conformity scores
# ---------------------------------------------------------------------------


def test_absolute_score_is_symmetric_around_the_point():
    cal, X_test, _ = _calibrated(score="absolute", n_cal=100)
    interval = cal.predict_gift_interval(X_test)
    point = cal.predict(X_test)
    inside = interval.lower > 0.0  # unclipped rows only
    assert np.allclose(
        (point - interval.lower)[inside], (interval.upper - point)[inside]
    )


def test_difficulty_score_scales_width_with_the_estimate():
    # Failing input: a difficulty estimator whose output is ignored, which
    # would give every row the same width.
    X, y = _panel(n=400)
    model = _fitted(X, y)

    def difficulty(features):
        return 1.0 + 9.0 * np.asarray(features)[:, 1]

    cal = GiftIntervalCalibrator(
        model,
        alpha=0.05,
        score="difficulty",
        difficulty_estimator=difficulty,
        lower_bound=None,  # clipping would mask the proportionality below
    ).fit(X[200:300], y[200:300])
    interval = cal.predict_gift_interval(X[300:])
    width = interval.upper - interval.lower

    # width is exactly 2 * q * sigma, so width / sigma is one constant.
    ratio = width / difficulty(X[300:])
    assert np.allclose(ratio, ratio[0])
    easy, hard = X[300:, 1] < 0.2, X[300:, 1] > 0.8
    assert np.median(width[hard]) > 3.0 * np.median(width[easy])


def test_difficulty_estimator_must_be_positive_and_present():
    X, y = _panel(n=300)
    model = _fitted(X, y)
    with pytest.raises(ValueError, match="needs a difficulty_estimator"):
        GiftIntervalCalibrator(model, alpha=0.05, score="difficulty").fit(
            X[200:300], y[200:300]
        )
    # Failing input: a zero scale, which divides; and a negative one, which
    # inverts the interval.
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="strictly positive"):
            GiftIntervalCalibrator(
                model,
                alpha=0.05,
                score="difficulty",
                difficulty_estimator=lambda f, v=bad: np.full(len(f), v),
            ).fit(X[200:300], y[200:300])
    with pytest.raises(ValueError, match="one positive scale per row"):
        GiftIntervalCalibrator(
            model,
            alpha=0.05,
            score="difficulty",
            difficulty_estimator=lambda f: np.ones(3),
        ).fit(X[200:300], y[200:300])


def test_difficulty_estimator_may_be_an_estimator_not_only_a_callable():
    X, y = _panel(n=400)
    model = _fitted(X, y)
    residual = np.abs(y[:200] - model.predict(X[:200]))
    sigma_model = LinearRegression().fit(X[:200], np.maximum(residual, 1.0))
    cal = GiftIntervalCalibrator(
        model,
        alpha=0.05,
        score="difficulty",
        difficulty_estimator=sigma_model,
    ).fit(X[200:300], y[200:300])
    interval = cal.predict_gift_interval(X[300:])
    assert np.all(interval.upper >= interval.lower)


def test_log_score_is_multiplicative_and_never_negative():
    # Width scales with the amount, which is the shape a right-skewed gift
    # distribution wants. Failing input: an additive score, which gives the
    # $500 prospect and the $5m prospect the same absolute width.
    X, y = _panel(n=400, noise=8_000.0, intercept=10_000.0)
    model = _fitted(X, y)
    cal = GiftIntervalCalibrator(model, alpha=0.05, score="log").fit(
        X[200:300], y[200:300]
    )
    interval = cal.predict_gift_interval(X[300:])
    point = cal.predict(X[300:])
    assert np.all(interval.lower >= 0.0)
    assert np.all(interval.upper >= interval.lower)
    small, large = point < np.median(point), point > np.median(point)
    width = interval.upper - interval.lower
    assert np.median(width[large]) > np.median(width[small])


def test_log_score_refuses_negative_dollars():
    # log1p is defined above -1, and a negative dollar amount is a sign the
    # score is the wrong one. Failing input: a target of -5.0.
    X, y = _panel(n=300)
    model = _fitted(X, y)
    y_bad = y.copy()
    y_bad[250] = -5.0
    with pytest.raises(ValueError, match="lower_bound"):
        GiftIntervalCalibrator(model, alpha=0.05, score="log").fit(
            X[200:300], y_bad[200:300]
        )
    with pytest.raises(ValueError, match="non-negative"):
        GiftIntervalCalibrator(
            model, alpha=0.05, score="log", lower_bound=None
        ).fit(X[200:300], y_bad[200:300])


def test_equal_tailed_two_rank_is_not_on_offer():
    # Two order statistics at alpha / 2 MORE than double the floor -- the ratio is
    # (2 - alpha)/(1 - alpha), above two at every level -- for no gain the
    # one-rank scores do not already deliver. Failing input: score="quantile"
    # or any other value, which must raise rather than fall through.
    X, y = _panel(n=300)
    model = _fitted(X, y)
    with pytest.raises(ValueError, match="score must be one of"):
        GiftIntervalCalibrator(model, score="equal_tailed").fit(
            X[200:300], y[200:300]
        )


# ---------------------------------------------------------------------------
# 6. Wrapper mechanics
# ---------------------------------------------------------------------------


def test_prefit_is_required_with_an_explanation():
    # Failing input: an unfitted regressor. Calibrating on rows the regressor
    # was trained on makes every score in-sample.
    X, y = _panel(n=300)
    with pytest.raises(NotFittedError, match="already fitted"):
        GiftIntervalCalibrator(LinearRegression()).fit(X[200:300], y[200:300])


def test_predict_delegates_to_the_wrapped_estimator_unchanged():
    # The point a gift officer sees must not move when an interval is added.
    X, y = _panel(n=400)
    model = _fitted(X, y)
    cal = GiftIntervalCalibrator(model, alpha=0.05).fit(X[200:300], y[200:300])
    assert np.array_equal(cal.predict(X[300:]), model.predict(X[300:]))


def test_wraps_the_packages_own_dollar_regressors():
    X, y = _panel(n=400)
    ask = AskAmountRecommender(max_iter=30, random_state=0).fit(X[:200], y[:200])
    cal = GiftIntervalCalibrator(ask, alpha=0.05).fit(X[200:300], y[200:300])
    interval = cal.predict_gift_interval(X[300:])
    assert interval.lower.shape == (100,)
    assert np.all(interval.upper >= interval.lower)


def test_per_row_estimators_only():
    # A forecaster returning one value per horizon step, not per row, is the
    # failing input; FinancialForecastModel.predict_revenue_forecast is the
    # real one.
    class Horizon(LinearRegression):
        def predict(self, X):
            return np.arange(4, dtype=float)

    X, y = _panel(n=300)
    model = Horizon().fit(X[:200], y[:200])
    with pytest.raises(ValueError, match="per-row regressors only"):
        GiftIntervalCalibrator(model, alpha=0.05).fit(X[200:300], y[200:300])


def test_alpha_must_be_a_level_and_a_number():
    X, y = _panel(n=300)
    model = _fitted(X, y)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="0 < alpha < 1"):
            GiftIntervalCalibrator(model, alpha=bad).fit(X[200:300], y[200:300])
    with pytest.raises(TypeError, match="alpha must be"):
        GiftIntervalCalibrator(model, alpha="0.05").fit(X[200:300], y[200:300])


def test_alpha_accepts_decimal_and_integer_spellings():
    # Failing input: Decimal("0.05"), which a float-only reader would reject or
    # silently round through binary.
    assert _exact_alpha(Decimal("0.05")) == Fraction(1, 20)
    assert _exact_alpha(1) == Fraction(1)  # then rejected as a level, below
    X, y = _panel(n=300)
    model = _fitted(X, y)
    cal = GiftIntervalCalibrator(model, alpha=Decimal("0.05"))
    cal.fit(X[200:300], y[200:300])
    assert cal.rank_ == 96
    with pytest.raises(ValueError, match="0 < alpha < 1"):
        GiftIntervalCalibrator(model, alpha=1).fit(X[200:300], y[200:300])


def test_wrapping_something_without_predict_is_refused():
    # Failing input: a fitted transformer. It passes check_is_fitted, so the
    # prefit check alone would let it through to an AttributeError deeper in.
    X, y = _panel(n=300)
    scaler = StandardScaler().fit(X[:200])
    with pytest.raises(TypeError, match="must have a predict method"):
        GiftIntervalCalibrator(scaler).fit(X[200:300], y[200:300])


def test_list_input_is_accepted_like_any_sklearn_estimator():
    # Failing input: a nested list, which has no .shape for the row count.
    cal, X_test, _ = _calibrated()
    interval = cal.predict_gift_interval(X_test[:4].tolist())
    assert interval.lower.shape == (4,)


def test_predict_before_fit_raises():
    X, y = _panel(n=300)
    model = _fitted(X, y)
    cal = GiftIntervalCalibrator(model)
    with pytest.raises(NotFittedError):
        cal.predict_gift_interval(X[200:300])


def test_feature_count_is_enforced_after_fit():
    cal, X_test, _ = _calibrated()
    with pytest.raises(ValueError):
        cal.predict_gift_interval(X_test[:, :2])


# ---------------------------------------------------------------------------
# 7. The metrics
# ---------------------------------------------------------------------------


def test_interval_score_is_the_published_formula():
    # Row 1 is covered and costs its width, 15. Row 2 misses by 10 at
    # alpha = 0.5, so it costs 15 + (2 / 0.5) * 10 = 55. Failing input: dropping
    # the 2 / alpha weight, which returns 20.0 instead of 35.0.
    assert interval_score([10.0, 30.0], [5.0, 5.0], [20.0, 20.0], alpha=0.5) == 35.0
    assert interval_score([10.0], [5.0], [20.0], alpha=0.5) == 15.0
    assert interval_score([0.0], [5.0], [20.0], alpha=0.5) == 15.0 + 4.0 * 5.0


def test_interval_score_cannot_be_gamed_by_widening():
    # Properness, as a check: widening past the point of coverage costs more
    # than it earns. Failing input: a score that is only width, or only
    # coverage.
    y = np.array([10.0, 12.0, 11.0])
    tight = interval_score(y, np.full(3, 9.0), np.full(3, 13.0), alpha=0.1)
    wide = interval_score(y, np.zeros(3), np.full(3, 1_000.0), alpha=0.1)
    assert tight < wide


def test_interval_report_separates_valid_from_useful():
    # Both arms cover everything. Only the ratio tells them apart, which is the
    # whole point of reporting it. Failing input: judging on coverage alone.
    y = np.array([1_000.0, 2_000.0, 3_000.0])
    useful = interval_report(y, y * 0.7, y * 1.4, alpha=0.05)
    useless = interval_report(y, np.full(3, 1.0), np.full(3, 500_000.0), alpha=0.05)
    assert useful.coverage == useless.coverage == 1.0
    assert useful.width_ratio < 1.0
    assert useless.width_ratio > 100.0
    assert useless.median_bound_ratio > 1_000.0
    assert useful.score_mean < useless.score_mean


def test_interval_report_median_and_trimmed_mean_resist_one_row():
    # A single missed major gift carries the mean. Failing input: reporting the
    # mean alone, which moves by three orders of magnitude while the interval
    # is unchanged for every other row.
    y = np.concatenate([np.full(99, 100.0), [10_000_000.0]])
    lower, upper = np.zeros(100), np.full(100, 200.0)
    report = interval_report(y, lower, upper, alpha=0.05)
    assert report.coverage == 0.99
    assert report.score_mean > 1_000_000.0
    assert report.score_median == 200.0
    assert report.score_trimmed_mean == 200.0
    assert np.isnan(report.median_bound_ratio)  # every lower bound is zero


def test_width_ratio_when_the_median_gift_is_zero():
    # A segment of declined asks has a median target of 0, and the ratio is
    # undefined rather than enormous. Failing input: median_target == 0, which
    # divides. Coverage counts the ternary as one line, so this branch needs its
    # own test to be exercised at all.
    y = np.zeros(4)
    wide = interval_report(y, np.zeros(4), np.full(4, 500.0), alpha=0.05)
    assert wide.median_target == 0.0
    assert wide.width_ratio == np.inf
    exact = interval_report(y, np.zeros(4), np.zeros(4), alpha=0.05)
    assert np.isnan(exact.width_ratio)
    assert exact.coverage == 1.0


def test_interval_report_trim_bounds_are_checked():
    y, lower, upper = np.arange(1.0, 11.0), np.zeros(10), np.full(10, 20.0)
    assert interval_report(y, lower, upper, trim=0.0).score_trimmed_mean == 20.0
    with pytest.raises(ValueError, match="0 <= trim < 0.5"):
        interval_report(y, lower, upper, trim=0.5)


def test_interval_metrics_reject_bad_input():
    with pytest.raises(ValueError, match="same shape"):
        interval_score([1.0, 2.0], [0.0], [3.0])
    with pytest.raises(ValueError, match="non-empty"):
        interval_score([], [], [])
    with pytest.raises(ValueError, match="0 < alpha < 1"):
        interval_score([1.0], [0.0], [2.0], alpha=0.0)
    with pytest.raises(ValueError, match="upper must be >= lower"):
        interval_score([1.0], [5.0], [2.0])


def test_report_grades_a_real_calibrated_interval():
    cal, X_test, y_test = _calibrated(alpha=0.05, n_cal=100)
    interval = cal.predict_gift_interval(X_test)
    report = interval_report(y_test, interval.lower, interval.upper, alpha=0.05)
    assert report.n == len(y_test)
    assert report.requested_level == 0.95
    assert report.coverage > 0.9
    assert 0.0 < report.width_ratio < 1.0
    assert report.score_mean >= report.median_width


def test_no_warnings_on_the_common_path():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cal, X_test, y_test = _calibrated(alpha=0.05, n_cal=100)
        interval = cal.predict_gift_interval(X_test)
        interval_report(y_test, interval.lower, interval.upper)
