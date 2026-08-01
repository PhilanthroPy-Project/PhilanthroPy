"""Regression guards for the correctness/security fixes from the audit.

Each test would fail against the pre-fix code, so they lock the behaviour in.
"""

import numpy as np
import pandas as pd
import pytest

from philanthropy.cli import _neutralise_csv_injection
from philanthropy.metrics import selection_rate_by_group, disparate_impact_ratio
from philanthropy.models import (
    DonorPropensityModel,
    LapsePredictor,
    MajorGiftClassifier,
    PlannedGivingIntentScorer,
)
from philanthropy.preprocessing import (
    DischargeToSolicitationWindowTransformer,
    EncounterRecencyTransformer,
    RFMTransformer,
    WealthScreeningImputerKNN,
)


def test_rfm_reference_date_frozen_in_fit():
    train = pd.DataFrame({
        "donor_id": [1, 1, 2],
        "gift_date": ["2019-01-01", "2020-01-01", "2019-06-01"],
        "gift_amount": [100.0, 200.0, 50.0],
    })
    t = RFMTransformer().fit(train)
    # Reference is frozen from TRAINING data, not recomputed at transform.
    assert hasattr(t, "reference_date_")
    assert t.reference_date_ == pd.Timestamp("2020-01-01")

    # A transform batch containing a LATER gift must not shift donor 1's recency:
    # it is measured from the frozen 2020 reference, not the batch max (2022).
    batch = pd.DataFrame({
        "donor_id": [1, 3],
        "gift_date": ["2020-01-01", "2022-01-01"],
        "gift_amount": [200.0, 999.0],
    })
    out = t.transform(batch)
    recency_donor1 = out.loc[out["donor_id"] == 1, "recency"].iloc[0]
    assert recency_donor1 == 0


def test_lapse_predictor_single_class_no_indexerror():
    rng = np.random.default_rng(0)
    X = rng.random((20, 3))

    # All-negative training fold: score must be all zeros, not an IndexError.
    m0 = LapsePredictor(n_estimators=5, random_state=0).fit(X, np.zeros(20, dtype=int))
    scores0 = m0.predict_lapse_score(X)
    assert scores0.shape == (20,)
    assert np.all(scores0 == 0.0)

    # All-positive fold: the sole class is the positive one → 100.0.
    m1 = LapsePredictor(n_estimators=5, random_state=0).fit(X, np.ones(20, dtype=int))
    assert np.all(m1.predict_lapse_score(X) == 100.0)


def test_fairness_metrics_reject_nan_group_labels():
    with pytest.raises(ValueError, match="missing"):
        selection_rate_by_group([1, 0, 1], [np.nan, 1.0, 1.0])
    with pytest.raises(ValueError, match="missing"):
        disparate_impact_ratio([1, 0, 1], [np.nan, 1.0, 1.0])


def test_cli_neutralises_csv_formula_injection():
    df = pd.DataFrame({
        "name": ["=cmd|'/c calc'!A1", "+1", "-2", "@x", "safe"],
        "score": [1, 2, 3, 4, 5],
    })
    out = _neutralise_csv_injection(df)
    assert list(out["name"]) == ["'=cmd|'/c calc'!A1", "'+1", "'-2", "'@x", "safe"]
    # Numeric columns are untouched.
    assert list(out["score"]) == [1, 2, 3, 4, 5]


def test_planned_giving_intent_score_contract():
    rng = np.random.default_rng(0)
    X = rng.random((60, 4))
    y = (X[:, 0] + rng.random(60) * 0.1 > 0.5).astype(int)
    m = PlannedGivingIntentScorer(n_estimators=10, random_state=0).fit(X, y)

    scores = m.predict_intent_score(X)
    assert scores.shape == (60,)
    assert np.all(scores >= 0.0) and np.all(scores <= 100.0)
    # The two domain methods are the same contract.
    np.testing.assert_array_equal(scores, m.predict_intent_score(X))
    # Score is P(class=1) * 100, rounded to 2 dp.
    expected = np.round(m.predict_proba(X)[:, 1] * 100.0, 2)
    np.testing.assert_array_almost_equal(scores, expected)


def test_discharge_window_raises_on_wrong_dataframe_columns():
    # A FiscalYearTransformer upstream used to feed `fiscal_year` in positionally,
    # which silently produced an all-zero feature block instead of an error.
    t = DischargeToSolicitationWindowTransformer()
    wrong = pd.DataFrame({"fiscal_year": [2020, 2021], "fy_offset": [0.1, 0.2]})
    t.fit(wrong)
    with pytest.raises(ValueError, match="not found in X"):
        t.transform(wrong)


def test_discharge_window_ndarray_path_still_positional():
    # check_estimator feeds bare ndarrays with no column names; keep that branch.
    t = DischargeToSolicitationWindowTransformer()
    X = np.array([[100.0], [10.0], [np.nan]])
    out = t.fit(X).transform(X)
    assert out.shape == (3, 2)
    np.testing.assert_array_equal(out[:, 0], [1.0, 0.0, 0.0])


def test_wealth_screening_imputer_knn_ndarray_path():
    rng = np.random.default_rng(0)
    X = rng.random((30, 4))
    X[X < 0.2] = np.nan  # inject missingness
    out = WealthScreeningImputerKNN(
        strategy="knn", n_neighbors=3, add_indicator=False
    ).fit_transform(X)
    assert out.shape[0] == 30
    assert not np.any(np.isnan(out))


def test_donor_propensity_affinity_score_single_class_forks():
    # A degenerate training fold gives decision_function a (n, 1) proba block.
    # Both single-class branches must yield the documented 0/100 endpoints.
    rng = np.random.default_rng(0)
    X = rng.random((25, 3))

    all_negative = DonorPropensityModel(n_estimators=5, random_state=0).fit(
        X, np.zeros(25, dtype=int)
    )
    np.testing.assert_array_equal(
        all_negative.decision_function(X), np.full(25, -0.5)
    )
    np.testing.assert_array_equal(
        all_negative.predict_affinity_score(X), np.zeros(25)
    )

    all_positive = DonorPropensityModel(n_estimators=5, random_state=0).fit(
        X, np.ones(25, dtype=int)
    )
    np.testing.assert_array_equal(
        all_positive.decision_function(X), np.full(25, 0.5)
    )
    np.testing.assert_array_equal(
        all_positive.predict_affinity_score(X), np.full(25, 100.0)
    )


def test_donor_propensity_affinity_score_multiclass_fork():
    # Three moves-management stages: decision_function returns the raw (n, 3)
    # proba matrix and the affinity score reads column 1.
    rng = np.random.default_rng(1)
    X = rng.random((60, 3))
    y = np.tile([0, 1, 2], 20)

    m = DonorPropensityModel(n_estimators=5, random_state=0).fit(X, y)
    df = m.decision_function(X)
    assert df.shape == (60, 3)

    scores = m.predict_affinity_score(X)
    assert scores.shape == (60,)
    np.testing.assert_array_almost_equal(scores, np.round(df[:, 1] * 100, 2))
    assert ((scores >= 0.0) & (scores <= 100.0)).all()


def test_major_gift_classifier_reports_real_n_iter():
    rng = np.random.default_rng(2)
    X = rng.random((80, 4))
    y = (X[:, 0] > 0.5).astype(int)
    m = MajorGiftClassifier(max_iter=10, random_state=0).fit(X, y)
    # Not the hardcoded 1 it used to report to satisfy check_estimator.
    assert m.n_iter_ == 10


def test_encounter_transformer_no_overflow_on_extreme_span():
    # Two representable dates >292 years apart overflow a datetime64[ns]
    # timedelta (int64). The transformer must fall back to day-resolution
    # instead of raising OverflowError. Regression for that crash.
    df = pd.DataFrame({"last_encounter_date": ["1806-01-01", "2099-01-01"]})
    out = EncounterRecencyTransformer().fit_transform(df)
    assert out.shape == (2, 3)
    assert np.isfinite(out[:, 0]).all()  # days_since finite for both rows


def test_csv_injection_neutralised_in_pandas_string_dtype_columns():
    # The neutraliser used to iterate select_dtypes(include=["object"]).
    # pandas 4 stops returning `str`-dtype columns for that query, which would
    # silently leave donor-supplied text unescaped — the exact CWE-1236 hole
    # the function exists to close. Column dtype must not decide this.
    df = pd.DataFrame({
        "name": pd.array(["=cmd|'/c calc'!A1", "safe"], dtype="string"),
        "note": ["@SUM(1+1)", "fine"],
        "score": [1.0, 2.0],
    })
    out = _neutralise_csv_injection(df)

    assert list(out["name"]) == ["'=cmd|'/c calc'!A1", "safe"]
    assert list(out["note"]) == ["'@SUM(1+1)", "fine"]
    assert list(out["score"]) == [1.0, 2.0]
