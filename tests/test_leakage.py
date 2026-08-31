"""
tests/test_leakage.py
======================
Temporal data-leakage prevention tests for PhilanthroPy pipelines.

These tests verify that the ``EncounterTransformer`` and surrounding pipeline
infrastructure **cannot** be exploited to inject future-period target values
into fitted transformer state, a critical correctness guarantee for any
time-series or cross-fiscal-year split in prospect-management models.

Background
----------
In longitudinal donor analytics, leakage occurs when information about
a future time window (e.g., next fiscal year's gift amounts) inadvertently
influences the parameters fitted to historical data.  The three most common
leakage vectors in medical philanthropy pipelines are:

1. **Target leakage via imputation**: an imputer fitted on the *full* dataset
   (including test rows) learns statistics contaminated by future target values.
2. **Encounter summary leakage**: a transformer that re-aggregates the encounter
   table on *every transform call* would expose future discharge dates from
   held-out rows.
3. **Feature calendar leakage**: computing ``days_since_last_discharge`` using
   the *test-set* reference date rather than the *train-set* snapshot date.

All three vectors are tested here.

Run with:
    pytest tests/test_leakage.py -v
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from philanthropy.preprocessing import (
    EncounterTransformer,
    WealthScreeningImputer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def train_encounters():
    """Encounter records for the *training* period (FY22 discharges only)."""
    return pd.DataFrame(
        {
            "donor_id": [1, 2, 3, 4, 1],
            "discharge_date": [
                "2022-01-10",
                "2022-03-05",
                "2022-07-20",
                "2022-11-30",
                "2022-12-01",
            ],
        }
    )


@pytest.fixture
def future_encounters():
    """Encounter records for the *future* period (FY23 discharges)."""
    return pd.DataFrame(
        {
            "donor_id": [1, 2, 5],
            "discharge_date": [
                "2023-02-14",  # future discharge for donor 1
                "2023-05-20",  # future discharge for donor 2
                "2023-08-01",  # new donor 5, only in future
            ],
        }
    )


@pytest.fixture
def train_gift_df():
    """Gift records for FY22 (training set)."""
    return pd.DataFrame(
        {
            "donor_id": [1, 2, 3, 4],
            "gift_date": ["2022-09-01", "2022-10-15", "2022-08-01", "2022-12-20"],
            "gift_amount": [10_000.0, 500.0, 250.0, 2_000.0],
        }
    )


@pytest.fixture
def test_gift_df():
    """Gift records for FY23 (held-out test set) with *inflated* future amounts."""
    return pd.DataFrame(
        {
            "donor_id": [1, 2, 3, 4, 5],
            "gift_date": ["2023-09-01", "2023-10-01", "2023-08-01", "2023-11-01", "2023-08-15"],
            "gift_amount": [
                1_000_000.0,  # extreme future gift, leakage would inject this
                9_999_999.0,
                5_555_555.0,
                7_777_777.0,
                3_333_333.0,
            ],
        }
    )


# ---------------------------------------------------------------------------
# Leakage Test 1: EncounterTransformer encounter_summary_ is frozen at fit()
# ---------------------------------------------------------------------------


class TestEncounterTransformerNoLeakage:
    """Verify that fit() freezes encounter_summary_ from training encounters only."""

    def test_encounter_summary_frozen_from_train_encounters(
        self, train_encounters, future_encounters, train_gift_df, test_gift_df
    ):
        """
        After fitting on train_encounters, the transformer must NOT incorporate
        future discharge dates when transforming the test set.

        Specifically: donor 1 has a *training* last_discharge of 2022-12-01
        and a *future* discharge of 2023-02-14.  The fitted transformer must
        use 2022-12-01, never 2023-02-14.
        """
        t = EncounterTransformer(encounter_df=train_encounters).set_output(transform="pandas")
        t.fit(train_gift_df)

        # Verify frozen last_discharge for donor 1 is from training period
        frozen_last = t.encounter_summary_.loc[1, "last_discharge"]
        assert frozen_last == pd.Timestamp("2022-12-01"), (
            f"Frozen last_discharge should be 2022-12-01 (training), "
            f"got {frozen_last}."
        )

        # Now create a 'leaky' encounter_df that includes future records
        leaky_encounters = pd.concat(
            [train_encounters, future_encounters], ignore_index=True
        )
        # Attempt to exploit by swapping out encounter_df AFTER fit
        t.encounter_df = leaky_encounters  # malicious reassignment

        # transform() must use encounter_summary_ (frozen) NOT encounter_df
        test_out = t.transform(test_gift_df.copy())

        # Compute days for donor 1 using only the training discharge 2022-12-01
        donor1_row = test_out.iloc[0]
        expected_days = (
            pd.Timestamp("2023-09-01") - pd.Timestamp("2022-12-01")
        ).days
        actual_days = donor1_row["days_since_last_discharge"]
        assert actual_days == pytest.approx(expected_days, abs=1), (
            f"days_since_last_discharge for donor 1 should reflect training "
            f"discharge (2022-12-01), not future discharge (2023-02-14).  "
            f"Expected ≈{expected_days}, got {actual_days}."
        )

    def test_new_donors_in_test_set_produce_nan(
        self, train_encounters, train_gift_df, test_gift_df
    ):
        """
        Donor 5 appears only in the test set.  After fitting on train_encounters,
        the transformer must NOT leak donor 5's future values; it must return NaN.
        """
        t = EncounterTransformer(encounter_df=train_encounters).set_output(transform="pandas")
        t.fit(train_gift_df)
        test_out = t.transform(test_gift_df)

        # Donor 5 is the last row (index 4)
        donor5_days = test_out.iloc[4]["days_since_last_discharge"]
        donor5_freq = test_out.iloc[4]["encounter_frequency_score"]

        assert np.isnan(donor5_days), (
            "Donor 5 (unseen at fit time) must yield NaN for "
            "days_since_last_discharge; returning a real value would "
            "constitute leakage of future data."
        )
        # log1p(0) == 0.0, unknown donors get frequency score of 0
        assert donor5_freq == pytest.approx(0.0), (
            "Donor 5 (unseen at fit time) must yield 0.0 encounter_frequency_score."
        )

    def test_gift_amounts_do_not_contaminate_encounter_summary(
        self, train_encounters, train_gift_df, test_gift_df
    ):
        """
        Deliberately attempt to pass extreme future gift amounts as  `y` during
        fit.  Verify that encounter_summary_ statistics are unaffected.

        This simulates an adversarial scenario where a buggy pipeline accidentally
        passes next year's gift amounts as training labels to the transformer's fit().
        """
        # Build a combined dataset that includes future gift amounts as labels
        future_amounts = np.array([1_000_000.0, 9_999_999.0, 5_555_555.0, 7_777_777.0])

        t = EncounterTransformer(encounter_df=train_encounters).set_output(transform="pandas")
        # Attempt to fit with future gift amounts as y; this must be ignored by EncounterTransformer
        t.fit(train_gift_df, y=future_amounts)

        # Encounter summary should still reflect encounter_df aggregation only
        summary = t.encounter_summary_
        # encounter_count for donor 1 should be 2 (two training discharges)
        assert summary.loc[1, "encounter_count"] == 2, (
            "encounter_summary_ must be computed from encounter_df only; "
            "future y values must not alter the count."
        )
        # last_discharge must still be max of training discharges for donor 1
        assert summary.loc[1, "last_discharge"] == pd.Timestamp("2022-12-01"), (
            "encounter_summary_ last_discharge must come from encounter_df only."
        )


# ---------------------------------------------------------------------------
# Leakage Test 2: WealthScreeningImputer fill values frozen at fit()
# ---------------------------------------------------------------------------


class TestWealthImputerNoLeakage:
    """Verify that imputation fill values cannot be contaminated by test-set data."""

    def test_fill_values_frozen_from_train_only(self):
        """
        After fitting on X_train, the fill value for estimated_net_worth must
        equal the training-set median.  Calling transform on X_test with extreme
        future values must NOT update fill_values_.
        """
        X_train = pd.DataFrame(
            {
                "estimated_net_worth": [100_000.0, np.nan, 300_000.0, np.nan],
                "gift_amount": [1000.0, 500.0, 2000.0, 750.0],
            }
        )
        X_test = pd.DataFrame(
            {
                "estimated_net_worth": [np.nan, 99_000_000.0],  # extreme test values
                "gift_amount": [10_000.0, 5_000.0],
            }
        )

        imp = WealthScreeningImputer(wealth_cols=["estimated_net_worth"], strategy="median")
        imp.fit(X_train)

        # Training median of [100_000, 300_000] = 200_000
        expected_fill = 200_000.0
        assert imp.fill_values_["estimated_net_worth"] == pytest.approx(expected_fill), (
            f"Fill value should be training median={expected_fill}, "
            f"got {imp.fill_values_['estimated_net_worth']}."
        )

        # Transform test set
        imp.transform(X_test)

        # Fill value must remain frozen
        assert imp.fill_values_["estimated_net_worth"] == pytest.approx(expected_fill), (
            "fill_values_ must not be updated after transform(); "
            "mutation after fit() is a leakage vector."
        )

    def test_wealth_imputer_fill_frozen_from_train_transform_test_no_leakage(self):
        """
        WealthScreeningImputer fit on X_train: fill_values_ must ONLY use X_train.
        transform(X_test) with extreme future wealth must NOT update fill_values_.
        """
        X_train = pd.DataFrame(
            {
                "estimated_net_worth": [100_000.0, np.nan, 300_000.0],
                "real_estate_value": [np.nan, 500_000.0, np.nan],
            }
        )
        X_test = pd.DataFrame(
            {
                "estimated_net_worth": [99_000_000.0],
                "real_estate_value": [50_000_000.0],
            }
        )

        imputer = WealthScreeningImputer(
            wealth_cols=["estimated_net_worth", "real_estate_value"],
            strategy="median",
        )
        imputer.fit(X_train)

        # Training median of real_estate_value: [500_000] = 500_000
        assert imputer.fill_values_["real_estate_value"] == pytest.approx(500_000.0), (
            "After fit(X_train), fill should be 500_000 (training median)."
        )

        _ = imputer.transform(X_test)

        # Fill value must still be 500_000, not contaminated by 50_000_000
        final_fill = imputer.fill_values_["real_estate_value"]
        assert final_fill == pytest.approx(500_000.0), (
            f"fill_values_['real_estate_value'] changed after transform(X_test): "
            f"got {final_fill}, expected 500_000.0.  "
            "This is a leakage bug: test-set statistics must not alter fitted values."
        )


# ---------------------------------------------------------------------------
# Leakage Test 3: Temporal split, future gift amounts cannot flow backward
# ---------------------------------------------------------------------------


class TestTemporalSplitLeakage:
    """
    Simulate a train/test split along the fiscal-year boundary and verify that
    future gift amounts do not contaminate training-period encounter features.
    """

    def test_future_gift_amounts_do_not_alter_encounter_features(
        self, train_encounters, train_gift_df, test_gift_df
    ):
        """
        Fit EncounterTransformer on train_gift_df, then transform both splits.
        The training-split features must be identical whether we transform
        train_gift_df alone or after computing test-split features.
        """
        t = EncounterTransformer(encounter_df=train_encounters).set_output(transform="pandas")
        t.fit(train_gift_df)

        out_train_A = t.transform(train_gift_df.copy())

        # Now transform the test set (with outrageous future gift amounts)
        _ = t.transform(test_gift_df.copy())

        # Transform train again after processing test, must be identical
        out_train_B = t.transform(train_gift_df.copy())

        pd.testing.assert_frame_equal(
            out_train_A.reset_index(drop=True),
            out_train_B.reset_index(drop=True),
            check_like=True,
            obj="Training-set encounter features must be identical before and "
                "after transforming the test set.",
        )

    def test_encounter_count_invariant_to_test_set_size(
        self, train_encounters, train_gift_df
    ):
        """
        encounter_frequency_score for training donors must not change regardless
        of how many test rows are processed after fit().
        """
        t = EncounterTransformer(encounter_df=train_encounters).set_output(transform="pandas")
        t.fit(train_gift_df)

        out_small_test = t.transform(train_gift_df.iloc[:1].copy())
        out_large_test = t.transform(
            pd.concat([train_gift_df] * 100, ignore_index=True)
        )

        # Frequency score for donor 1 must be consistent across different test sizes
        freq_small = out_small_test.iloc[0]["encounter_frequency_score"]
        freq_large = out_large_test.iloc[0]["encounter_frequency_score"]

        assert freq_small == pytest.approx(freq_large), (
            "encounter_frequency_score must be invariant to test-set size; "
            "it changed between small and large transforms."
        )


from hypothesis import given, settings, strategies as st
from hypothesis.extra.pandas import data_frames, column, range_indexes

class TestTemporalLeakagePrevention:
    @settings(max_examples=100)
    @given(
        df_train=data_frames(
            columns=[
                column("estimated_net_worth", elements=st.one_of(st.floats(0, 1e7), st.just(np.nan))),
                column("real_estate_value", elements=st.one_of(st.floats(0, 1e7), st.just(np.nan)))
            ],
            index=range_indexes(min_size=2, max_size=50)
        ),
        n_test_samples=st.integers(min_value=1, max_value=1000)
    )
    def test_wealth_imputer_fill_value_independent_of_test_size(self, df_train, n_test_samples):
        imputer = WealthScreeningImputer(strategy="median", add_indicator=False)
        imputer.fit(df_train)
        initial_fill = imputer.fill_values_.copy()
        
        rng = np.random.default_rng(n_test_samples)
        df_test = pd.DataFrame({
            "estimated_net_worth": rng.uniform(0, 1e7, n_test_samples),
            "real_estate_value": rng.uniform(0, 1e7, n_test_samples)
        })
        imputer.transform(df_test)
        assert imputer.fill_values_ == initial_fill

    def test_encounter_summary_frozen_after_fit(self, train_encounters, train_gift_df, test_gift_df):
        enc_df = train_encounters.copy()
        t = EncounterTransformer(encounter_df=enc_df).set_output(transform="pandas")
        t.fit(train_gift_df)
        
        out1 = t.transform(test_gift_df)
        
        enc_df.loc[len(enc_df)] = {"donor_id": 1, "discharge_date": "2023-12-31"}
        out2 = t.transform(test_gift_df)
        
        pd.testing.assert_frame_equal(out1, out2)

    def test_no_future_data_in_encounter_summary(self, train_encounters, train_gift_df):
        future_row = pd.DataFrame({"donor_id": [1], "discharge_date": ["2025-01-01"]})
        enc_df = pd.concat([train_encounters, future_row], ignore_index=True)
        
        t = EncounterTransformer(encounter_df=enc_df, allow_negative_days=False).set_output(transform="pandas")
        t.fit(train_gift_df)
        
        out = t.transform(train_gift_df)
        # donor 1 first gift is "2022-09-01", discharge is "2025-01-01". 
        # so it's a negative day -> NaN since allow_negative_days=False
        donor1_mask = train_gift_df["donor_id"] == 1
        days = out.loc[donor1_mask, "days_since_last_discharge"]
        assert days.isna().all()


class TestFiscalYearNoLeakage:
    """Verify that FiscalYearTransformer only uses the provided columns in X."""

    def test_fiscal_year_stateless(self):
        """FiscalYearTransformer is stateless; verify it doesn't store data."""
        from philanthropy.preprocessing import FiscalYearTransformer
        df1 = pd.DataFrame({"gift_date": ["2023-01-01"]})
        df2 = pd.DataFrame({"gift_date": ["2024-01-01"]})

        t = FiscalYearTransformer(fiscal_year_start=7).set_output(transform="pandas")
        t.fit(df1)
        out1 = t.transform(df1)
        # 2023-01-01 (Jan) is FY23 if FY starts in July (7)
        assert out1.iloc[0]["fiscal_year"] == 2023

        # Transforming df2 should not be influenced by df1
        out2 = t.transform(df2)
        assert out2.iloc[0]["fiscal_year"] == 2024


def test_fiscal_year_transformer_uses_no_future_data():
    """
    FiscalYearTransformer is stateless: fit on training split, transform
    test split with completely disjoint date ranges; both must be correct.
    """
    from philanthropy.preprocessing import FiscalYearTransformer

    train_df = pd.DataFrame({"gift_date": ["2020-07-01", "2020-12-31"]})
    test_df = pd.DataFrame({"gift_date": ["2023-07-01", "2023-12-31"]})

    t = FiscalYearTransformer(fiscal_year_start=7).set_output(transform="pandas")
    t.fit(train_df)

    # Only n_features_in_ and feature_names_in_ should be set
    fitted_attrs = [a for a in vars(t) if a.endswith("_")]
    non_schema_attrs = [
        a for a in fitted_attrs if a not in ("n_features_in_", "feature_names_in_")
    ]
    assert len(non_schema_attrs) == 0, (
        f"FiscalYearTransformer must be stateless (no domain fitted attrs): "
        f"{non_schema_attrs}"
    )

    out_train = t.transform(train_df)
    out_test = t.transform(test_df)

    # Training: July 1, 2020 → FY2021; Dec 31, 2020 → FY2021
    assert out_train.iloc[0]["fiscal_year"] == 2021
    assert out_train.iloc[1]["fiscal_year"] == 2021
    # Test: July 1, 2023 → FY2024; Dec 31, 2023 → FY2024
    assert out_test.iloc[0]["fiscal_year"] == 2024
    assert out_test.iloc[1]["fiscal_year"] == 2024


def test_encounter_transformer_summary_is_fit_time_snapshot():
    """
    Mutate encounter_df AFTER fit() completes.
    Assert that transform() output is unchanged, proving encounter_summary_
    is a snapshot, not a view into the original DataFrame.
    """
    enc_df = pd.DataFrame({
        "donor_id": [1, 2],
        "discharge_date": ["2022-01-01", "2022-06-01"],
    })
    X_train = pd.DataFrame({
        "donor_id": [1, 2],
        "gift_date": ["2022-09-01", "2022-10-01"],
        "gift_amount": [1000.0, 500.0],
    })
    X_test = pd.DataFrame({
        "donor_id": [1, 2],
        "gift_date": ["2023-01-01", "2023-02-01"],
        "gift_amount": [2000.0, 750.0],
    })

    t = EncounterTransformer(encounter_df=enc_df)
    t.fit(X_train)
    original_output = t.transform(X_test).copy()

    # Mutate the original enc_df AFTER fit, should not affect transform()
    enc_df.iloc[0, enc_df.columns.get_loc("discharge_date")] = "2099-01-01"
    post_mutation_output = t.transform(X_test)

    np.testing.assert_array_equal(
        original_output,
        post_mutation_output,
        err_msg=(
            "transform() output changed after mutating encounter_df, "
            "encounter_summary_ must be a snapshot taken at fit() time."
        ),
    )


def test_wealth_imputer_fill_statistics_are_fold_specific_in_cv():
    """
    In 5-fold CV, fill statistics computed in each training fold must NOT all
    be identical, which would indicate the full dataset was used (leakage).
    """
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({
        "estimated_net_worth": np.where(
            rng.random(n) < 0.4, np.nan, rng.lognormal(14, 2, n)
        )
    })
    y = rng.integers(0, 2, n)

    fill_values_per_fold = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, _ in skf.split(X, y):
        imp = WealthScreeningImputer(
            wealth_cols=["estimated_net_worth"],
            strategy="median",
            add_indicator=False,
        )
        imp.fit(X.iloc[train_idx])
        fill_values_per_fold.append(imp.fill_values_["estimated_net_worth"])

    # If all fold fill values are identical, test-set leakage is likely
    assert len(set(fill_values_per_fold)) > 1, (
        "All fold fill values are identical: possible full-dataset leakage. "
        f"Fill values: {fill_values_per_fold}"
    )


# ---------------------------------------------------------------------------
# Conformal interval coverage: a row split flatters it, a donor split does not
# ---------------------------------------------------------------------------


def _donor_panel():
    """A panel whose capacity term lives with the donor, not the row.

    Donors 0-79 have rows in FY2019 (training), FY2020 (calibration) and
    FY2021. Donors 80-119 appear only in FY2021, so they are new to the file.
    ``donor_id`` is a feature, which is what lets the regressor memorise a
    donor's capacity term rather than learning it from the other columns.
    """
    rng = np.random.default_rng(11)
    capacity = rng.normal(0.0, 30_000.0, 120)

    rows = []
    for donor in range(80):
        for fy, n in ((2019, 10), (2020, 1), (2021, 1)):
            rows.extend((donor, fy) for _ in range(n))
    for donor in range(80, 120):
        rows.extend((donor, 2021) for _ in range(2))

    donor_id = np.array([r[0] for r in rows], dtype=float)
    fiscal_year = np.array([r[1] for r in rows])
    x1 = rng.uniform(0.0, 1.0, len(rows))
    y = np.maximum(
        40_000.0
        + 40_000.0 * x1
        + capacity[donor_id.astype(int)]
        + rng.normal(0.0, 500.0, len(rows)),
        0.0,
    )
    return np.column_stack([donor_id, x1]), y, fiscal_year, donor_id


def test_row_split_flatters_conformal_coverage_and_a_donor_split_does_not():
    """Calibrating on donors the model already knows hides the real coverage.

    The capacity term is memorised from training, so it is absent from the
    residuals of every donor in the training file and present in full for a
    donor who is new. Calibrate on FY2020 rows (all training donors) and the
    interval looks excellent on FY2021 rows from those same donors and falls
    apart on the donors that are new in FY2021.

    ``FiscalYearGroupedSplitter(drop_repeat_donors=True)`` is what separates the
    two folds. The failing input for anyone tempted to split by row instead: the
    shared-donor coverage below sits near the attained level while the honest
    number is a third of it.

    The assertions are on the *gap*, not on the shared-donor fold clearing the
    attained level. That fold has no guarantee to clear: exchangeability is
    exactly what the leak breaks, so its coverage is an artefact and drifts with
    the boosting fit (0.96 locally, 0.93 on another sklearn build). The gap is
    the finding and it is stable.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    from philanthropy.model_selection import FiscalYearGroupedSplitter
    from philanthropy.models import GiftIntervalCalibrator

    X, y, fiscal_year, donor_id = _donor_panel()

    train = fiscal_year == 2019
    model = HistGradientBoostingRegressor(
        max_iter=400, min_samples_leaf=1, random_state=0
    ).fit(X[train], y[train])

    # Calibration and evaluation live in FY2020/FY2021; FY2019 is the model's.
    later = fiscal_year >= 2020
    X_later, y_later = X[later], y[later]
    fy_later, donors_later = fiscal_year[later], donor_id[later]

    # Flag off: the FY2021 fold keeps donors that are also in the FY2020
    # calibration rows. Flag on: those donors are dropped.
    splitter = FiscalYearGroupedSplitter(n_splits=1, drop_repeat_donors=False)
    cal_idx, all_test_idx = next(
        iter(splitter.split(X_later, groups=fy_later))
    )
    grouped = FiscalYearGroupedSplitter(n_splits=1, drop_repeat_donors=True)
    with pytest.warns(UserWarning, match="drop_repeat_donors"):
        _, new_donor_idx = next(
            iter(
                grouped.split(
                    X_later, groups=np.column_stack([fy_later, donors_later])
                )
            )
        )
    shared_donor_idx = np.setdiff1d(all_test_idx, new_donor_idx)
    assert shared_donor_idx.size and new_donor_idx.size

    calibrator = GiftIntervalCalibrator(model, alpha=0.05).fit(
        X_later[cal_idx], y_later[cal_idx]
    )

    def coverage(idx):
        interval = calibrator.predict_gift_interval(X_later[idx])
        return float(
            np.mean(
                (y_later[idx] >= interval.lower)
                & (y_later[idx] <= interval.upper)
            )
        )

    shared = coverage(shared_donor_idx)
    new = coverage(new_donor_idx)

    # Measured: shared 0.96, new 0.35, against an attained level of 0.9506.
    assert new < 0.6, (shared, new)
    assert shared > 0.85, (shared, new)
    assert shared - new > 0.35, (shared, new)
    assert new < calibrator.attained_level_ - 0.3, (new, calibrator.attained_level_)


# ---------------------------------------------------------------------------
# Registry meta-test: every stateful transformer must have a leakage test
#
# "Stateful" is derived from the source, not from an author-maintained list, so
# the registry cannot rot: any class in preprocessing.__all__ whose module
# assigns a trailing-underscore attribute in fit() holds frozen fit-time state
# and therefore has something to leak.
# ---------------------------------------------------------------------------

import inspect  # noqa: E402
import re  # noqa: E402

import philanthropy.preprocessing as _pp  # noqa: E402

_FITTED_ATTR = re.compile(r"^\s*self\.([a-z_][a-z0-9_]*_)\s*=", re.MULTILINE)

# Files whose tests establish the leakage contract for these classes.
_LEAKAGE_TEST_FILES = (
    "tests/test_leakage.py",
    "tests/test_transformer_leakage_guards.py",
    "tests/test_audit_regressions.py",
)

# Classes that hold fit-time state but cannot leak across transform batches,
# with the reason. Keep this list short and argued.
_NO_LEAKAGE_TEST_NEEDED = {
    # Stores only feature_names_in_/n_features_in_ and coerces two columns.
    "CRMCleaner",
    # Pure row-wise arithmetic on the fiscal calendar; no fitted statistic.
    "FiscalYearTransformer",
    # Row-wise thresholds from constructor params only.
    "DischargeToSolicitationWindowTransformer",
    "SolicitationWindowTransformer",
    "PlannedGivingSignalTransformer",
}


def _stateful_preprocessing_classes():
    stateful = {}
    for name in _pp.__all__:
        cls = getattr(_pp, name)
        source = inspect.getsource(inspect.getmodule(cls))
        attrs = {
            a for a in _FITTED_ATTR.findall(source)
            if a not in {"n_features_in_", "feature_names_in_"}
        }
        if attrs:
            stateful[name] = sorted(attrs)
    return stateful


def test_every_stateful_transformer_has_a_leakage_test():
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    corpus = "\n".join(
        (repo_root / f).read_text() for f in _LEAKAGE_TEST_FILES
    )

    stateful = _stateful_preprocessing_classes()
    assert stateful, "source scan found no stateful transformers: regex broke"

    missing = sorted(
        name for name in stateful
        if name not in _NO_LEAKAGE_TEST_NEEDED and name not in corpus
    )
    assert not missing, (
        f"{missing} freeze fit-time state but no leakage test names them. "
        f"Add one to {_LEAKAGE_TEST_FILES[1]}, or justify an entry in "
        f"_NO_LEAKAGE_TEST_NEEDED."
    )


def test_no_leakage_exemption_is_stale():
    """An exempted class that stopped being public (or stopped existing) must
    not sit in the allowlist pretending to be accounted for."""
    unknown = sorted(_NO_LEAKAGE_TEST_NEEDED - set(_pp.__all__))
    assert not unknown, f"_NO_LEAKAGE_TEST_NEEDED names non-public classes: {unknown}"
