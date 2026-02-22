"""
tests/test_leakage.py
======================
Temporal data-leakage prevention tests for PhilanthroPy pipelines.

These tests verify that the ``EncounterTransformer`` and surrounding pipeline
infrastructure **cannot** be exploited to inject future-period target values
into fitted transformer state — a critical correctness guarantee for any
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

import warnings

import numpy as np
import pandas as pd
import pytest

from philanthropy.preprocessing import (
    CRMCleaner,
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
                "2023-08-01",  # new donor 5 — only in future
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
                1_000_000.0,  # extreme future gift — leakage would inject this
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
        the transformer must NOT leak donor 5's future values — it must return NaN.
        """
        t = EncounterTransformer(encounter_df=train_encounters).set_output(transform="pandas")
        t.fit(train_gift_df)
        test_out = t.transform(test_gift_df)

        # Donor 5 is the last row (index 4)
        donor5_days = test_out.iloc[4]["days_since_last_discharge"]
        donor5_freq = test_out.iloc[4]["encounter_frequency_score"]

        assert np.isnan(donor5_days), (
            "Donor 5 (unseen at fit time) must yield NaN for "
            "days_since_last_discharge — returning a real value would "
            "constitute leakage of future data."
        )
        # log1p(0) == 0.0 — unknown donors get frequency score of 0
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
        # Attempt to fit with future gift amounts as y — this must be ignored by EncounterTransformer
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

    def test_crm_cleaner_wealth_imputer_does_not_see_test_set_during_fit(self):
        """
        CRMCleaner with embedded WealthScreeningImputer: the imputer's fit()
        must ONLY use X_train, even if X_test is later piped through transform().
        """
        X_train = pd.DataFrame(
            {
                "gift_date": ["2022-01-01", "2022-06-01", "2022-09-01"],
                "gift_amount": [100.0, 200.0, 300.0],
                "real_estate_value": [np.nan, 500_000.0, np.nan],
            }
        )
        X_test = pd.DataFrame(
            {
                "gift_date": ["2023-01-01"],
                "gift_amount": [999_999.0],       # extreme future gift
                "real_estate_value": [50_000_000.0],  # extreme future wealth
            }
        )

        imputer = WealthScreeningImputer(
            wealth_cols=["real_estate_value"], strategy="median"
        )
        cleaner = CRMCleaner(wealth_imputer=imputer)
        cleaner.fit(X_train)

        # Training fill = median of [500_000] = 500_000
        assert imputer.fill_values_["real_estate_value"] == pytest.approx(500_000.0), (
            "After CRMCleaner.fit(X_train), fill should be 500_000 (training median)."
        )

        # Transform test set
        out_test = cleaner.transform(X_test)

        # Fill value must still be 500_000, not contaminated by 50_000_000
        final_fill = imputer.fill_values_["real_estate_value"]
        assert final_fill == pytest.approx(500_000.0), (
            f"fill_values_['real_estate_value'] changed after transform(X_test): "
            f"got {final_fill}, expected 500_000.0.  "
            "This is a leakage bug — test-set statistics must not alter fitted values."
        )


# ---------------------------------------------------------------------------
# Leakage Test 3: Temporal split — future gift amounts cannot flow backward
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

        # Transform train again after processing test — must be identical
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
        
        df_test = pd.DataFrame({
            "estimated_net_worth": np.random.uniform(0, 1e7, n_test_samples),
            "real_estate_value": np.random.uniform(0, 1e7, n_test_samples)
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

        t = FiscalYearTransformer(fiscal_year_start=7)
        t.fit(df1)
        out1 = t.transform(df1)
        # 2023-01-01 (Jan) is FY23 if FY starts in July (7)
        assert out1.iloc[0]["fiscal_year"] == 2023

        # Transforming df2 should not be influenced by df1
        out2 = t.transform(df2)
        assert out2.iloc[0]["fiscal_year"] == 2024


def test_fiscal_year_transformer_uses_no_future_data():
    """
    FiscalYearTransformer is stateless — fit on training split, transform
    test split with completely disjoint date ranges; both must be correct.
    """
    from philanthropy.preprocessing import FiscalYearTransformer

    train_df = pd.DataFrame({"gift_date": ["2020-07-01", "2020-12-31"]})
    test_df = pd.DataFrame({"gift_date": ["2023-07-01", "2023-12-31"]})

    t = FiscalYearTransformer(fiscal_year_start=7)
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
    Assert that transform() output is unchanged — proving encounter_summary_
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

    # Mutate the original enc_df AFTER fit — should not affect transform()
    enc_df.iloc[0, enc_df.columns.get_loc("discharge_date")] = "2099-01-01"
    post_mutation_output = t.transform(X_test)

    np.testing.assert_array_equal(
        original_output,
        post_mutation_output,
        err_msg=(
            "transform() output changed after mutating encounter_df — "
            "encounter_summary_ must be a snapshot taken at fit() time."
        ),
    )


def test_wealth_imputer_fill_statistics_are_fold_specific_in_cv():
    """
    In 5-fold CV, fill statistics computed in each training fold must NOT all
    be identical — which would indicate the full dataset was used (leakage).
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
        "All fold fill values are identical — possible full-dataset leakage. "
        f"Fill values: {fill_values_per_fold}"
    )


