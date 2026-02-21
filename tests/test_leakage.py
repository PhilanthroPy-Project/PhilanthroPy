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
        t = EncounterTransformer(encounter_df=train_encounters)
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
        t = EncounterTransformer(encounter_df=train_encounters)
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

        t = EncounterTransformer(encounter_df=train_encounters)
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
        t = EncounterTransformer(encounter_df=train_encounters)
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
        t = EncounterTransformer(encounter_df=train_encounters)
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
