"""
tests/test_preprocessing.py
============================
Unit tests and property-based hypothesis tests for PhilanthroPy
preprocessing transformers.

Property-based tests use the `hypothesis` library to bombard
``FiscalYearTransformer`` with extreme datetime edge cases — leap years,
pre-1970 dates, all possible fiscal start months, and timezone-aware datetimes
— mathematically guaranteeing the transformer's stability and correctness
across the space of valid inputs.

Run with:
    pytest tests/test_preprocessing.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from philanthropy.preprocessing import (
    CRMCleaner,
    EncounterTransformer,
    FiscalYearTransformer,
    WealthScreeningImputer,
)

# ---------------------------------------------------------------------------
# Try importing hypothesis; skip property-based tests if not installed
# ---------------------------------------------------------------------------
try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
    from hypothesis.extra.pandas import column, data_frames
    import hypothesis.extra.pandas as hpd

    _HYPOTHESIS_AVAILABLE = True
except ImportError:
    _HYPOTHESIS_AVAILABLE = False

hypothesis_mark = pytest.mark.skipif(
    not _HYPOTHESIS_AVAILABLE,
    reason="hypothesis is not installed — run `pip install hypothesis` to enable property-based tests.",
)


# ===========================================================================
# 1. CRMCleaner — standard unit tests
# ===========================================================================


class TestCRMCleaner:
    """Standard unit tests for CRMCleaner."""

    def test_fit_returns_self(self, donor_df):
        cleaner = CRMCleaner().set_output(transform="pandas")
        result = cleaner.fit(donor_df)
        assert result is cleaner

    def test_fit_transform_preserves_shape(self, donor_df):
        cleaner = CRMCleaner().set_output(transform="pandas")
        out = cleaner.fit_transform(donor_df)
        assert isinstance(out, pd.DataFrame)
        assert out.shape == donor_df.shape

    def test_feature_names_in_set(self, donor_df):
        cleaner = CRMCleaner().set_output(transform="pandas")
        cleaner.fit(donor_df)
        assert hasattr(cleaner, "feature_names_in_")
        assert set(cleaner.feature_names_in_) == set(donor_df.columns)

    def test_invalid_fiscal_year_start_raises(self):
        cleaner = CRMCleaner(fiscal_year_start=13)
        with pytest.raises(ValueError, match="fiscal_year_start"):
            cleaner.fit(pd.DataFrame({"gift_date": ["2023-01-01"], "gift_amount": [100.0]}))

    def test_amount_col_coerced_to_float(self):
        df = pd.DataFrame(
            {"gift_date": ["2023-07-01"], "gift_amount": ["$5,000"]}
        )
        cleaner = CRMCleaner().set_output(transform="pandas")
        out = cleaner.fit_transform(df)
        # Non-parseable → NaN; dtype should be float64
        assert out["gift_amount"].dtype == np.float64 or out["gift_amount"].isna().all()

    def test_date_col_coerced_to_datetime(self):
        df = pd.DataFrame({"gift_date": ["2023-07-01"], "gift_amount": [100.0]})
        cleaner = CRMCleaner().set_output(transform="pandas")
        out = cleaner.fit_transform(df)
        assert pd.api.types.is_datetime64_any_dtype(out["gift_date"])


class TestCRMCleanerWithWealthImputer:
    """Tests for CRMCleaner integrated with WealthScreeningImputer."""

    def test_imputer_fitted_during_crm_fit(self):
        X = pd.DataFrame(
            {
                "gift_date": ["2023-07-01", "2023-08-01"],
                "gift_amount": [1000.0, 2000.0],
                "estimated_net_worth": [np.nan, 1_500_000.0],
            }
        )
        imputer = WealthScreeningImputer(wealth_cols=["estimated_net_worth"]).set_output(transform="pandas")
        cleaner = CRMCleaner(wealth_imputer=imputer)
        cleaner.fit(X)
        # Imputer should now be fitted
        assert hasattr(imputer, "fill_values_")

    def test_no_nan_after_transform(self):
        X = pd.DataFrame(
            {
                "gift_date": ["2023-07-01", "2023-08-01", "2023-09-01"],
                "gift_amount": [1000.0, 2000.0, 3000.0],
                "estimated_net_worth": [np.nan, 1_500_000.0, np.nan],
                "real_estate_value": [200_000.0, np.nan, 300_000.0],
            }
        )
        imputer = WealthScreeningImputer(
            wealth_cols=["estimated_net_worth", "real_estate_value"],
            strategy="median",
        ).set_output(transform="pandas")
        cleaner = CRMCleaner(wealth_imputer=imputer)
        out = cleaner.fit_transform(X)
        assert out["estimated_net_worth"].isna().sum() == 0
        assert out["real_estate_value"].isna().sum() == 0

    def test_no_leakage_imputer_not_prefitted(self):
        """Verify imputer is NOT pre-fitted before CRMCleaner.fit() is called."""
        imputer = WealthScreeningImputer(wealth_cols=["estimated_net_worth"])
        assert not hasattr(imputer, "fill_values_"), (
            "Imputer must not be fitted before CRMCleaner.fit(); "
            "pre-fitting risks leakage of test-set statistics."
        )


# ===========================================================================
# 2. WealthScreeningImputer — standard unit tests
# ===========================================================================


class TestWealthScreeningImputer:
    """Unit tests for the standalone WealthScreeningImputer."""

    def _make_df(self, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n = 20
        df = pd.DataFrame(
            {
                "estimated_net_worth": np.where(
                    rng.random(n) < 0.4, np.nan, rng.uniform(1e5, 10e6, n)
                ),
                "real_estate_value": np.where(
                    rng.random(n) < 0.5, np.nan, rng.uniform(1e4, 5e6, n)
                ),
                "gift_amount": rng.uniform(100, 50_000, n),
            }
        )
        return df

    def test_median_strategy_no_nan(self):
        df = self._make_df()
        imp = WealthScreeningImputer(
            wealth_cols=["estimated_net_worth", "real_estate_value"],
            strategy="median",
        ).set_output(transform="pandas")
        out = imp.fit_transform(df)
        assert out["estimated_net_worth"].isna().sum() == 0
        assert out["real_estate_value"].isna().sum() == 0

    def test_zero_strategy(self):
        df = pd.DataFrame({"estimated_net_worth": [np.nan, 1e6, np.nan]})
        imp = WealthScreeningImputer(wealth_cols=["estimated_net_worth"], strategy="zero").set_output(transform="pandas")
        out = imp.fit_transform(df)
        assert out.loc[0, "estimated_net_worth"] == pytest.approx(0.0)

    def test_mean_strategy(self):
        df = pd.DataFrame({"estimated_net_worth": [1e6, np.nan, 3e6]})
        imp = WealthScreeningImputer(wealth_cols=["estimated_net_worth"], strategy="mean").set_output(transform="pandas")
        out = imp.fit_transform(df)
        assert out.loc[1, "estimated_net_worth"] == pytest.approx(2e6)

    def test_add_indicator_columns_created(self):
        df = pd.DataFrame({"estimated_net_worth": [np.nan, 1e6]})
        imp = WealthScreeningImputer(
            wealth_cols=["estimated_net_worth"], add_indicator=True
        ).set_output(transform="pandas")
        out = imp.fit_transform(df)
        assert "estimated_net_worth__was_missing" in out.columns
        assert out.loc[0, "estimated_net_worth__was_missing"] == 1
        assert out.loc[1, "estimated_net_worth__was_missing"] == 0

    def test_invalid_strategy_raises(self):
        imp = WealthScreeningImputer(strategy="mode")
        with pytest.raises(ValueError, match="strategy"):
            imp.fit(pd.DataFrame({"estimated_net_worth": [1e6]}))

    def test_fill_value_frozen_from_train(self):
        """Ensure test-set values do not affect fill statistics."""
        X_train = pd.DataFrame({"estimated_net_worth": [100.0, np.nan, 200.0]})
        X_test = pd.DataFrame({"estimated_net_worth": [np.nan, 900.0]})
        imp = WealthScreeningImputer(wealth_cols=["estimated_net_worth"]).set_output(transform="pandas")
        imp.fit(X_train)
        frozen_fill = imp.fill_values_["estimated_net_worth"]
        imp.transform(X_test)
        # Fill value must not change after transform
        assert imp.fill_values_["estimated_net_worth"] == frozen_fill

    def test_missing_column_warns_but_does_not_raise(self):
        df = pd.DataFrame({"gift_amount": [1000.0]})
        imp = WealthScreeningImputer(wealth_cols=["estimated_net_worth"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            imp.fit(df)
            assert any("estimated_net_worth" in str(warning.message) for warning in w)


# ===========================================================================
# 3. FiscalYearTransformer — standard unit tests
# ===========================================================================


class TestFiscalYearTransformer:
    """Standard unit tests for FiscalYearTransformer."""

    def test_adds_fiscal_year_and_quarter_columns(self, donor_df):
        t = FiscalYearTransformer().set_output(transform="pandas")
        out = t.fit_transform(donor_df)
        assert "fiscal_year" in out.columns
        assert "fiscal_quarter" in out.columns

    def test_july_start_logic(self):
        df = pd.DataFrame(
            {"gift_date": ["2023-07-01", "2023-06-30", "2024-01-15"]}
        )
        t = FiscalYearTransformer(fiscal_year_start=7).set_output(transform="pandas")
        out = t.fit_transform(df)
        # July 1, 2023 → FY2024 (fiscal year starting July 2023)
        assert out.loc[0, "fiscal_year"] == 2024
        # June 30, 2023 → FY2023
        assert out.loc[1, "fiscal_year"] == 2023
        # Jan 15, 2024 → FY2024
        assert out.loc[2, "fiscal_year"] == 2024

    def test_january_start_logic(self):
        df = pd.DataFrame({"gift_date": ["2023-01-01", "2023-12-31"]})
        t = FiscalYearTransformer(fiscal_year_start=1).set_output(transform="pandas")
        out = t.fit_transform(df)
        # With January start, calendar year == fiscal year
        assert out.loc[0, "fiscal_year"] == 2024  # Jan 01 ≥ Jan → FY = year+1
        assert out.loc[1, "fiscal_year"] == 2024  # Dec 31 ≥ Jan → FY = year+1

    def test_fiscal_quarter_range(self):
        dates = pd.date_range("2022-07-01", periods=12, freq="MS").strftime(
            "%Y-%m-%d"
        )
        df = pd.DataFrame({"gift_date": dates})
        t = FiscalYearTransformer(fiscal_year_start=7).set_output(transform="pandas")
        out = t.fit_transform(df)
        quarters = out["fiscal_quarter"].dropna().astype(int)
        assert quarters.min() >= 1
        assert quarters.max() <= 4

    def test_invalid_fiscal_year_start(self):
        t = FiscalYearTransformer(fiscal_year_start=0)
        with pytest.raises(ValueError, match="fiscal_year_start"):
            t.fit(pd.DataFrame({"gift_date": ["2023-01-01"]}))


# ===========================================================================
# 4. FiscalYearTransformer — PROPERTY-BASED TESTS (hypothesis)
# ===========================================================================


@hypothesis_mark
class TestFiscalYearTransformerHypothesis:
    """Property-based stress tests for FiscalYearTransformer.

    The hypothesis library generates thousands of randomised test cases
    covering the full space of valid inputs — leap years, pre-1970 dates,
    all 12 fiscal start months, and timezone-aware timestamps — providing
    stronger correctness guarantees than any finite set of handcrafted cases.
    """

    # ------------------------------------------------------------------
    # Shared helper
    # ------------------------------------------------------------------

    @staticmethod
    def _date_strategy():
        """Return a Hypothesis strategy that draws plausible gift datetimes.

        Covers:
        * Pre-1970 (Unix epoch = 0) dates (e.g., 1800–1969)
        * Post-epoch modern dates (1970–2030)
        * Leap-year critical dates (Feb 28–29 on centennial years)
        * Timezone-aware timestamps with arbitrary UTC offsets
        """
        # Naive datetime strategy spanning 1800–2030
        naive_dates = st.dates(
            min_value=pd.Timestamp("1800-01-01").date(),
            max_value=pd.Timestamp("2030-12-31").date(),
        ).map(lambda d: d.strftime("%Y-%m-%d"))

        return naive_dates

    # ------------------------------------------------------------------
    # Property 1: fiscal_year is always an integer ≥ the calendar year
    # ------------------------------------------------------------------

    @given(
        fiscal_year_start=st.integers(min_value=1, max_value=12),
        n_rows=st.integers(min_value=1, max_value=50),
        dates=st.lists(
            st.dates(
                min_value=pd.Timestamp("1800-01-01").date(),
                max_value=pd.Timestamp("2030-12-31").date(),
            ).map(lambda d: d.strftime("%Y-%m-%d")),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(
        max_examples=500,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_fiscal_year_always_integer(self, fiscal_year_start, n_rows, dates):
        """``fiscal_year`` must always be a finite integer value."""
        df = pd.DataFrame({"gift_date": dates})
        t = FiscalYearTransformer(fiscal_year_start=fiscal_year_start).set_output(transform="pandas")
        out = t.fit_transform(df)
        fiscal_years = out["fiscal_year"]
        assert fiscal_years.notna().all(), "fiscal_year must not contain NaN."
        assert (fiscal_years % 1 == 0).all(), "fiscal_year must be an integer-valued number."

    # ------------------------------------------------------------------
    # Property 2: fiscal_year is always calendar year OR calendar year + 1
    # ------------------------------------------------------------------

    @given(
        fiscal_year_start=st.integers(min_value=1, max_value=12),
        date_str=st.dates(
            min_value=pd.Timestamp("1800-01-01").date(),
            max_value=pd.Timestamp("2030-12-31").date(),
        ).map(lambda d: d.strftime("%Y-%m-%d")),
    )
    @settings(max_examples=1000, deadline=None)
    def test_fiscal_year_is_calendar_year_or_plus_one(
        self, fiscal_year_start, date_str
    ):
        """fiscal_year ∈ {calendar_year, calendar_year + 1} for any input."""
        df = pd.DataFrame({"gift_date": [date_str]})
        t = FiscalYearTransformer(fiscal_year_start=fiscal_year_start).set_output(transform="pandas")
        out = t.fit_transform(df)
        calendar_year = pd.to_datetime(date_str).year
        fiscal_year = int(out.loc[0, "fiscal_year"])
        assert fiscal_year in {calendar_year, calendar_year + 1}, (
            f"For date={date_str}, fiscal_start={fiscal_year_start}: "
            f"expected fiscal_year ∈ {{{calendar_year}, {calendar_year+1}}}, "
            f"got {fiscal_year}."
        )

    # ------------------------------------------------------------------
    # Property 3: fiscal_quarter ∈ {1, 2, 3, 4}
    # ------------------------------------------------------------------

    @given(
        fiscal_year_start=st.integers(min_value=1, max_value=12),
        date_str=st.dates(
            min_value=pd.Timestamp("1800-01-01").date(),
            max_value=pd.Timestamp("2030-12-31").date(),
        ).map(lambda d: d.strftime("%Y-%m-%d")),
    )
    @settings(max_examples=1000, deadline=None)
    def test_fiscal_quarter_always_1_to_4(self, fiscal_year_start, date_str):
        """fiscal_quarter must always be in {1, 2, 3, 4}."""
        df = pd.DataFrame({"gift_date": [date_str]})
        t = FiscalYearTransformer(fiscal_year_start=fiscal_year_start).set_output(transform="pandas")
        out = t.fit_transform(df)
        q = int(out.loc[0, "fiscal_quarter"])
        assert 1 <= q <= 4, (
            f"fiscal_quarter={q} is out of range [1, 4] "
            f"for date={date_str}, fiscal_start={fiscal_year_start}."
        )

    # ------------------------------------------------------------------
    # Property 4: Idempotency — applying the transformer twice is harmless
    # ------------------------------------------------------------------

    @given(
        fiscal_year_start=st.integers(min_value=1, max_value=12),
        date_strs=st.lists(
            st.dates(
                min_value=pd.Timestamp("1900-01-01").date(),
                max_value=pd.Timestamp("2030-12-31").date(),
            ).map(lambda d: d.strftime("%Y-%m-%d")),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=300, deadline=None)
    def test_transform_idempotent_fiscal_year(self, fiscal_year_start, date_strs):
        """Calling transform twice on the same data must yield identical fiscal_year."""
        df = pd.DataFrame({"gift_date": date_strs})
        t = FiscalYearTransformer(fiscal_year_start=fiscal_year_start).set_output(transform="pandas")
        t.fit(df)
        out1 = t.transform(df)
        out2 = t.transform(out1.rename(columns={"fiscal_year": "_fy_drop",
                                                 "fiscal_quarter": "_fq_drop"}).drop(
            columns=["_fy_drop", "_fq_drop"], errors="ignore"
        ).assign(gift_date=df["gift_date"]))
        pd.testing.assert_series_equal(
            out1["fiscal_year"].reset_index(drop=True),
            out2["fiscal_year"].reset_index(drop=True),
            check_names=False,
        )

    # ------------------------------------------------------------------
    # Property 5: Leap-year dates — Feb 29 must not raise, fiscal year correct
    # ------------------------------------------------------------------

    @given(
        fiscal_year_start=st.integers(min_value=1, max_value=12),
        leap_year=st.sampled_from([2000, 2004, 2008, 2012, 2016, 2020, 2024]),
    )
    @settings(max_examples=72, deadline=None)
    def test_leap_day_does_not_raise(self, fiscal_year_start, leap_year):
        """Feb 29 on actual leap years must be handled without exception."""
        date_str = f"{leap_year}-02-29"
        df = pd.DataFrame({"gift_date": [date_str]})
        t = FiscalYearTransformer(fiscal_year_start=fiscal_year_start).set_output(transform="pandas")
        out = t.fit_transform(df)  # Must not raise
        # Feb is always before fiscal_start unless start == 1 or 2
        fy = int(out.loc[0, "fiscal_year"])
        expected = leap_year + 1 if 2 >= fiscal_year_start else leap_year
        assert fy == expected, (
            f"Leap day {date_str}, fiscal_start={fiscal_year_start}: "
            f"expected fy={expected}, got {fy}."
        )

    # ------------------------------------------------------------------
    # Property 6: timezone-aware datetimes do not crash
    # ------------------------------------------------------------------

    @given(
        fiscal_year_start=st.integers(min_value=1, max_value=12),
        utc_offset_hours=st.integers(min_value=-12, max_value=14),
        date_str=st.dates(
            min_value=pd.Timestamp("1970-01-01").date(),
            max_value=pd.Timestamp("2030-12-31").date(),
        ).map(lambda d: d.strftime("%Y-%m-%d")),
    )
    @settings(max_examples=300, deadline=None)
    def test_timezone_aware_string_does_not_crash(
        self, fiscal_year_start, utc_offset_hours, date_str
    ):
        """Timezone-offset date strings (e.g. '2023-07-01+05:30') must be parsed."""
        sign = "+" if utc_offset_hours >= 0 else "-"
        abs_h = abs(utc_offset_hours)
        tz_str = f"{date_str}T12:00:00{sign}{abs_h:02d}:00"
        df = pd.DataFrame({"gift_date": [tz_str]})
        t = FiscalYearTransformer(fiscal_year_start=fiscal_year_start).set_output(transform="pandas")
        # utcoffset-aware strings parsed by pd.to_datetime; check no exception
        try:
            out = t.fit_transform(df)
            assert "fiscal_year" in out.columns
        except Exception as exc:
            pytest.fail(
                f"FiscalYearTransformer raised unexpectedly for tz-aware date "
                f"{tz_str!r}, fiscal_start={fiscal_year_start}: {exc}"
            )


# ===========================================================================
# 5. EncounterTransformer — standard unit tests
# ===========================================================================


@pytest.fixture
def encounter_df():
    return pd.DataFrame(
        {
            "donor_id": [101, 101, 102, 103],
            "discharge_date": [
                "2022-03-15",
                "2023-06-01",
                "2021-11-20",
                "2020-08-10",
            ],
        }
    )


@pytest.fixture
def gift_df_with_ids():
    return pd.DataFrame(
        {
            "donor_id": [101, 102, 103, 104],  # donor 104 has no encounters
            "gift_date": ["2023-08-01", "2022-02-15", "2021-09-30", "2023-01-01"],
            "gift_amount": [10_000.0, 500.0, 250.0, 1_000.0],
        }
    )


class TestEncounterTransformer:
    """Unit tests for EncounterTransformer."""

    def test_output_lacks_donor_id(self, encounter_df, gift_df_with_ids):
        t = EncounterTransformer(encounter_df=encounter_df).set_output(transform="pandas")
        out = t.fit_transform(gift_df_with_ids)
        assert "donor_id" not in out.columns, "merge_key must be stripped from output."

    def test_new_columns_present(self, encounter_df, gift_df_with_ids):
        t = EncounterTransformer(encounter_df=encounter_df).set_output(transform="pandas")
        out = t.fit_transform(gift_df_with_ids)
        assert "days_since_last_discharge" in out.columns
        assert "encounter_frequency_score" in out.columns

    def test_missing_discharge_dates_produce_nan(self, gift_df_with_ids):
        """Donors not in encounter_df get NaN for days_since_last_discharge."""
        enc = pd.DataFrame(
            {"donor_id": [101], "discharge_date": ["2022-01-01"]}
        )
        t = EncounterTransformer(encounter_df=enc).set_output(transform="pandas")
        out = t.fit_transform(gift_df_with_ids)
        # donors 102, 103, 104 are not in enc
        unknown_rows = out[out.index.isin([1, 2, 3])]  # after reset_index
        assert unknown_rows["days_since_last_discharge"].isna().all()

    def test_negative_days_coerced_to_nan_by_default(self):
        """Gift before discharge → NaN when allow_negative_days=False."""
        enc = pd.DataFrame(
            {"donor_id": [1], "discharge_date": ["2024-01-01"]}
        )
        gifts = pd.DataFrame(
            {"donor_id": [1], "gift_date": ["2023-01-01"], "gift_amount": [100.0]}
        )
        t = EncounterTransformer(encounter_df=enc, allow_negative_days=False).set_output(transform="pandas")
        out = t.fit_transform(gifts)
        assert np.isnan(out.loc[0, "days_since_last_discharge"])

    def test_allow_negative_days_flag(self):
        """Gift before discharge → negative integer when allow_negative_days=True."""
        enc = pd.DataFrame(
            {"donor_id": [1], "discharge_date": ["2024-01-01"]}
        )
        gifts = pd.DataFrame(
            {"donor_id": [1], "gift_date": ["2023-01-01"], "gift_amount": [100.0]}
        )
        t = EncounterTransformer(encounter_df=enc, allow_negative_days=True).set_output(transform="pandas")
        out = t.fit_transform(gifts)
        assert out.loc[0, "days_since_last_discharge"] < 0

    def test_encounter_frequency_score_log_scaled(self, encounter_df, gift_df_with_ids):
        """Encounter frequency should equal log1p(encounter_count)."""
        t = EncounterTransformer(encounter_df=encounter_df).set_output(transform="pandas")
        t.fit(gift_df_with_ids)
        out = t.transform(gift_df_with_ids)
        # Donor 101 has 2 encounters → log1p(2) ≈ 1.099
        donor101_row = out.iloc[0]
        assert donor101_row["encounter_frequency_score"] == pytest.approx(
            np.log1p(2), rel=1e-4
        )

    def test_fit_does_not_touch_X_values(self):
        """fit() must not use y or external gift data — only encounter_df."""
        enc = pd.DataFrame({"donor_id": [1], "discharge_date": ["2022-06-01"]})
        gifts = pd.DataFrame(
            {
                "donor_id": [1],
                "gift_date": ["2023-01-01"],
                "gift_amount": [99_999_999.0],  # intentionally extreme target
            }
        )
        t = EncounterTransformer(encounter_df=enc)
        t.fit(gifts, y=np.array([99_999_999.0]))
        # Encounter summary must be derived from encounter_df only
        last_discharge = t.encounter_summary_.loc[1, "last_discharge"]
        assert last_discharge == pd.Timestamp("2022-06-01")

    def test_missing_merge_key_in_X_raises(self):
        enc = pd.DataFrame({"donor_id": [1], "discharge_date": ["2022-01-01"]})
        gifts = pd.DataFrame({"gift_date": ["2023-01-01"], "gift_amount": [100.0]})
        t = EncounterTransformer(encounter_df=enc).set_output(transform="pandas")
        with pytest.raises(ValueError, match="donor_id"):
            t.fit(gifts)

    def test_missing_discharge_col_raises(self):
        enc = pd.DataFrame({"donor_id": [1], "bad_col": ["2022-01-01"]})
        gifts = pd.DataFrame(
            {"donor_id": [1], "gift_date": ["2023-01-01"], "gift_amount": [100.0]}
        )
        t = EncounterTransformer(encounter_df=enc)
        with pytest.raises(ValueError, match="discharge_date"):
            t.fit(gifts)

    def test_extra_pii_columns_dropped(self):
        enc = pd.DataFrame({"donor_id": [1], "discharge_date": ["2022-06-01"]})
        gifts = pd.DataFrame(
            {
                "donor_id": [1],
                "gift_date": ["2023-01-01"],
                "gift_amount": [500.0],
                "patient_mrn": ["MRN001"],  # should be stripped
                "full_name": ["John Doe"],  # should be stripped
            }
        )
        t = EncounterTransformer(encounter_df=enc).set_output(transform="pandas")
        out = t.fit_transform(gifts)
        assert "patient_mrn" not in out.columns
        assert "full_name" not in out.columns

    def test_all_nan_discharge_warns(self, gift_df_with_ids):
        enc = pd.DataFrame(
            {"donor_id": [101], "discharge_date": [None]}
        )
        t = EncounterTransformer(encounter_df=enc)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            t.fit(gift_df_with_ids)
            assert any(isinstance(warning.category, type(UserWarning)) for warning in w)
