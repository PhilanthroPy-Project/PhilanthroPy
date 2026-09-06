"""
tests/test_preprocessing.py
============================
Unit tests and property-based hypothesis tests for PhilanthroPy
preprocessing transformers.

Property-based tests use the `hypothesis` library to bombard
``FiscalYearTransformer`` with extreme datetime edge cases: leap years,
pre-1970 dates, all possible fiscal start months, and timezone-aware datetimes
to check the transformer's stability and correctness
across the space of valid inputs.

Run with:
    pytest tests/test_preprocessing.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

try:
    from sklearn.utils._param_validation import InvalidParameterError
except ImportError:
    InvalidParameterError = ValueError

from philanthropy.preprocessing import (
    CRMCleaner,
    EncounterTransformer,
    FiscalYearTransformer,
    WealthPercentileTransformer,
    WealthScreeningImputer,
)

# Property-based tests for these transformers live in tests/test_properties.py.


# ===========================================================================
# 1. CRMCleaner: standard unit tests
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
        with pytest.raises((ValueError, InvalidParameterError), match="fiscal_year_start"):
            cleaner = CRMCleaner(fiscal_year_start=13)
            cleaner.fit(pd.DataFrame({"gift_date": ["2023-01-01"], "gift_amount": [100.0]}))

    def test_amount_col_coerced_to_float(self):
        df = pd.DataFrame(
            {"gift_date": ["2023-07-01"], "gift_amount": ["$5,000"]}
        )
        cleaner = CRMCleaner().set_output(transform="pandas")
        out = cleaner.fit_transform(df)
        assert out["gift_amount"].dtype == np.float64
        assert out["gift_amount"].iloc[0] == 5000.0

    def test_amount_col_strips_currency_formatting(self):
        # Raiser's Edge NXT / Salesforce NPSP export amounts exactly like this
        # by default. A bare pd.to_numeric NaNs the whole column.
        df = pd.DataFrame({
            "gift_date": ["2024-01-01", "2024-02-01", "2024-03-01"],
            "gift_amount": ["$1,000.00", "$250.50", "($75.00)"],
        })
        cleaner = CRMCleaner().set_output(transform="pandas")
        out = cleaner.fit_transform(df)
        assert list(out["gift_amount"]) == [1000.0, 250.5, -75.0]

    def test_amount_col_unparseable_column_raises(self):
        df = pd.DataFrame({
            "gift_date": ["2024-01-01", "2024-02-01"],
            "gift_amount": ["not a number", "also not a number"],
        })
        cleaner = CRMCleaner()
        with pytest.raises(ValueError, match="could not parse"):
            cleaner.fit_transform(df)

    def test_complex_amount_becomes_nan_with_warning_not_corruption(self):
        # #129: a single np.complex128 cell (Excel/openpyxl formula column
        # round-trip) used to slip through the object-cast retry and corrupt
        # into a plausible finite float — str(3+4j) == "(3+4j)" matched the
        # parenthesised-negative rule, yielding -34.0. Contract now: in a
        # column with other parseable values, the complex cell becomes NaN
        # with a warning, consistent with "values that still don't parse
        # become NaN".
        X_fit = pd.DataFrame({
            "gift_date": ["2023-01-01", "2023-02-01"],
            "gift_amount": [5.0, 10.0],
        })
        cleaner = CRMCleaner().set_output(transform="pandas").fit(X_fit)
        X_bad = pd.DataFrame({
            "gift_date": ["2023-03-01", "2023-04-01"],
            "gift_amount": pd.Series([np.complex128(3 + 4j), 25.0], dtype=object),
        })
        with pytest.warns(UserWarning, match="complex value"):
            out = cleaner.transform(X_bad)
        assert np.isnan(out["gift_amount"].iloc[0])
        assert out["gift_amount"].iloc[1] == 25.0

    def test_complex_whole_column_still_raises(self):
        # A column where nothing parses still raises, complex cells included:
        # masking must not swallow the all-unparseable contract.
        df = pd.DataFrame({
            "gift_date": ["2023-03-01"],
            "gift_amount": pd.Series([np.complex128(3 + 4j)], dtype=object),
        })
        cleaner = CRMCleaner()
        with pytest.raises(ValueError, match="could not parse"):
            cleaner.fit_transform(df)

    def test_date_col_coerced_to_datetime(self):
        df = pd.DataFrame({"gift_date": ["2023-07-01"], "gift_amount": [100.0]})
        cleaner = CRMCleaner().set_output(transform="pandas")
        out = cleaner.fit_transform(df)
        assert pd.api.types.is_datetime64_any_dtype(out["gift_date"])



# ===========================================================================
# 2. WealthScreeningImputer: standard unit tests
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
        with pytest.raises((ValueError, InvalidParameterError), match="strategy"):
            imp = WealthScreeningImputer(strategy="mode")
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
# 3. FiscalYearTransformer: standard unit tests
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
        with pytest.raises((ValueError, InvalidParameterError), match="fiscal_year_start"):
            t = FiscalYearTransformer(fiscal_year_start=0)
            t.fit(pd.DataFrame({"gift_date": ["2023-01-01"]}))


# ===========================================================================
# 5. EncounterTransformer: standard unit tests
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
        """fit() must not use y or external gift data, only encounter_df."""
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


# --------------------------------------------------------------------------- #
# EncounterRecencyTransformer: parameter validation and input shapes
# (moved here from the deleted tests/test_coverage_boost.py)
# --------------------------------------------------------------------------- #
class TestEncounterRecencyTransformerEdgeCases:

    def test_fiscal_year_start_out_of_range_raises(self):
        from philanthropy.preprocessing import EncounterRecencyTransformer

        with pytest.raises(
            ValueError,
            match=r"`fiscal_year_start` must be between 1 and 12, got 13\.",
        ):
            EncounterRecencyTransformer(fiscal_year_start=13).fit(
                pd.DataFrame({"last_encounter_date": ["2023-01-01"]})
            )

    def test_fiscal_year_start_non_integer_raises(self):
        from philanthropy.preprocessing import EncounterRecencyTransformer

        with pytest.raises(
            ValueError, match=r"`fiscal_year_start` must be an integer in \[1, 12\]"
        ):
            EncounterRecencyTransformer(fiscal_year_start="july").fit(
                pd.DataFrame({"last_encounter_date": ["2023-01-01"]})
            )

    def test_multiple_date_columns_emit_three_features_each(self):
        from philanthropy.preprocessing import EncounterRecencyTransformer

        X = pd.DataFrame({
            "date1": ["2023-01-01", "2023-02-01"],
            "date2": ["2022-01-01", "2022-02-01"],
        })
        t = EncounterRecencyTransformer(date_col=["date1", "date2"])
        out = t.fit_transform(X)
        assert out.shape == (2, 6)
        assert t.get_feature_names_out().shape == (6,)

    def test_ndarray_transform_rebuilds_frame_from_feature_names_in(self):
        from philanthropy.preprocessing import EncounterRecencyTransformer

        X = pd.DataFrame({"date1": ["2023-01-01", "2023-02-01"]})
        t = EncounterRecencyTransformer(date_col="date1").fit(X)
        assert t.transform(X.to_numpy()).shape == (2, 3)

    def test_fit_on_ndarray_warns_and_defaults_reference_date_to_today(self):
        from philanthropy.preprocessing import EncounterRecencyTransformer

        X = np.array([["2023-01-01"], ["2023-02-01"]], dtype=object)
        with pytest.warns(UserWarning, match="X is not a DataFrame"):
            t = EncounterRecencyTransformer().fit(X)
        assert t.reference_date_ == pd.Timestamp.today().normalize()

    def test_transform_ndarray_after_ndarray_fit_returns_all_nan(self):
        from philanthropy.preprocessing import EncounterRecencyTransformer

        # Fitted on an ndarray: no column names exist, so transform cannot
        # resolve the date columns and must return an all-NaN block instead
        # of guessing.
        X = np.array([["2023-01-01"], ["2023-02-01"]], dtype=object)
        with pytest.warns(UserWarning, match="X is not a DataFrame"):
            t = EncounterRecencyTransformer().fit(X)

        out = t.transform(X)
        assert out.shape == (2, 3)
        assert np.isnan(out).all()

    def test_fit_ignores_a_configured_column_missing_from_training_frame(self):
        from philanthropy.preprocessing import EncounterRecencyTransformer

        # "missing_col" is declared but absent from the frame; fit must skip
        # it (no crash, no warning) and infer the reference date from the one
        # column that is present.
        X = pd.DataFrame({"date1": ["2023-01-01", "2023-06-30"]})
        t = EncounterRecencyTransformer(
            date_col=["date1", "missing_col"], fiscal_year_start=7
        ).fit(X)
        assert t.reference_date_ == pd.Timestamp("2023-06-30")

    def test_transform_fills_nan_for_a_column_missing_at_inference_time(self):
        from philanthropy.preprocessing import EncounterRecencyTransformer

        X = pd.DataFrame({"date1": ["2023-01-01", "2023-06-01"]})
        t = EncounterRecencyTransformer(date_col=["date1", "date2"]).fit(X)

        inference_frame = pd.DataFrame({"date1": ["2023-07-01"]})
        with pytest.warns(UserWarning, match="'date2' not found"):
            out = t.transform(inference_frame)

        assert out.shape == (1, 6)
        # date1 features are computed normally; 2023-07-01 is *after* the
        # reference date frozen at fit time (2023-06-01), so the day count is
        # the documented negative "future date" value.
        assert out[0][0] == -30.0
        # date2 features follow the missing-date contract exactly:
        # NaN days, 0.0 flag ("missing dates -> 0.0"), NaN fiscal year.
        np.testing.assert_allclose(out[0][3:], [np.nan, 0.0, np.nan])

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_days_since_day_resolution_stays_finite_across_centuries(self):
        from philanthropy.preprocessing._encounter_recency import (
            EncounterRecencyTransformer,
        )

        # A span wider than int64 nanoseconds (~292 years) overflows the
        # primary differencing path; the static day-resolution fallback must
        # return finite day counts, with NaT mapping to NaN like the primary
        # path. Both endpoints sit safely inside the ns-representable window
        # (~1677-09-21 .. ~2262-04-11) so construction succeeds on every
        # pandas, while their ~584-year distance still forces day resolution.
        ref_ts = pd.Timestamp("2262-01-01")
        first = pd.Timestamp("1678-01-01")
        dates = pd.Series(pd.to_datetime([first, None]))

        days = EncounterRecencyTransformer._days_since_day_resolution(ref_ts, dates)

        # Expected value via plain-Python dates: a pandas Timestamp
        # subtraction would overflow int64 nanoseconds on pandas 2.x, which
        # is exactly the regime this fallback serves.
        expected_days = (
            ref_ts.to_pydatetime().date() - first.to_pydatetime().date()
        ).days
        assert days[0] == float(expected_days)
        assert np.isnan(days[1])

        # The same holds when either side is timezone-aware: the helper strips
        # both to naive UTC before differencing.
        aware_dates = pd.Series(
            pd.to_datetime([first], utc=True).tz_convert("America/Chicago")
        )
        days_aware = EncounterRecencyTransformer._days_since_day_resolution(
            pd.Timestamp(ref_ts, tz="UTC"), aware_dates
        )
        assert days_aware[0] == float(expected_days)

    # NOTE on the `except (OverflowError, OutOfBoundsTimedelta)` guard in
    # _compute_recency_features: we could not construct any public input that
    # reaches it, on pandas 2.x or 3.x. On pandas>=3 timestamps are second-
    # resolution so nanosecond overflow cannot occur; on pandas 2.x,
    # out-of-ns-range dates cannot be parsed by to_datetime (coerced to NaT),
    # and numpy-built datetime64[ns] series subtract with silent int64
    # *wraparound* instead of raising. If a future pandas raises again, a
    # regression test should assert exact day-resolution values here.


class TestGetPandasOutputFallbacks:
    """The defensive fallbacks in `_get_pandas_output` (issue #62, group 1).

    The modern `sklearn.utils._set_output._get_output_config` lookup is
    imported inside the function, so monkeypatching the sklearn attribute
    reaches it; the legacy attribute check is exercised by setting
    `_sklearn_output_config` directly.
    """

    @staticmethod
    def _estimator_with(config):
        from philanthropy.preprocessing._transformers import CRMCleaner

        est = CRMCleaner()
        est._sklearn_output_config = config
        return est

    def test_string_shaped_legacy_config_returns_true(self, monkeypatch):
        import sklearn.utils._set_output as so
        from philanthropy.preprocessing._transformers import _get_pandas_output

        def raise_attr_error(*args, **kwargs):
            raise AttributeError("removed in this sklearn")

        monkeypatch.setattr(so, "_get_output_config", raise_attr_error)

        assert (
            _get_pandas_output(self._estimator_with({"transform": "pandas"}))
            is True
        )

    def test_dict_shaped_legacy_config_returns_true(self, monkeypatch):
        import sklearn.utils._set_output as so
        from philanthropy.preprocessing._transformers import _get_pandas_output

        monkeypatch.setattr(so, "_get_output_config", lambda *a, **k: {})

        est = self._estimator_with({"transform": {"dense": "pandas"}})
        assert _get_pandas_output(est) is True

    def test_non_dict_legacy_config_returns_false(self, monkeypatch):
        import sklearn.utils._set_output as so
        from philanthropy.preprocessing._transformers import _get_pandas_output

        monkeypatch.setattr(so, "_get_output_config", lambda *a, **k: {})

        est = self._estimator_with("not-a-dict")
        assert _get_pandas_output(est) is False


@pytest.mark.usefixtures()
class TestValidationRetryBranches:
    """The object-cast retry in fit/transform (issue #62, group 2).

    A DataFrame carrying a datetime64 column makes the first `validate_data`
    pass raise DTypePromotionError; casting to object dtype recovers, so both
    transformers must fit and transform such input instead of raising.
    """

    def test_crm_cleaner_fit_retries_datetime_columns(self):
        X = pd.DataFrame(
            {"gift_date": pd.to_datetime(["2023-01-01"]), "gift_amount": [5.0]}
        )
        out = CRMCleaner().fit(X).transform(X)
        assert np.asarray(out).shape == (1, 2)

    def test_fiscal_year_transformer_fit_and_transform_retry_datetime_columns(self):
        X = pd.DataFrame({"gift_date": pd.to_datetime(["2023-01-15"])})
        t = FiscalYearTransformer()

        t.fit(X)  # must not raise
        out = np.asarray(t.transform(X))
        assert out.shape == (1, 2)
        # Default fiscal_year_start=7: January precedes July, so FY == 2023.
        assert out[0][0] == 2023.0

    def test_fiscal_year_transformer_missing_date_col_yields_nan_features(self):
        # Fitting on an unnamed ndarray leaves no feature_names_in_, so at
        # transform time the frame's columns are positional integers and the
        # configured date_col cannot be located: fiscal features degrade to
        # NaN instead of raising.
        X = pd.DataFrame({"gift_date": ["2023-01-15"]})
        t = FiscalYearTransformer(fiscal_year_start=7).fit(X.to_numpy())

        out = np.asarray(t.transform(X.to_numpy()))

        assert out.shape == (1, 2)
        assert np.isnan(out).all()

    def test_fiscal_year_transformer_retries_mixed_dtype_frames(self):
        # A lone datetime64 column validates cleanly; mixing it with a float
        # column is what triggers the DTypePromotionError that the object-cast
        # retry exists for.
        X = pd.DataFrame(
            {"gift_date": pd.to_datetime(["2023-01-15"]), "extra": [1.5]}
        )
        out = np.asarray(FiscalYearTransformer().fit(X).transform(X))
        assert out[0].tolist() == [2023.0, 3.0]

    def test_complex_column_is_re_raised_not_retried(self):
        # The retry must not swallow the explicit complex-data rejection:
        # a complex column fails validation with exactly that message and is
        # re-raised instead of being cast to object and retried.
        X = pd.DataFrame({"gift_date": [np.complex128(1 + 2j)]})

        with pytest.raises(ValueError, match="Complex data not supported"):
            FiscalYearTransformer().fit(X)

        with pytest.raises(ValueError, match="Complex data not supported"):
            CRMCleaner().fit(X)


class TestCoerceCurrencyNumericShortCircuit:
    def test_numeric_amount_column_passes_through_coercion(self):
        # Already-numeric amounts take the early return in
        # _coerce_currency_to_float rather than the string-cleaning path.
        X = pd.DataFrame({"gift_date": ["2023-01-01"], "gift_amount": [1250.50]})
        out = CRMCleaner().set_output(transform="pandas").fit_transform(X)
        assert out.loc[0, "gift_amount"] == 1250.50


class TestValidationReRaiseGuards:
    def test_numeric_series_short_circuits_string_cleaning(self):
        from philanthropy.preprocessing._transformers import (
            _coerce_currency_to_float,
        )

        col = pd.Series([1250.50, 0.0])
        out = _coerce_currency_to_float(col)
        assert out.dtype == "float64"
        assert out.tolist() == [1250.50, 0.0]

    def test_crm_transform_re_raises_complex_column(self):
        # A complex ndarray at transform time fails validation with the
        # explicit complex-data message, which the retry guard re-raises
        # instead of casting to object and retrying.
        X_fit = pd.DataFrame({"gift_date": ["2023-01-01"], "gift_amount": [5.0]})
        t = CRMCleaner().fit(X_fit)

        X_bad = np.array([[1 + 2j]], dtype=np.complex128)
        with pytest.raises(ValueError, match="Complex data not supported"):
            t.transform(X_bad)

    def test_fiscal_year_transform_re_raises_complex_column(self):
        t = FiscalYearTransformer().fit(pd.DataFrame({"gift_date": ["2023-01-15"]}))

        X_bad = pd.DataFrame({"gift_date": [np.complex128(1 + 2j)]})
        with pytest.raises(ValueError, match="Complex data not supported"):
            t.transform(X_bad)

    def test_crm_cleaner_on_unnamed_ndarrays_skips_named_columns(self):
        # Fit and transform on plain ndarrays: no feature names exist, so the
        # configured date/amount column lookups both miss and the cleaner
        # still returns the right shape.
        X = np.array([["2023-01-01", "1250.50"]], dtype=object)
        out = np.asarray(CRMCleaner().fit(X).transform(X))
        assert out.shape == (1, 2)


def test_wealth_percentile_feature_names_match_for_frame_and_array_input():
    df = pd.DataFrame({"net_worth": [1.0, 2.0, 3.0], "other": [1.0, 2.0, 3.0]})
    a = WealthPercentileTransformer().fit(df)
    b = WealthPercentileTransformer().fit(df.to_numpy())
    assert list(a.get_feature_names_out()) == ["net_worth", "other", "net_worth_pct_rank"]
    assert list(b.get_feature_names_out()) == ["x0", "x1"]

