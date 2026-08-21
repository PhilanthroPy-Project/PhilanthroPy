"""tests/test_properties.py

The single home for property-based (Hypothesis) tests over the transformers.

This file absorbed tests/test_preprocessing_properties.py,
tests/test_transformers_property.py and
tests/test_preprocessing.py::TestFiscalYearTransformerHypothesis, which between
them asserted the same fiscal-year and wealth-imputer properties three times at
300–1000 examples each. Each property is stated once here; example counts are
sized to the search space (12 fiscal months x a date range needs hundreds, not
thousands).
"""

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays, floating_dtypes
from hypothesis.extra.pandas import column, data_frames, range_indexes

from philanthropy.preprocessing import (
    DischargeToSolicitationWindowTransformer,
    EncounterRecencyTransformer,
    FiscalYearTransformer,
    WealthPercentileTransformer,
    WealthScreeningImputer,
)

fiscal_start_months = st.integers(min_value=1, max_value=12)

valid_dates = st.dates(
    min_value=pd.Timestamp("1800-01-01").date(),
    max_value=pd.Timestamp("2099-12-31").date(),
)

gift_date_series = st.lists(valid_dates, min_size=1, max_size=200).map(
    lambda dates: pd.DataFrame({"gift_date": pd.to_datetime(dates)})
)

wealth_dataframe = data_frames(
    columns=[
        column("estimated_net_worth", elements=st.one_of(
            st.floats(min_value=0, max_value=1e9, allow_nan=False, allow_infinity=False),
            st.just(float("nan")),
        )),
        column("real_estate_value", elements=st.one_of(
            st.floats(min_value=0, max_value=5e8, allow_nan=False, allow_infinity=False),
            st.just(float("nan")),
        )),
        column("gift_amount", elements=st.floats(min_value=1.0, max_value=1e7)),
    ],
    index=range_indexes(min_size=2, max_size=300),
)

_SETTINGS = settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)


# ---------------------------------------------------------------------------
# FiscalYearTransformer
# ---------------------------------------------------------------------------

class TestFiscalYearTransformerProperties:

    @_SETTINGS
    @given(df=gift_date_series, fy_start=fiscal_start_months)
    def test_row_count_preserved_and_fiscal_year_is_a_finite_integer(self, df, fy_start):
        out = (
            FiscalYearTransformer(fiscal_year_start=fy_start)
            .set_output(transform="pandas")
            .fit_transform(df)
        )
        assert len(out) == len(df)
        assert pd.api.types.is_numeric_dtype(out["fiscal_year"])
        assert out["fiscal_year"].notna().all()
        assert (out["fiscal_year"] % 1 == 0).all()

    @_SETTINGS
    @given(date=valid_dates, fy_start=fiscal_start_months)
    def test_fiscal_year_is_the_calendar_year_or_the_next_one(self, date, fy_start):
        out = (
            FiscalYearTransformer(fiscal_year_start=fy_start)
            .set_output(transform="pandas")
            .fit_transform(pd.DataFrame({"gift_date": [date.strftime("%Y-%m-%d")]}))
        )
        assert int(out.loc[0, "fiscal_year"]) in {date.year, date.year + 1}

    @_SETTINGS
    @given(df=gift_date_series, fy_start=fiscal_start_months)
    def test_fiscal_quarter_is_always_one_to_four(self, df, fy_start):
        out = (
            FiscalYearTransformer(fiscal_year_start=fy_start)
            .set_output(transform="pandas")
            .fit_transform(df)
        )
        assert out["fiscal_quarter"].isin([1, 2, 3, 4]).all()

    @_SETTINGS
    @given(fy_start=fiscal_start_months)
    def test_the_first_month_of_the_fiscal_year_is_quarter_one(self, fy_start):
        df = pd.DataFrame({"gift_date": [pd.Timestamp(f"2023-{fy_start:02d}-01")]})
        out = (
            FiscalYearTransformer(fiscal_year_start=fy_start)
            .set_output(transform="pandas")
            .fit_transform(df)
        )
        assert out["fiscal_quarter"].iloc[0] == 1

    @_SETTINGS
    @given(df=gift_date_series, fy_start=fiscal_start_months)
    def test_transform_is_idempotent(self, df, fy_start):
        t1 = FiscalYearTransformer(fiscal_year_start=fy_start).set_output(transform="pandas")
        t2 = FiscalYearTransformer(fiscal_year_start=fy_start).set_output(transform="pandas")
        out1 = t1.fit_transform(df)
        out2 = t2.fit(df).transform(df)
        pd.testing.assert_frame_equal(out1, out2)

    @settings(max_examples=84, deadline=None)
    @given(
        fy_start=fiscal_start_months,
        leap_year=st.sampled_from([2000, 2004, 2008, 2012, 2016, 2020, 2024]),
    )
    def test_leap_day_maps_to_the_right_fiscal_year(self, fy_start, leap_year):
        out = (
            FiscalYearTransformer(fiscal_year_start=fy_start)
            .set_output(transform="pandas")
            .fit_transform(pd.DataFrame({"gift_date": [f"{leap_year}-02-29"]}))
        )
        expected = leap_year + 1 if 2 >= fy_start else leap_year
        assert int(out.loc[0, "fiscal_year"]) == expected

    @_SETTINGS
    @given(fy_start=fiscal_start_months)
    def test_pre_unix_epoch_dates_produce_positive_fiscal_years(self, fy_start):
        df = pd.DataFrame({
            "gift_date": [pd.Timestamp("1899-12-31"), pd.Timestamp("1923-07-04")]
        })
        out = (
            FiscalYearTransformer(fiscal_year_start=fy_start)
            .set_output(transform="pandas")
            .fit_transform(df)
        )
        assert (out["fiscal_year"] > 0).all()

    @_SETTINGS
    @given(
        fy_start=fiscal_start_months,
        utc_offset_hours=st.integers(min_value=-12, max_value=14),
        date=st.dates(
            min_value=pd.Timestamp("1970-01-01").date(),
            max_value=pd.Timestamp("2030-12-31").date(),
        ),
    )
    def test_timezone_offset_strings_parse(self, fy_start, utc_offset_hours, date):
        sign = "+" if utc_offset_hours >= 0 else "-"
        tz_str = f"{date:%Y-%m-%d}T12:00:00{sign}{abs(utc_offset_hours):02d}:00"
        out = (
            FiscalYearTransformer(fiscal_year_start=fy_start)
            .set_output(transform="pandas")
            .fit_transform(pd.DataFrame({"gift_date": [tz_str]}))
        )
        assert out["fiscal_year"].notna().all()


# ---------------------------------------------------------------------------
# WealthScreeningImputer
# ---------------------------------------------------------------------------

class TestWealthScreeningImputerProperties:

    @_SETTINGS
    @given(df=wealth_dataframe, strategy=st.sampled_from(["median", "mean", "zero"]))
    def test_no_nulls_and_finite_output_in_imputed_columns(self, df, strategy):
        out = (
            WealthScreeningImputer(strategy=strategy, add_indicator=False)
            .set_output(transform="pandas")
            .fit_transform(df)
        )
        assert len(out) == len(df)
        for col in ("estimated_net_worth", "real_estate_value"):
            assert out[col].isna().sum() == 0
            assert np.isfinite(out[col]).all()

    @_SETTINGS
    @given(df=wealth_dataframe, strategy=st.sampled_from(["median", "mean", "zero"]))
    def test_fill_values_frozen_after_fit(self, df, strategy):
        transformer = WealthScreeningImputer(strategy=strategy).fit(df)
        initial_fills = transformer.fill_values_.copy()

        df2 = df.copy()
        df2["estimated_net_worth"] = 999999.0
        transformer.transform(df2)

        assert transformer.fill_values_ == initial_fills

    @_SETTINGS
    @given(df=wealth_dataframe, strategy=st.sampled_from(["median", "mean", "zero"]))
    def test_indicator_columns_are_binary(self, df, strategy):
        out = (
            WealthScreeningImputer(strategy=strategy, add_indicator=True)
            .set_output(transform="pandas")
            .fit_transform(df)
        )
        for col in ("estimated_net_worth", "real_estate_value"):
            ind_col = f"{col}__was_missing"
            if ind_col in out.columns:
                assert out[ind_col].isin([0, 1]).all()

    @_SETTINGS
    @given(
        df_train=wealth_dataframe,
        df_test=wealth_dataframe,
        strategy=st.sampled_from(["median", "mean", "zero"]),
    )
    def test_a_missing_row_is_filled_with_the_frozen_train_statistic(
        self, df_train, df_test, strategy
    ):
        transformer = (
            WealthScreeningImputer(strategy=strategy, add_indicator=False)
            .set_output(transform="pandas")
            .fit(df_train)
        )
        df_test = df_test.copy()
        df_test.loc[len(df_test)] = {
            "estimated_net_worth": np.nan,
            "real_estate_value": np.nan,
            "gift_amount": 100,
        }
        out = transformer.transform(df_test)
        for col in ("estimated_net_worth", "real_estate_value"):
            assert out.iloc[-1][col] == transformer.fill_values_[col]

    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        deadline=None,
    )
    @given(
        arr=arrays(
            # float16 has no nanmedian on numpy 1.x and overflows on modest
            # values. No explicit `elements`: from_dtype() already generates NaN
            # and inf and respects the chosen float width.
            dtype=floating_dtypes(sizes=(32, 64)),
            shape=st.tuples(
                st.integers(min_value=1, max_value=100),
                st.integers(min_value=1, max_value=10),
            ),
        )
    )
    def test_extreme_float_inputs_never_crash(self, arr):
        try:
            result = WealthScreeningImputer(strategy="median").fit_transform(arr)
        except (ValueError, ZeroDivisionError):
            return  # controlled rejection is acceptable; a crash is not
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == arr.shape[0]


# ---------------------------------------------------------------------------
# WealthPercentileTransformer
# ---------------------------------------------------------------------------

class TestWealthPercentileTransformerProperties:

    @_SETTINGS
    @given(df=wealth_dataframe)
    def test_percentiles_are_within_zero_and_one_hundred(self, df):
        out = WealthPercentileTransformer().set_output(transform="pandas").fit_transform(df)
        for col in ("estimated_net_worth_pct_rank", "real_estate_value_pct_rank"):
            if col in out.columns:
                non_nan = out[col].dropna()
                if len(non_nan):
                    assert (non_nan >= 0.0).all() and (non_nan <= 100.0).all()

    @_SETTINGS
    @given(df=wealth_dataframe)
    def test_nan_in_produces_nan_out(self, df):
        out = WealthPercentileTransformer().set_output(transform="pandas").fit_transform(df)
        for col in ("estimated_net_worth", "real_estate_value"):
            if col in df.columns and f"{col}_pct_rank" in out.columns:
                nan_mask = df[col].isna()
                assert out.loc[nan_mask, f"{col}_pct_rank"].isna().all()

    @_SETTINGS
    @given(df=wealth_dataframe)
    def test_rank_is_monotone_in_the_underlying_value(self, df):
        out = WealthPercentileTransformer().set_output(transform="pandas").fit_transform(df)
        for col in ("estimated_net_worth", "real_estate_value"):
            if col not in df.columns or f"{col}_pct_rank" not in out.columns:
                continue
            valid = df[col].dropna()
            ranks = out[f"{col}_pct_rank"].dropna()
            if len(valid) < 2:
                continue
            idx_a, idx_b = valid.index[0], valid.index[1]
            if valid.loc[idx_a] > valid.loc[idx_b]:
                assert ranks.loc[idx_a] >= ranks.loc[idx_b]
            elif valid.loc[idx_a] < valid.loc[idx_b]:
                assert ranks.loc[idx_a] <= ranks.loc[idx_b]


# ---------------------------------------------------------------------------
# DischargeToSolicitationWindowTransformer
# ---------------------------------------------------------------------------

@_SETTINGS
@given(
    days=st.lists(
        st.one_of(st.integers(min_value=0, max_value=5000), st.none()),
        min_size=1,
        max_size=50,
    )
)
def test_solicitation_window_output_shape_and_ranges(days):
    parsed = [float(d) if d is not None else np.nan for d in days]
    df = pd.DataFrame({"days_since_last_discharge": parsed})
    t = DischargeToSolicitationWindowTransformer(
        days_since_discharge_col="days_since_last_discharge",
        min_days_post_discharge=100,
        max_days_post_discharge=200,
    ).fit(df)
    out = t.transform(df)

    assert isinstance(out, np.ndarray)
    assert out.shape == (len(days), 2)
    assert set(np.unique(out[:, 0])).issubset({0.0, 1.0})
    # The score is NaN exactly where the input day is missing, and in [0, 1]
    # everywhere else, so "no discharge on record" never reads as a real 0.0.
    missing = np.isnan(np.asarray(parsed, dtype=float))
    np.testing.assert_array_equal(np.isnan(out[:, 1]), missing)
    scored = out[~missing, 1]
    assert ((scored >= 0.0) & (scored <= 1.0)).all()
    # in_window == 1 exactly when the value sits inside the closed window.
    inside = np.array(
        [(not np.isnan(d)) and 100.0 <= d <= 200.0 for d in parsed], dtype=float
    )
    np.testing.assert_array_equal(out[:, 0], inside)


# ---------------------------------------------------------------------------
# EncounterRecencyTransformer
# ---------------------------------------------------------------------------

@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(
    dates=st.lists(
        st.one_of(st.datetimes(), st.none(), st.just(np.nan)),
        min_size=1,
        max_size=10,
    )
)
def test_encounter_recency_never_crashes_on_ragged_dates(dates):
    df = pd.DataFrame({"last_encounter_date": dates})
    try:
        result = EncounterRecencyTransformer().fit_transform(df)
    except (ValueError, TypeError):
        return  # controlled rejection is acceptable
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(df), 3)
