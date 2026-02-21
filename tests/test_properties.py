import numpy as np
import pandas as pd
from hypothesis import given, settings, assume, strategies as st
from hypothesis.extra.pandas import data_frames, column, range_indexes
from philanthropy.preprocessing import (
    FiscalYearTransformer,
    WealthScreeningImputer,
    WealthPercentileTransformer,
)

fiscal_start_months = st.integers(min_value=1, max_value=12)

valid_dates = st.dates(
    min_value=pd.Timestamp("1900-01-01").date(),
    max_value=pd.Timestamp("2099-12-31").date(),
)

gift_date_series = st.lists(valid_dates, min_size=1, max_size=500).map(
    lambda dates: pd.DataFrame({"gift_date": pd.to_datetime(dates)})
)

wealth_dataframe = data_frames(
    columns=[
        column("estimated_net_worth", elements=st.one_of(
            st.floats(min_value=0, max_value=1e9, allow_nan=True, allow_infinity=False),
            st.just(float("nan")),
        )),
        column("real_estate_value", elements=st.one_of(
            st.floats(min_value=0, max_value=5e8, allow_nan=True, allow_infinity=False),
            st.just(float("nan")),
        )),
        column("gift_amount", elements=st.floats(min_value=1.0, max_value=1e7)),
    ],
    index=range_indexes(min_size=2, max_size=300),
)


class TestFiscalYearTransformerProperties:
    @settings(max_examples=500)
    @given(df=gift_date_series, fy_start=fiscal_start_months)
    def test_fiscal_year_is_integer(self, df, fy_start):
        transformer = FiscalYearTransformer(fiscal_year_start=fy_start)
        out = transformer.fit_transform(df)
        assert pd.api.types.is_numeric_dtype(out["fiscal_year"])
        assert out["fiscal_year"].isna().sum() == 0

    @settings(max_examples=500)
    @given(df=gift_date_series, fy_start=fiscal_start_months)
    def test_fiscal_quarter_range(self, df, fy_start):
        transformer = FiscalYearTransformer(fiscal_year_start=fy_start)
        out = transformer.fit_transform(df)
        assert out["fiscal_quarter"].isin([1, 2, 3, 4]).all()

    @settings(max_examples=500)
    @given(fy_start=fiscal_start_months)
    def test_boundary_month_is_quarter_one(self, fy_start):
        df = pd.DataFrame({"gift_date": [pd.Timestamp(f"2023-{fy_start:02d}-01")]})
        transformer = FiscalYearTransformer(fiscal_year_start=fy_start)
        out = transformer.fit_transform(df)
        assert out["fiscal_quarter"].iloc[0] == 1

    @settings(max_examples=500)
    @given(df=gift_date_series, fy_start=fiscal_start_months)
    def test_idempotent_transform(self, df, fy_start):
        t1 = FiscalYearTransformer(fiscal_year_start=fy_start)
        t2 = FiscalYearTransformer(fiscal_year_start=fy_start)
        out1 = t1.fit_transform(df)
        t2.fit(df)
        out2 = t2.transform(df)
        pd.testing.assert_frame_equal(out1, out2)

    @settings(max_examples=500)
    @given(fy_start=fiscal_start_months)
    def test_leap_year_feb29_does_not_crash(self, fy_start):
        df = pd.DataFrame({"gift_date": [pd.Timestamp("2000-02-29")]})
        transformer = FiscalYearTransformer(fiscal_year_start=fy_start)
        out = transformer.fit_transform(df)
        assert len(out) == 1

    @settings(max_examples=500)
    @given(fy_start=fiscal_start_months)
    def test_pre_unix_epoch_dates(self, fy_start):
        df = pd.DataFrame({"gift_date": [pd.Timestamp("1899-12-31"), pd.Timestamp("1923-07-04")]})
        transformer = FiscalYearTransformer(fiscal_year_start=fy_start)
        out = transformer.fit_transform(df)
        assert (out["fiscal_year"] > 0).all()


class TestWealthScreeningImputerProperties:
    @settings(max_examples=300)
    @given(df=wealth_dataframe, strategy=st.sampled_from(["median", "mean", "zero"]))
    def test_no_nulls_in_imputed_columns_after_transform(self, df, strategy):
        transformer = WealthScreeningImputer(strategy=strategy, add_indicator=False)
        out = transformer.fit_transform(df)
        assert out["estimated_net_worth"].isna().sum() == 0
        assert out["real_estate_value"].isna().sum() == 0

    @settings(max_examples=300)
    @given(df=wealth_dataframe, strategy=st.sampled_from(["median", "mean", "zero"]))
    def test_fill_values_frozen_after_fit(self, df, strategy):
        transformer = WealthScreeningImputer(strategy=strategy)
        transformer.fit(df)
        initial_fills = transformer.fill_values_.copy()
        
        # Mutate input
        df2 = df.copy()
        df2["estimated_net_worth"] = 999999.0
        transformer.transform(df2)
        
        assert transformer.fill_values_ == initial_fills

    @settings(max_examples=300)
    @given(df=wealth_dataframe, strategy=st.sampled_from(["median", "mean", "zero"]))
    def test_indicator_columns_are_binary(self, df, strategy):
        transformer = WealthScreeningImputer(strategy=strategy, add_indicator=True)
        out = transformer.fit_transform(df)
        for col in ["estimated_net_worth", "real_estate_value"]:
            ind_col = f"{col}__was_missing"
            if ind_col in out.columns:
                assert out[ind_col].isin([0, 1]).all()

    @settings(max_examples=300)
    @given(df_train=wealth_dataframe, df_test=wealth_dataframe, strategy=st.sampled_from(["median", "mean", "zero"]))
    def test_train_test_fill_value_invariance(self, df_train, df_test, strategy):
        transformer = WealthScreeningImputer(strategy=strategy, add_indicator=False)
        transformer.fit(df_train)
        
        df_test.loc[len(df_test)] = {"estimated_net_worth": np.nan, "real_estate_value": np.nan, "gift_amount": 100}
        out = transformer.transform(df_test)
        
        assert out.iloc[-1]["estimated_net_worth"] == transformer.fill_values_["estimated_net_worth"]
        assert out.iloc[-1]["real_estate_value"] == transformer.fill_values_["real_estate_value"]

    @settings(max_examples=300)
    @given(df=wealth_dataframe, strategy=st.sampled_from(["median", "mean", "zero"]))
    def test_all_strategies_produce_finite_output(self, df, strategy):
        transformer = WealthScreeningImputer(strategy=strategy, add_indicator=False)
        out = transformer.fit_transform(df)
        assert np.isfinite(out["estimated_net_worth"]).all()
        assert np.isfinite(out["real_estate_value"]).all()


class TestWealthPercentileTransformerProperties:
    @settings(max_examples=200)
    @given(df=wealth_dataframe)
    def test_output_percentiles_in_range(self, df):
        transformer = WealthPercentileTransformer()
        out = transformer.fit_transform(df)
        for col in ["estimated_net_worth_pct_rank", "real_estate_value_pct_rank"]:
            if col in out.columns:
                non_nan = out[col].dropna()
                if len(non_nan) > 0:
                    assert (non_nan >= 0.0).all() and (non_nan <= 100.0).all()

    @settings(max_examples=200)
    @given(df=wealth_dataframe)
    def test_nan_input_produces_nan_output(self, df):
        transformer = WealthPercentileTransformer()
        out = transformer.fit_transform(df)
        for col in ["estimated_net_worth", "real_estate_value"]:
            if col in df.columns and f"{col}_pct_rank" in out.columns:
                nan_mask = df[col].isna()
                assert out.loc[nan_mask, f"{col}_pct_rank"].isna().all()

    @settings(max_examples=200)
    @given(df=wealth_dataframe)
    def test_rank_monotone(self, df):
        transformer = WealthPercentileTransformer()
        out = transformer.fit_transform(df)
        for col in ["estimated_net_worth", "real_estate_value"]:
            if col in df.columns and f"{col}_pct_rank" in out.columns:
                valid_dt = df[col].dropna()
                valid_out = out[f"{col}_pct_rank"].dropna()
                
                if len(valid_dt) >= 2:
                    idx_a, idx_b = valid_dt.index[0], valid_dt.index[1]
                    val_a, val_b = valid_dt.loc[idx_a], valid_dt.loc[idx_b]
                    rank_a, rank_b = valid_out.loc[idx_a], valid_out.loc[idx_b]
                    
                    if val_a > val_b:
                        assert rank_a >= rank_b
                    elif val_a < val_b:
                        assert rank_a <= rank_b
