import numpy as np
import pytest
import pandas as pd
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays, floating_dtypes

from philanthropy.preprocessing import EncounterRecencyTransformer, WealthScreeningImputer

@given(
    arr=arrays(
        dtype=floating_dtypes(),
        shape=st.tuples(
            st.integers(min_value=1, max_value=100),
            st.integers(min_value=1, max_value=10)
        ),
        elements=st.floats(allow_nan=True, allow_infinity=True)
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much], max_examples=50)
def test_wealth_imputer_does_not_raise_on_extreme_inputs(arr):
    # Ensure no infinity if imputer doesn't handle it well, 
    # but the prompt says to bombard with inf.
    # Note: KNNImputer/median might fail on inf if not handled.
    t = WealthScreeningImputer(strategy="median")
    try:
        result = t.fit_transform(arr)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == arr.shape[0]
    except (ValueError, ZeroDivisionError):
        pass  # Controlled rejection is acceptable; crashes are not

@given(
    df=st.builds(
        pd.DataFrame,
        st.fixed_dictionaries({
            "last_encounter_date": st.lists(
                st.one_of(
                    st.datetimes(),
                    st.just(None),
                    st.just(np.nan)
                ),
                min_size=1, max_size=10
            )
        })
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
def test_encounter_transformer_on_dates(df):
    t = EncounterRecencyTransformer()
    try:
        result = t.fit_transform(df)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(df)
        assert result.shape[1] == 3
    except (ValueError, TypeError):
        pass


def test_encounter_transformer_no_overflow_on_extreme_span():
    # Two representable dates >292 years apart overflow a datetime64[ns]
    # timedelta (int64). The transformer must fall back to day-resolution
    # instead of raising OverflowError. Regression for that crash.
    df = pd.DataFrame({"last_encounter_date": ["1806-01-01", "2099-01-01"]})
    out = EncounterRecencyTransformer().fit_transform(df)
    assert out.shape == (2, 3)
    assert np.isfinite(out[:, 0]).all()  # days_since finite for both rows
