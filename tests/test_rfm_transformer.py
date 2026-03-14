"""
tests/test_rfm_transformer.py
Test suite for RFMTransformer.
"""

import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from philanthropy.preprocessing import RFMTransformer


@pytest.fixture
def sample_transactions():
    return pd.DataFrame({
        'donor_id': [1, 1, 2, 2, 2, 3],
        'gift_date': ['2023-01-01', '2023-06-01', '2022-01-01', '2022-12-01', '2023-05-01', '2021-01-01'],
        'gift_amount': [100.0, 200.0, 50.0, 50.0, 100.0, 1000.0]
    })


def test_output_columns_present(sample_transactions):
    t = RFMTransformer(reference_date='2024-01-01')
    rfm = t.fit_transform(sample_transactions)
    assert 'recency' in rfm.columns
    assert 'frequency' in rfm.columns
    assert 'monetary' in rfm.columns
    assert 'donor_id' in rfm.columns


def test_recency_non_negative(sample_transactions):
    t = RFMTransformer(reference_date='2024-01-01')
    rfm = t.fit_transform(sample_transactions)
    assert (rfm['recency'] >= 0).all()


def test_frequency_positive_integer(sample_transactions):
    t = RFMTransformer(reference_date='2024-01-01')
    rfm = t.fit_transform(sample_transactions)
    assert (rfm['frequency'] >= 1).all()
    assert rfm['frequency'].dtype in ['int64', 'int32'] or rfm['frequency'].dtype.name in ('int64', 'int32')


def test_monetary_non_negative(sample_transactions):
    t = RFMTransformer(reference_date='2024-01-01')
    rfm = t.fit_transform(sample_transactions)
    assert (rfm['monetary'] >= 0).all()


def test_reference_date_shifts_recency_correctly(sample_transactions):
    t1 = RFMTransformer(reference_date='2024-01-01')
    t2 = RFMTransformer(reference_date='2025-01-01')
    rfm1 = t1.fit_transform(sample_transactions)
    rfm2 = t2.fit_transform(sample_transactions)
    assert (rfm2['recency'] > rfm1['recency']).all()


def test_fit_train_transform_test_no_leakage(sample_transactions):
    train = sample_transactions[sample_transactions['donor_id'].isin([1, 2])]
    test = pd.concat([
        sample_transactions[sample_transactions['donor_id'] == 3],
        sample_transactions[sample_transactions['donor_id'] == 1].iloc[:1]
    ], ignore_index=True)
    t = RFMTransformer(reference_date='2023-12-31')
    t.fit(train)
    out = t.transform(test)
    assert set(out.columns) == {'donor_id', 'recency', 'frequency', 'monetary'}


def test_not_fitted_raises(sample_transactions):
    t = RFMTransformer()
    with pytest.raises(NotFittedError):
        t.transform(sample_transactions)


def test_pipeline_compatibility(sample_transactions):
    pipe = Pipeline([("rfm", RFMTransformer(reference_date='2024-01-01'))])
    out = pipe.fit_transform(sample_transactions)
    assert 'recency' in out.columns
    assert 'frequency' in out.columns
    assert len(out) == 3


def test_clone_compatibility(sample_transactions):
    t = RFMTransformer()
    t.fit(sample_transactions)
    cloned = clone(t)
    assert not hasattr(cloned, 'feature_names_in_') or cloned.feature_names_in_ is not None


def test_empty_dataframe_returns_empty_with_schema():
    empty = pd.DataFrame(columns=['donor_id', 'gift_date', 'gift_amount'])
    t = RFMTransformer(reference_date='2024-01-01')
    rfm = t.fit_transform(empty)
    assert len(rfm) == 0
    assert set(rfm.columns) == {'donor_id', 'recency', 'frequency', 'monetary'}


def test_single_row_dataframe(sample_transactions):
    single = sample_transactions[sample_transactions['donor_id'] == 3]
    t = RFMTransformer(reference_date='2024-01-01')
    rfm = t.fit_transform(single)
    assert len(rfm) == 1
    assert rfm.iloc[0]['frequency'] == 1
    assert rfm.iloc[0]['monetary'] == 1000.0


def test_all_same_date_recency_zero(sample_transactions):
    same_date = pd.DataFrame({
        'donor_id': [1, 1, 2],
        'gift_date': ['2024-06-15', '2024-06-15', '2024-06-15'],
        'gift_amount': [100.0, 200.0, 50.0]
    })
    t = RFMTransformer(reference_date='2024-06-15')
    rfm = t.fit_transform(same_date)
    assert (rfm['recency'] == 0).all()


def test_agg_func_mean(sample_transactions):
    t = RFMTransformer(reference_date='2024-01-01', agg_func='mean')
    rfm = t.fit_transform(sample_transactions)
    d1 = rfm[rfm['donor_id'] == 1].iloc[0]
    assert d1['monetary'] == 150.0


def test_agg_func_sum(sample_transactions):
    t = RFMTransformer(reference_date='2024-01-01', agg_func='sum')
    rfm = t.fit_transform(sample_transactions)
    d1 = rfm[rfm['donor_id'] == 1].iloc[0]
    assert d1['monetary'] == 300.0


def test_fit_returns_self(sample_transactions):
    t = RFMTransformer()
    result = t.fit(sample_transactions)
    assert result is t


def test_get_feature_names_out(sample_transactions):
    t = RFMTransformer()
    t.fit(sample_transactions)
    names = t.get_feature_names_out()
    assert 'recency' in names
    assert 'frequency' in names
    assert 'monetary' in names


def test_n_features_in_after_fit(sample_transactions):
    t = RFMTransformer()
    t.fit(sample_transactions)
    assert hasattr(t, 'n_features_in_')
    assert t.n_features_in_ == 3


def test_recency_days_calculation(sample_transactions):
    t = RFMTransformer(reference_date='2024-01-01')
    rfm = t.fit_transform(sample_transactions)
    d1 = rfm[rfm['donor_id'] == 1].iloc[0]
    expected = (pd.Timestamp('2024-01-01') - pd.Timestamp('2023-06-01')).days
    assert d1['recency'] == expected


def test_reference_date_none_uses_max_date(sample_transactions):
    t = RFMTransformer()
    rfm = t.fit_transform(sample_transactions)
    d1 = rfm[rfm['donor_id'] == 1].iloc[0]
    assert d1['recency'] == 0


def test_frequency_per_donor(sample_transactions):
    t = RFMTransformer(reference_date='2024-01-01')
    rfm = t.fit_transform(sample_transactions)
    assert rfm[rfm['donor_id'] == 1]['frequency'].iloc[0] == 2
    assert rfm[rfm['donor_id'] == 2]['frequency'].iloc[0] == 3
    assert rfm[rfm['donor_id'] == 3]['frequency'].iloc[0] == 1
