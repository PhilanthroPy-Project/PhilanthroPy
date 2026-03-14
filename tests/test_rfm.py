import pandas as pd
import pytest
from philanthropy.preprocessing import RFMTransformer

def test_rfm_transformer_basic():
    # Create sample transaction data
    data = pd.DataFrame({
        'donor_id': [1, 1, 2, 2, 2, 3],
        'gift_date': ['2023-01-01', '2023-06-01', '2022-01-01', '2022-12-01', '2023-05-01', '2021-01-01'],
        'gift_amount': [100, 200, 50, 50, 100, 1000]
    })
    
    reference_date = '2024-01-01'
    transformer = RFMTransformer(reference_date=reference_date, agg_func='sum')
    
    rfm = transformer.fit_transform(data)
    
    # Check shape
    assert set(rfm.columns) == {'donor_id', 'recency', 'frequency', 'monetary'}
    assert len(rfm) == 3
    
    # Check values for donor_id 1
    d1 = rfm[rfm['donor_id'] == 1].iloc[0]
    assert d1['frequency'] == 2
    assert d1['monetary'] == 300
    expected_recency_1 = (pd.to_datetime('2024-01-01') - pd.to_datetime('2023-06-01')).days
    assert d1['recency'] == expected_recency_1
    
    # Check values for donor_id 2
    d2 = rfm[rfm['donor_id'] == 2].iloc[0]
    assert d2['frequency'] == 3
    assert d2['monetary'] == 200
    expected_recency_2 = (pd.to_datetime('2024-01-01') - pd.to_datetime('2023-05-01')).days
    assert d2['recency'] == expected_recency_2

def test_rfm_transformer_agg_mean():
    data = pd.DataFrame({
        'donor_id': [1, 1],
        'gift_date': ['2023-01-01', '2023-06-01'],
        'gift_amount': [100, 200]
    })
    
    transformer = RFMTransformer(agg_func='mean')
    rfm = transformer.fit_transform(data)
    
    d1 = rfm[rfm['donor_id'] == 1].iloc[0]
    assert d1['monetary'] == 150.0

def test_rfm_transformer_no_reference_date():
    data = pd.DataFrame({
        'donor_id': [1, 2],
        'gift_date': ['2023-01-01', '2023-06-01'],
        'gift_amount': [100, 200]
    })
    
    transformer = RFMTransformer()
    rfm = transformer.fit_transform(data)
    
    d1 = rfm[rfm['donor_id'] == 1].iloc[0]
    d2 = rfm[rfm['donor_id'] == 2].iloc[0]
    
    assert d2['recency'] == 0
    assert d1['recency'] == (pd.to_datetime('2023-06-01') - pd.to_datetime('2023-01-01')).days

def test_rfm_transformer_validation():
    transformer = RFMTransformer()
    data = pd.DataFrame({
        'donor_id': [1, 2],
        'gift_date': ['2023-01-01', '2023-06-01']
    })
    
    with pytest.raises(ValueError, match="X must contain columns:"):
        transformer.fit(data)

    with pytest.raises(TypeError, match="X must be a pandas DataFrame"):
        transformer.transform([1, 2, 3])
