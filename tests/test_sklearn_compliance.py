"""
tests/test_sklearn_compliance.py
================================
Formal battery of scikit-learn compliance tests for all estimators.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from philanthropy.models import (
    DonorPropensityModel,
    ShareOfWalletRegressor,
    LapsePredictor,
    MajorGiftClassifier
)
from philanthropy.preprocessing import (
    FiscalYearTransformer,
    WealthScreeningImputer,
    RFMTransformer,
    SolicitationWindowTransformer,
    PlannedGivingSignalTransformer,
    EncounterTransformer,
    GratefulPatientFeaturizer,
    WealthPercentileTransformer
)

@parametrize_with_checks([
    DonorPropensityModel(),
    ShareOfWalletRegressor(),
    # LapsePredictor(), # fit() has non-standard signature (requires gift_dates)
    MajorGiftClassifier(),
    FiscalYearTransformer(),
    WealthScreeningImputer(wealth_cols=["x0"]),
    RFMTransformer(),
    SolicitationWindowTransformer(days_since_last_gift_col="x0"),
    PlannedGivingSignalTransformer(years_active_col="x0", total_gifts_col="x1"),
    GratefulPatientFeaturizer(intensity_score_col="x0", service_line_col="x1"),
    WealthPercentileTransformer(wealth_cols=["x0"])
])
def test_sklearn_compliance(estimator, check):
    """Run standard sklearn checks on compliant estimators."""
    check(estimator)

def test_encounter_transformer_compliance():
    """Manual compliance check for EncounterTransformer."""
    enc_df = pd.DataFrame({"donor_id": [1, 2], "discharge_date": ["2023-01-01", "2023-02-01"]})
    X = pd.DataFrame({"donor_id": [1, 2], "gift_date": ["2023-03-01", "2023-04-01"], "amount": [100, 200]})
    
    t = EncounterTransformer(encounter_df=enc_df)
    t.fit(X)
    out = t.transform(X)
    assert isinstance(out, np.ndarray)
    assert out.shape[1] == 3 # amount, days_since, freq_score
    
def test_grateful_patient_featurizer_compliance():
    """Manual compliance check for GratefulPatientFeaturizer."""
    X = pd.DataFrame({
        "service_line": ["Cardiology", "Oncology"],
        "intensity_score": [10.0, 5.0]
    })
    t = GratefulPatientFeaturizer()
    t.fit(X)
    out = t.transform(X)
    assert isinstance(out, np.ndarray)
    assert out.shape[1] == 2 # intensity, weighted_intensity
    
def test_pipeline_smoke_test():
    """Ensure components can work in a pipeline."""
    X = pd.DataFrame({
        "gift_date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"],
        "gift_amount": [100.0, 200.0, 300.0, 400.0],
        "days_since_last_gift": [10, 20, 30, 40],
        "is_major_donor": [0, 1, 0, 1]
    })
    y = X["is_major_donor"]
    
    pipe = Pipeline([
        ("fy", FiscalYearTransformer(date_col="gift_date")),
        ("dropper", FunctionTransformer(lambda x: x.drop(columns=["gift_date"]))),
        ("model", DonorPropensityModel(n_estimators=5))
    ])
    
    # Pass the full X which includes 'gift_date' required by fy
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == 4
