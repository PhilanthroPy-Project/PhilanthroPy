import numpy as np
import pandas as pd
import pytest
from philanthropy.preprocessing import (
    EncounterRecencyTransformer,
    WealthScreeningImputerKNN,
    ShareOfWalletScorer,
    GratefulPatientFeaturizer,
)
from philanthropy.model_selection import TemporalDonorSplitter
from philanthropy.models import MovesManagementClassifier

def test_encounter_recency_transformer_edge_cases():
    # Test invalid fiscal year start
    with pytest.raises(ValueError, match="fiscal_year_start"):
        t = EncounterRecencyTransformer(fiscal_year_start=13)
        t.fit(pd.DataFrame({"last_encounter_date": ["2023-01-01"]}))

    # Test list of columns
    X = pd.DataFrame({
        "date1": ["2023-01-01", "2023-02-01"],
        "date2": ["2022-01-01", "2022-02-01"]
    })
    t = EncounterRecencyTransformer(date_col=["date1", "date2"])
    out = t.fit_transform(X)
    assert out.shape == (2, 6)
    assert t.get_feature_names_out().shape == (6,)

    # Test timezone localization
    X_tz = pd.DataFrame({"last_encounter_date": ["2023-01-01T12:00:00Z"]})
    t_tz = EncounterRecencyTransformer(timezone="America/New_York")
    out_tz = t_tz.fit_transform(X_tz)
    assert out_tz.shape == (1, 3)
    
    # Test with ndarray transform (forces df creation from feature_names_in_)
    t_arr = EncounterRecencyTransformer(date_col="date1")
    t_arr.fit(X[["date1"]])
    out_arr = t_arr.transform(X[["date1"]].to_numpy())
    assert out_arr.shape == (2, 3)

def test_wealth_imputer_knn_coverage():
    # Invalid strategy
    with pytest.raises(ValueError, match="strategy"):
        imp = WealthScreeningImputerKNN(strategy="invalid")
        imp.fit(pd.DataFrame({"net_worth": [1, 2]}))

    # Invalid n_neighbors
    with pytest.raises(ValueError, match="n_neighbors"):
        imp = WealthScreeningImputerKNN(strategy="knn", n_neighbors=0)
        imp.fit(pd.DataFrame({"net_worth": [1, 2]}))

    # Substring resolution
    X = pd.DataFrame({
        "donor_net_worth": [1e6, np.nan],
        "other": [1, 2]
    })
    imp = WealthScreeningImputerKNN(wealth_cols=None)
    imp.fit(X)
    assert "donor_net_worth" in imp.imputed_cols_
    assert "other" not in imp.imputed_cols_

    # Different strategies
    X2 = pd.DataFrame({
        "real_estate": [100.0, np.nan, 300.0],
        "net_worth": [1.0, 2.0, 3.0]
    })
    for strategy in ["median", "mean", "zero"]:
        imp = WealthScreeningImputerKNN(strategy=strategy, add_indicator=True)
        out = imp.fit_transform(X2)
        assert out.shape[1] == 4 # original 2 + 2 indicators (both qualify by substring)

def test_share_of_wallet_scorer_coverage():
    # Validation
    with pytest.raises(ValueError, match="epsilon"):
        ShareOfWalletScorer(epsilon=-1).fit(np.ones((5, 2)))
    with pytest.raises(ValueError, match="capacity_col_idx"):
        ShareOfWalletScorer(capacity_col_idx=-1).fit(np.ones((5, 2)))

    # Tier labels and outlier clipping
    X = np.array([
        [100, 100],
        [10, 1000000], # outlier
        [50, 100]
    ])
    scorer = ShareOfWalletScorer(capacity_col_idx=0)
    scorer.fit(X)
    labels = scorer.get_tier_labels(X)
    assert len(labels) == 3
    assert "Principal" in labels or "Major" in labels or "Leadership" in labels

def test_temporal_donor_splitter_coverage():
    X = np.zeros((10, 2))
    groups = [2018, 2018, 2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022]
    
    # Gap years
    splitter = TemporalDonorSplitter(n_splits=2, gap_years=1)
    splits = list(splitter.split(X, groups=groups))
    assert len(splits) == 2
    
    # Insufficient years (set gap_years high)
    with pytest.raises(ValueError, match="Not enough fiscal years"):
        list(TemporalDonorSplitter(n_splits=2, gap_years=5).split(X, groups=groups))

    # get_params / repr
    assert "TemporalDonorSplitter" in repr(splitter)
    assert splitter.get_n_splits(groups=groups) == 2

def test_moves_management_classifier_coverage():
    X = np.random.rand(20, 5)
    y = np.random.choice(["IDENTIFY", "QUALIFY", "CULTIVATE"], 20)
    
    clf = MovesManagementClassifier(max_iter=10)
    clf.fit(X, y)
    
    # Predict proba
    prob = clf.predict_proba(X)
    assert prob.shape == (20, 3)
    
    # Priority
    priority = clf.predict_action_priority(X)
    assert "stage" in priority
    assert "confidence" in priority
    assert "portfolio_summary" in priority
    assert len(priority["stage"]) == 20

def test_grateful_patient_featurizer_extra():
    # Test with no data
    enc = pd.DataFrame({"donor_id": [1], "discharge_date": ["2022-01-01"]})
    t = GratefulPatientFeaturizer(encounter_df=enc)
    X = pd.DataFrame({"donor_id": [1], "visit_date": ["2023-01-01"]})
    t.fit(X)
    out = t.transform(X)
    assert out.shape[0] == 1
