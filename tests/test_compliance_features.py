"""
tests/test_compliance_features.py

Covers the compliance-hardening behaviours: configurable/broadened PII column
dropping and the negative-days warning on EncounterTransformer, and the
overridable service-line weights on GratefulPatientFeaturizer.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from philanthropy.preprocessing import EncounterTransformer, GratefulPatientFeaturizer


def _encounters():
    return pd.DataFrame(
        {
            "donor_id": [1, 1, 2],
            "discharge_date": ["2022-01-01", "2023-06-15", "2022-09-30"],
        }
    )


def _gifts_with_pii():
    return pd.DataFrame(
        {
            "donor_id": [1, 2, 3],
            "gift_date": ["2023-08-01", "2023-01-01", "2023-05-01"],
            "gift_amount": [10000.0, 750.0, 250.0],
            "date_of_birth": ["1950-01-01", "1960-02-02", "1970-03-03"],
            "PatientMRN": ["m1", "m2", "m3"],
        }
    )


def _fit_transform_pandas(transformer, X):
    transformer.set_output(transform="pandas")
    return transformer.fit_transform(X)


def test_broadened_pii_patterns_drop_birth_and_patient_columns():
    t = EncounterTransformer(encounter_df=_encounters(), merge_key="donor_id")
    out = _fit_transform_pandas(t, _gifts_with_pii())
    # date_of_birth (via "birth") and PatientMRN (via "patient"/"mrn") must go.
    assert "date_of_birth" not in out.columns
    assert "PatientMRN" not in out.columns
    # A legitimate feature survives.
    assert "gift_amount" in out.columns


def test_pii_patterns_override_replaces_defaults():
    gifts = pd.DataFrame(
        {
            "donor_id": [1, 2],
            "gift_date": ["2023-08-01", "2023-01-01"],
            "gift_amount": [100.0, 200.0],
            "account_balance": [5.0, 6.0],
        }
    )
    t = EncounterTransformer(
        encounter_df=_encounters(),
        merge_key="donor_id",
        pii_patterns=("balance",),
    )
    out = _fit_transform_pandas(t, gifts)
    assert "account_balance" not in out.columns  # matched by override
    assert "gift_amount" in out.columns  # not matched by override
    assert "donor_id" not in out.columns  # merge_key always dropped


def test_allow_negative_days_emits_warning():
    t = EncounterTransformer(
        encounter_df=_encounters(), merge_key="donor_id", allow_negative_days=True
    )
    gifts = pd.DataFrame({"donor_id": [1], "gift_date": ["2023-08-01"]})
    with pytest.warns(UserWarning, match="allow_negative_days=True"):
        t.fit(gifts)


def test_default_does_not_emit_negative_days_warning():
    t = EncounterTransformer(encounter_df=_encounters(), merge_key="donor_id")
    gifts = pd.DataFrame({"donor_id": [1], "gift_date": ["2023-08-01"]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t.fit(gifts)
    assert not any("allow_negative_days" in str(w.message) for w in caught)


def _clinical_encounters():
    return pd.DataFrame(
        {
            "donor_id": [1, 1],
            "discharge_date": ["2022-01-01", "2023-06-15"],
            "service_line": ["cardiac", "cardiac"],
            "attending_physician_id": ["P1", "P2"],
        }
    )


def test_capacity_weights_override_changes_gravity_score():
    X = pd.DataFrame({"donor_id": [1]})
    default = GratefulPatientFeaturizer(encounter_df=_clinical_encounters())
    override = GratefulPatientFeaturizer(
        encounter_df=_clinical_encounters(), capacity_weights={"cardiac": 10.0}
    )
    # clinical_gravity_score is the first output column.
    default_score = default.fit(X).transform(X)[0, 0]
    override_score = override.fit(X).transform(X)[0, 0]
    assert np.isclose(default_score, 2 * 3.2)  # 2 cardiac encounters * default 3.2
    assert np.isclose(override_score, 2 * 10.0)
