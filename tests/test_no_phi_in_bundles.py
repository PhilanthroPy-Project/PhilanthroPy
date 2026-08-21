"""A saved model bundle must not carry the raw clinical encounter table.

``EncounterTransformer`` and ``GratefulPatientFeaturizer`` take the encounter
table as a constructor parameter, so before ``__getstate__`` was added every
``joblib.dump`` of a fitted pipeline wrote raw PHI into the artefact.
"""

import io

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from philanthropy.preprocessing import EncounterTransformer, GratefulPatientFeaturizer

MRN = "MRN-CANARY-8675309"


@pytest.fixture
def encounters():
    return pd.DataFrame({
        "donor_id": [1, 1, 2],
        "discharge_date": ["2020-01-01", "2021-06-15", "2020-09-30"],
        "service_line": ["cardiac", "cardiac", "oncology"],
        "attending_physician_id": ["P1", "P2", "P3"],
        "mrn": [MRN, MRN, "MRN-CANARY-0000001"],
    })


@pytest.fixture
def gifts():
    return pd.DataFrame({
        "donor_id": [1, 2, 3],
        "gift_date": ["2022-01-01", "2022-02-01", "2022-03-01"],
        "gift_amount": [10000.0, 750.0, 250.0],
    })


def _dump(obj):
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    return buf.getvalue()


def test_encounter_transformer_bundle_has_no_raw_encounters(encounters, gifts):
    t = EncounterTransformer(encounter_df=encounters).fit(gifts)
    blob = _dump(t)
    assert MRN.encode() not in blob
    assert b"attending_physician_id" not in blob


def test_grateful_patient_bundle_has_no_raw_encounters(encounters, gifts):
    f = GratefulPatientFeaturizer(encounter_df=encounters).fit(gifts[["donor_id"]])
    blob = _dump(f)
    assert MRN.encode() not in blob


@pytest.mark.parametrize("cls", [EncounterTransformer, GratefulPatientFeaturizer])
def test_round_tripped_transformer_still_transforms_identically(
    cls, encounters, gifts
):
    X = gifts if cls is EncounterTransformer else gifts[["donor_id"]]
    fitted = cls(encounter_df=encounters).fit(X)
    before = fitted.transform(X)

    restored = joblib.load(io.BytesIO(_dump(fitted)))
    np.testing.assert_array_equal(
        np.asarray(before, dtype=float), np.asarray(restored.transform(X), dtype=float)
    )
    # The table is gone, so a refit has to be given it again rather than
    # silently reusing stale clinical rows.
    assert restored.encounter_df is None
    with pytest.raises(ValueError, match="encounter_df"):
        restored.fit(X)


@pytest.mark.parametrize("cls", [EncounterTransformer, GratefulPatientFeaturizer])
def test_clone_is_unaffected(cls, encounters):
    # clone() goes through get_params, not pickle, so cross-validation and grid
    # search still receive a usable estimator.
    fresh = clone(cls(encounter_df=encounters))
    pd.testing.assert_frame_equal(fresh.encounter_df, encounters)


def test_getstate_does_not_mutate_the_live_instance(encounters, gifts):
    # __getstate__ must copy: mutating the returned dict in place would strip
    # the table off the object being pickled.
    t = EncounterTransformer(encounter_df=encounters).fit(gifts)
    t.__getstate__()
    pd.testing.assert_frame_equal(t.encounter_df, encounters)
