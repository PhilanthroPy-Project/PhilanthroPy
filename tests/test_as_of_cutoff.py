"""The as-of cutoff on the encounter transformers.

Without it, a gift dated 2020 is featurised from encounters recorded in 2024:
`days_since_last_discharge` is measured from the all-time max discharge, so the
more a donor engages *after* the gift, the more the feature is destroyed for
exactly the donors it should be strongest for.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from philanthropy.preprocessing import EncounterTransformer, GratefulPatientFeaturizer

# Donor 1 has a pre-gift encounter (2019) and a post-gift one (2024).
# Donor 2 has only a pre-gift encounter.
ENCOUNTERS = pd.DataFrame({
    "donor_id": [1, 1, 2],
    "discharge_date": ["2019-01-01", "2024-06-15", "2019-09-30"],
    "service_line": ["cardiac", "cardiac", "oncology"],
    "attending_physician_id": ["P1", "P2", "P3"],
})
GIFTS = pd.DataFrame({
    "donor_id": [1, 2],
    "gift_date": ["2021-01-01", "2021-01-01"],
    "gift_amount": [10000.0, 500.0],
})
CUTOFF = "2021-01-01"


def test_default_still_sees_the_whole_table():
    # Unchanged behaviour: as_of=None is the default and must not shift results.
    t = EncounterTransformer(encounter_df=ENCOUNTERS).fit(GIFTS)
    assert t.encounter_summary_.loc[1, "last_discharge"] == pd.Timestamp("2024-06-15")
    assert t.encounter_summary_.loc[1, "encounter_count"] == 2


def test_cutoff_excludes_post_decision_encounters():
    t = EncounterTransformer(encounter_df=ENCOUNTERS, as_of=CUTOFF).fit(GIFTS)
    # Donor 1's 2024 encounter had not happened at the 2021 decision point.
    assert t.encounter_summary_.loc[1, "last_discharge"] == pd.Timestamp("2019-01-01")
    assert t.encounter_summary_.loc[1, "encounter_count"] == 1
    # Donor 2 is entirely pre-cutoff and must be untouched.
    assert t.encounter_summary_.loc[2, "last_discharge"] == pd.Timestamp("2019-09-30")
    assert t.encounter_summary_.loc[2, "encounter_count"] == 1


def test_cutoff_changes_the_feature_not_just_the_summary():
    # The point of the parameter: the feature a model actually sees moves.
    leaky_t = EncounterTransformer(encounter_df=ENCOUNTERS).fit(GIFTS)
    honest_t = EncounterTransformer(encounter_df=ENCOUNTERS, as_of=CUTOFF).fit(GIFTS)
    col = list(leaky_t.get_feature_names_out()).index("days_since_last_discharge")

    leaky = leaky_t.transform(GIFTS)
    honest = honest_t.transform(GIFTS)

    # Donor 1 (row 0). Under the leaky default the all-time max discharge is the
    # 2024 encounter, which post-dates the 2021 gift, so the recency feature is
    # negative and coerced to NaN: destroyed for the donor it should be
    # strongest for. With the cutoff it is the real 731-day gap.
    assert np.isnan(leaky[0, col])
    assert honest[0, col] == pytest.approx(731.0)

    # Donor 2 has no post-cutoff encounter, so nothing about them moves.
    assert leaky[1, col] == honest[1, col]


def test_boundary_is_inclusive():
    enc = pd.DataFrame({"donor_id": [1], "discharge_date": [CUTOFF]})
    t = EncounterTransformer(encounter_df=enc, as_of=CUTOFF).fit(GIFTS)
    assert t.encounter_summary_.loc[1, "encounter_count"] == 1


def test_cutoff_before_all_history_warns_and_empties_the_summary():
    with pytest.warns(UserWarning, match="excluded every encounter row"):
        t = EncounterTransformer(encounter_df=ENCOUNTERS, as_of="1990-01-01").fit(GIFTS)
    assert t.encounter_summary_.empty


def test_unparseable_cutoff_raises():
    with pytest.raises(ValueError, match="parseable date"):
        EncounterTransformer(encounter_df=ENCOUNTERS, as_of="not-a-date").fit(GIFTS)


def test_grateful_patient_cutoff_reduces_clinical_gravity():
    X = GIFTS[["donor_id"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = GratefulPatientFeaturizer(encounter_df=ENCOUNTERS).fit(X)
        bounded = GratefulPatientFeaturizer(
            encounter_df=ENCOUNTERS, as_of=CUTOFF
        ).fit(X)
    # Donor 1 loses one of two cardiac encounters, so gravity halves.
    assert (
        bounded.encounter_summary_.loc[1, "clinical_gravity_score"]
        < full.encounter_summary_.loc[1, "clinical_gravity_score"]
    )
    assert bounded.encounter_summary_.loc[1, "distinct_physicians"] == 1


def test_as_of_round_trips_through_get_params():
    for cls in (EncounterTransformer, GratefulPatientFeaturizer):
        est = cls(encounter_df=ENCOUNTERS, as_of=CUTOFF)
        assert est.get_params()["as_of"] == CUTOFF
