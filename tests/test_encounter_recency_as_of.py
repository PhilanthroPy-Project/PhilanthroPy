"""``as_of`` cutoff on EncounterRecencyTransformer (issue #210).

The transformer is row-wise, so the cutoff blanks post-cutoff encounters to
``NaT`` rather than dropping their rows: dropping would change the output
length and break the transformer inside a ``Pipeline``.
"""

import numpy as np
import pandas as pd
import pytest

from philanthropy.preprocessing import EncounterRecencyTransformer

_FUTURE = pd.DataFrame({
    "last_encounter_date": ["2021-06-01", "2025-01-01", None],
})


def test_as_of_blanks_encounters_after_the_cutoff():
    t = EncounterRecencyTransformer(reference_date="2022-01-01", as_of="2022-01-01")
    out = t.fit_transform(_FUTURE)

    assert out.shape == (3, 3)          # row count preserved
    assert out[0, 0] == 214.0           # 2021-06-01 survives
    assert np.isnan(out[1, 0])          # 2025-01-01 blanked
    assert out[1, 1] == 0.0
    assert np.isnan(out[1, 2])


def test_as_of_is_inclusive_of_its_own_date():
    X = pd.DataFrame({"last_encounter_date": ["2022-01-01"]})
    t = EncounterRecencyTransformer(reference_date="2022-01-01", as_of="2022-01-01")
    assert t.fit_transform(X)[0, 0] == 0.0


def test_as_of_none_warns_instead_of_silently_going_negative():
    t = EncounterRecencyTransformer(reference_date="2022-01-01")
    with pytest.warns(UserWarning, match="scoring 1 encounter"):
        out = t.fit_transform(_FUTURE)
    # Unchanged behaviour, only louder.
    assert out[1, 0] < 0


def test_as_of_bounds_an_inferred_reference_date():
    t = EncounterRecencyTransformer(as_of="2022-01-01").fit(_FUTURE)
    assert t.reference_date_ == pd.Timestamp("2021-06-01")


def test_as_of_survives_a_timezone_aware_column():
    t = EncounterRecencyTransformer(
        reference_date="2022-01-01",
        timezone="America/Chicago",
        as_of="2022-01-01",
    )
    out = t.fit_transform(_FUTURE)
    assert np.isnan(out[1, 0])
    assert out[0, 0] > 0


def test_as_of_rejects_an_unparseable_date_in_fit():
    with pytest.raises(ValueError, match="EncounterRecencyTransformer"):
        EncounterRecencyTransformer(as_of="not-a-date").fit(_FUTURE)
