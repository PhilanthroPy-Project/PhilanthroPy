"""tests/conftest.py pins TZ=UTC so date-dependent tests are reproducible off
a CI runner. These two tests make sure that pin is real *and* that it does not
hide the transformer's own `timezone=` handling.
"""

import os

import numpy as np
import pandas as pd

from philanthropy.preprocessing import EncounterRecencyTransformer


def test_process_timezone_is_pinned_to_utc():
    assert os.environ["TZ"] == "UTC"
    # A naive "today" and an explicit UTC "today" must land on the same day.
    naive = pd.Timestamp.today().normalize()
    utc = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    assert naive == utc


def test_encounter_recency_localises_to_a_non_utc_timezone():
    X = pd.DataFrame({"last_encounter_date": ["2023-01-01", "2023-06-01"]})

    utc = EncounterRecencyTransformer(reference_date="2023-12-31").fit_transform(X)
    eastern = EncounterRecencyTransformer(
        reference_date="2023-12-31", timezone="America/New_York"
    ).fit_transform(X)

    assert utc.shape == eastern.shape == (2, 3)
    assert np.isfinite(eastern).all()
    # Both the encounters and the reference date shift by the same offset, so
    # the day counts are unchanged. The point of the test is that passing a
    # non-UTC timezone runs the localisation branch without crashing or
    # silently drifting a day.
    np.testing.assert_allclose(eastern[:, 0], utc[:, 0])
    np.testing.assert_allclose(eastern[:, 0], [364.0, 213.0])


def test_unspecified_reference_date_uses_the_pinned_clock():
    X = pd.DataFrame({"last_encounter_date": ["2020-01-01"]})
    t = EncounterRecencyTransformer().fit(X)
    # Batch max, not the wall clock: the fallback only fires with no dates.
    assert t.reference_date_ == pd.Timestamp("2020-01-01")

    empty = pd.DataFrame({"last_encounter_date": [None]})
    t_empty = EncounterRecencyTransformer().fit(empty)
    assert t_empty.reference_date_ == pd.Timestamp.today().normalize()


def test_tz_aware_dates_with_a_naive_reference_date():
    # dates aware + naive reference: the reference must be localised to the
    # configured timezone before differencing (branch in
    # _compute_recency_features), and day counts stay finite.
    X = pd.DataFrame({"last_encounter_date": ["2023-01-01", "2023-06-01"]})
    t = EncounterRecencyTransformer(
        reference_date="2023-12-31", timezone="America/New_York"
    )

    out = t.fit_transform(X)

    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[:, 0], [364.0, 213.0])


def test_naive_dates_with_a_tz_aware_reference_date():
    # naive dates + aware reference: the dates are localised to UTC instead,
    # which is the other half of the tz-mismatch branch pair.
    X = pd.DataFrame({"last_encounter_date": ["2023-01-01", "2023-06-01"]})
    t = EncounterRecencyTransformer(reference_date="2023-12-31T00:00:00+00:00")

    out = t.fit_transform(X)

    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[:, 0], [364.0, 213.0])
