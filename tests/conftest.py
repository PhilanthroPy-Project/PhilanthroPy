"""
Shared pytest fixtures.
"""

import os
import time

import pytest

from philanthropy.datasets import make_donor_dataset

# Pin the process timezone before any test imports pandas' date machinery.
# `EncounterRecencyTransformer` falls back to `pd.Timestamp.today().normalize()`
# when no reference_date is given, which is local-clock dependent, an unpinned
# TZ makes those tests pass on a CI runner in UTC and fail on a laptop that
# isn't. Explicit non-UTC behaviour is covered in tests/test_encounter_timezone.py.
os.environ["TZ"] = "UTC"
if hasattr(time, "tzset"):  # not available on Windows
    time.tzset()


@pytest.fixture(scope="session")
def donor_df():
    return make_donor_dataset(n_donors=50, random_state=0)
