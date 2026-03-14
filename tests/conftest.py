"""
Shared pytest fixtures.
"""

import pytest
from philanthropy.utils.testing import make_donor_dataset


@pytest.fixture(scope="session")
def donor_df():
    return make_donor_dataset(n_donors=50, random_state=0)
