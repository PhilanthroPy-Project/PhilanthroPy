"""
philanthropy.ingest
===================
On-ramps from an upstream donor system to a PhilanthroPy donor-level feature
table.

UniSchema: ``read_constituent_events`` loads UniSchema's JSON / NDJSON egress
files; ``constituent_events_to_features`` aggregates them into the
one-row-per-donor feature frame the estimators consume.

CiviCRM: ``read_civicrm_contributions`` loads a contribution export CSV;
``civicrm_contributions_to_features`` aggregates it the same way, dropping
test-mode and non-``Completed`` rows first.
"""

from ._civicrm import (
    civicrm_contributions_to_features,
    read_civicrm_contributions,
)
from ._constituent_events import (
    constituent_events_to_features,
    read_constituent_events,
)

__all__ = [
    "civicrm_contributions_to_features",
    "constituent_events_to_features",
    "read_civicrm_contributions",
    "read_constituent_events",
]
