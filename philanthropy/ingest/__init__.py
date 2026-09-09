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

Raiser's Edge: ``read_raisers_edge_gifts`` loads a Blackbaud gift export CSV;
``raisers_edge_gifts_to_features`` aggregates it, dropping the commitment rows
(pledges, matching gift pledges, recurring gift templates) first so a pledged
dollar is not counted both as the promise and as the payments against it.
"""

from ._civicrm import (
    civicrm_contributions_to_features,
    read_civicrm_contributions,
)
from ._constituent_events import (
    constituent_events_to_features,
    read_constituent_events,
)
from ._raisers_edge import (
    DEFAULT_EXCLUDED_GIFT_TYPES,
    raisers_edge_gifts_to_features,
    read_raisers_edge_gifts,
)

__all__ = [
    "DEFAULT_EXCLUDED_GIFT_TYPES",
    "civicrm_contributions_to_features",
    "constituent_events_to_features",
    "raisers_edge_gifts_to_features",
    "read_civicrm_contributions",
    "read_constituent_events",
    "read_raisers_edge_gifts",
]
