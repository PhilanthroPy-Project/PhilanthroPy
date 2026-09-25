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

NPSP: ``read_npsp_opportunities`` loads a Salesforce Nonprofit Success Pack
Opportunity export CSV; ``npsp_opportunities_to_features`` aggregates it,
dropping ``Pledged`` instalment rows first so a Recurring Donation instalment
is not counted both in its ``Pledged`` stage and again once it closes ``Won``.

``map_columns`` renames a user-supplied export's headers to the canonical
names a bridge above expects, raising one error listing every column still
missing after the rename.

``activities_to_features`` aggregates a long, multi-source activity log
(event attendance, volunteer shifts, email clicks, ...) into per-donor,
per-type engagement features, generalising the pattern above to an
open-ended set of activity types discovered from the data itself.

``read_gifts(path_or_df, source=...)`` looks up the CiviCRM, Raiser's Edge or
NPSP reader-and-aggregator pair by name, for a caller working with more than
one CRM export format.
"""

from ._activities import activities_to_features
from ._civicrm import (
    civicrm_contributions_to_features,
    read_civicrm_contributions,
)
from ._constituent_events import (
    constituent_events_to_features,
    read_constituent_events,
)
from ._map_columns import map_columns
from ._npsp import (
    DEFAULT_EXCLUDED_STAGES,
    npsp_opportunities_to_features,
    read_npsp_opportunities,
)
from ._raisers_edge import (
    DEFAULT_EXCLUDED_GIFT_TYPES,
    raisers_edge_gifts_to_features,
    read_raisers_edge_gifts,
)
from ._read_gifts import GIFT_SOURCES, read_gifts

__all__ = [
    "DEFAULT_EXCLUDED_GIFT_TYPES",
    "DEFAULT_EXCLUDED_STAGES",
    "GIFT_SOURCES",
    "activities_to_features",
    "civicrm_contributions_to_features",
    "constituent_events_to_features",
    "map_columns",
    "npsp_opportunities_to_features",
    "raisers_edge_gifts_to_features",
    "read_civicrm_contributions",
    "read_constituent_events",
    "read_gifts",
    "read_npsp_opportunities",
    "read_raisers_edge_gifts",
]
