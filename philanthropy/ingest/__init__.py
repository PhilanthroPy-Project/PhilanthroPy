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
keeping only closed/won stages (``Closed Won``, ``Awarded``, ``Posted``) so a
``Pledged`` instalment, an open pipeline row, or ``Closed Lost`` is not
counted as a gift.

Bloomerang: ``read_bloomerang_transactions`` loads a Bloomerang transaction
export CSV; ``bloomerang_transactions_to_features`` aggregates it, dropping
the commitment rows (``Pledge``, ``Recurring Donation``) first so a pledged
dollar is not counted both as the promise and as the payments against it.

DonorPerfect: ``read_donorperfect_gifts`` loads a DonorPerfect gift export
CSV; ``donorperfect_gifts_to_features`` aggregates it, dropping Pledge
records and split-gift Main totals (``record_type`` ``P`` / ``M``) first so
a pledged or split dollar is not counted both as the promise/total and as
the payments/splits against it.

Neon CRM: ``read_neon_donations`` loads a Neon donation export CSV;
``neon_donations_to_features`` aggregates it, dropping pledge commitments
first so pledged dollars are not counted before payment.

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

``build_leadership_snapshots`` builds a per-donor, per-fiscal-year training
table for an upgrade model: one row per (donor, fiscal year T) for every
donor whose FY T giving falls in a mid-level band, features computed only
from data through the end of T, and a target reading whether FY T+1 crossed
a leadership threshold. ``build_snapshots`` is the same idea for every
question (upgrade, lapse, response next year, next gift amount) on one shared
column set, with an optional minimum number of giving years.
"""

from ._activities import activities_to_features
from ._bloomerang import (
    DEFAULT_EXCLUDED_ENTRY_TYPES,
    bloomerang_transactions_to_features,
    read_bloomerang_transactions,
)
from ._civicrm import (
    civicrm_contributions_to_features,
    read_civicrm_contributions,
)
from ._constituent_events import (
    constituent_events_to_features,
    read_constituent_events,
)
from ._donorperfect import (
    DEFAULT_EXCLUDED_RECORD_TYPES,
    donorperfect_gifts_to_features,
    read_donorperfect_gifts,
)
from ._map_columns import map_columns
from ._neon import (
    DEFAULT_EXCLUDED_NEON_DONATION_TYPES,
    neon_donations_to_features,
    read_neon_donations,
)
from ._npsp import (
    DEFAULT_INCLUDED_STAGES,
    npsp_opportunities_to_features,
    read_npsp_opportunities,
)
from ._raisers_edge import (
    DEFAULT_EXCLUDED_GIFT_TYPES,
    raisers_edge_gifts_to_features,
    read_raisers_edge_gifts,
)
from ._read_gifts import GIFT_SOURCES, read_gifts
from ._snapshots import build_snapshots
from ._upgrade_snapshots import build_leadership_snapshots

__all__ = [
    "DEFAULT_EXCLUDED_ENTRY_TYPES",
    "DEFAULT_EXCLUDED_GIFT_TYPES",
    "DEFAULT_EXCLUDED_NEON_DONATION_TYPES",
    "DEFAULT_EXCLUDED_RECORD_TYPES",
    "DEFAULT_INCLUDED_STAGES",
    "GIFT_SOURCES",
    "activities_to_features",
    "bloomerang_transactions_to_features",
    "build_leadership_snapshots",
    "build_snapshots",
    "civicrm_contributions_to_features",
    "constituent_events_to_features",
    "donorperfect_gifts_to_features",
    "map_columns",
    "neon_donations_to_features",
    "npsp_opportunities_to_features",
    "raisers_edge_gifts_to_features",
    "read_bloomerang_transactions",
    "read_civicrm_contributions",
    "read_constituent_events",
    "read_donorperfect_gifts",
    "read_gifts",
    "read_neon_donations",
    "read_npsp_opportunities",
    "read_raisers_edge_gifts",
]
