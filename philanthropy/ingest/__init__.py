"""
philanthropy.ingest
===================
On-ramp from a UniSchema ``ConstituentEvent`` stream to a PhilanthroPy
donor-level feature table.

``read_constituent_events`` loads UniSchema's JSON / NDJSON egress files;
``constituent_events_to_features`` aggregates them into the one-row-per-donor
feature frame the estimators consume.
"""

from ._constituent_events import (
    constituent_events_to_features,
    read_constituent_events,
)

__all__ = [
    "constituent_events_to_features",
    "read_constituent_events",
]
