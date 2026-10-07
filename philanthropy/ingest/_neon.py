"""
philanthropy.ingest._neon
=========================
Bridge from a Neon CRM donation export to a PhilanthroPy donor-level feature
table.

Neon CRM documents donations as transactions with a linked account, donation
date and donation amount. Pledges and pledge payments share the donation
shape, but a pledge is the commitment while a pledge payment is money received.
This bridge normalises Neon export/API labels to the canonical
``contact_id`` / ``receive_date`` / ``total_amount`` columns, drops pledge
commitments by default, and delegates the donor roll-up to the shared CiviCRM
aggregator.

Sources: Neon CRM Developer Center, *Transactions*
(https://developer.neoncrm.com/transactions/), which documents linked account,
donation date and donation amount, and Neon One Support Center, *Pledges*
(https://support.neonone.com/hc/en-us/articles/4407399189517-Pledges),
which describes pledges and pledge payments as donation types.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from ._civicrm import (
    _NON_ALNUM,
    _REQUIRED,
    _canonical,
    _empty_feature_frame,
    _normalise_headers,
    _to_frame,
    civicrm_contributions_to_features,
    read_civicrm_contributions,
)

__all__ = [
    "DEFAULT_EXCLUDED_NEON_DONATION_TYPES",
    "neon_donations_to_features",
    "read_neon_donations",
]

#: Donation types excluded from the roll-up by default: pledge commitments,
#: not the later pledge payments made against them. Matching ignores case,
#: spacing and punctuation, so ``Matching Pledge`` and ``matching_pledge``
#: collapse to the same value.
DEFAULT_EXCLUDED_NEON_DONATION_TYPES = (
    "Pledge",
    "Matching Pledge",
)

_HEADER_ALIASES = {
    # Neon docs call this the linked account; CSV reports commonly expose an
    # Account ID / Account Number / Donor ID spelling.
    "linked_account": "contact_id",
    "linked_account_id": "contact_id",
    "linkedaccount": "contact_id",
    "linkedaccountid": "contact_id",
    "account": "contact_id",
    "account_id": "contact_id",
    "account_number": "contact_id",
    "constituent_id": "contact_id",
    "donor_id": "contact_id",
    "donor": "contact_id",
    # Required donation fields.
    "date": "receive_date",
    "donation_date": "receive_date",
    "transaction_date": "receive_date",
    "amount": "total_amount",
    "donation_amount": "total_amount",
    "transaction_amount": "total_amount",
    # Pledge-vs-payment discriminator.
    "type": "gift_type",
    "donation_type": "gift_type",
    "transaction_type": "gift_type",
    # Allocation dimensions, used as the shared financial type feature.
    "campaign": "financial_type",
    "campaign_name": "financial_type",
    "fund": "financial_type",
    "fund_name": "financial_type",
    "fundname": "financial_type",
    "purpose": "financial_type",
    "purpose_name": "financial_type",
    "purposename": "financial_type",
}

_NON_ALNUM_ALL = re.compile(r"[^a-z0-9]+")


def neon_donations_to_features(
    donations: Iterable[Mapping] | pd.DataFrame,
    *,
    reference_date: str | pd.Timestamp | None = None,
    exclude_donation_types: Sequence[str] | None = (
        DEFAULT_EXCLUDED_NEON_DONATION_TYPES
    ),
) -> pd.DataFrame:
    """Aggregate a Neon CRM donation export into donor-level features.

    Pledge commitment rows are dropped first, then the surviving donation and
    pledge-payment rows are handed to
    :func:`civicrm_contributions_to_features`, which emits the one-row-per-
    donor frame.
    """
    df = _normalise_headers(_to_frame(donations), _canonical_neon)
    if df.empty:
        return _empty_feature_frame()

    missing = [col for col in _REQUIRED if col not in df.columns]
    if missing:
        raise KeyError(
            f"Neon donation export is missing {missing}. Export the linked "
            f"account, donation date and donation amount fields; got "
            f"{sorted(df.columns)}."
        )

    if exclude_donation_types:
        if "gift_type" in df.columns:
            unwanted = {_donation_type_key(t) for t in exclude_donation_types}
            keys = df["gift_type"].map(_donation_type_key)
            df = df[~keys.isin(unwanted)]
        else:
            warnings.warn(
                f"Neon donation export has no donation type column, so "
                f"exclude_donation_types={tuple(exclude_donation_types)!r} "
                f"could not be applied: pledge commitments, if present, are "
                f"summed alongside payments made against them. Add the "
                f"'Donation Type' field to the export.",
                stacklevel=2,
            )

    if df.empty:
        return _empty_feature_frame()
    return civicrm_contributions_to_features(
        df, reference_date=reference_date, statuses=None
    )


def read_neon_donations(path: str | Path) -> pd.DataFrame:
    """Read Neon CRM donation export CSV(s) into one frame."""
    return read_civicrm_contributions(path)


def _canonical_neon(header: str) -> str:
    key = _NON_ALNUM.sub("_", str(header).strip().lower()).strip("_")
    return _HEADER_ALIASES.get(key) or _canonical(header)


def _donation_type_key(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return _NON_ALNUM_ALL.sub("", str(value).strip().lower())
