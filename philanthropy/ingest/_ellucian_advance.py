"""
philanthropy.ingest._ellucian_advance
=====================================
Bridge from an Ellucian CRM Advance transaction export to a PhilanthroPy
donor-level feature table.

Ellucian CRM Advance presents giving as constituent transactions with a
transaction date, legal amount, transaction type and allocation/designation.
This bridge accepts those export labels, plus common API/database spellings,
normalises them to PhilanthroPy's canonical gift-log columns, drops pledge
commitment rows by default, and delegates the donor roll-up to the shared
CiviCRM aggregator.

Sources: public Advance help pages for transaction list fields and constituent
IDs:

* https://advance.olemiss.edu/Help/Advance/Transaction_List.htm
* https://advance.olemiss.edu/Help/Advance/Entity_Profile.htm
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
    "DEFAULT_EXCLUDED_ADVANCE_TRANSACTION_TYPES",
    "ellucian_advance_gifts_to_features",
    "read_ellucian_advance_gifts",
]

#: Ellucian Advance transaction types excluded from the roll-up by default:
#: pledge/commitment rows are promises, while pledge-payment and outright-gift
#: rows are money received. Matching ignores case, spacing and punctuation.
DEFAULT_EXCLUDED_ADVANCE_TRANSACTION_TYPES = (
    "Pledge",
    "Planned Gift",
    "Recurring Pledge",
)

_HEADER_ALIASES = {
    "entity_id": "contact_id",
    "entityid": "contact_id",
    "id_number": "contact_id",
    "id": "contact_id",
    "constituent_id": "contact_id",
    "donor_id": "contact_id",
    "date": "receive_date",
    "transaction_date": "receive_date",
    "gift_date": "receive_date",
    "legal": "total_amount",
    "legal_amount": "total_amount",
    "amount": "total_amount",
    "gift_amount": "total_amount",
    "type": "gift_type",
    "transaction_type": "gift_type",
    "gift_type": "gift_type",
    "allocation": "financial_type",
    "allocation_code": "financial_type",
    "designation": "financial_type",
    "designation_code": "financial_type",
}

_NON_ALNUM_ALL = re.compile(r"[^a-z0-9]+")


def ellucian_advance_gifts_to_features(
    gifts: Iterable[Mapping] | pd.DataFrame,
    *,
    reference_date: str | pd.Timestamp | None = None,
    exclude_transaction_types: Sequence[str] | None = (
        DEFAULT_EXCLUDED_ADVANCE_TRANSACTION_TYPES
    ),
) -> pd.DataFrame:
    """Aggregate an Ellucian CRM Advance gift export into donor features.

    Pledge-like commitment rows are dropped first, then the surviving outright
    gift and payment rows are handed to
    :func:`civicrm_contributions_to_features`.
    """
    df = _normalise_headers(_to_frame(gifts), _canonical_ellucian_advance)
    if df.empty:
        return _empty_feature_frame()

    missing = [col for col in _REQUIRED if col not in df.columns]
    if missing:
        raise KeyError(
            f"Ellucian CRM Advance gift export is missing {missing}. Export "
            f"the entity ID, Date, Legal amount and Type fields; got "
            f"{sorted(df.columns)}."
        )

    if exclude_transaction_types:
        if "gift_type" in df.columns:
            unwanted = {
                _transaction_type_key(t) for t in exclude_transaction_types
            }
            keys = df["gift_type"].map(_transaction_type_key)
            df = df[~keys.isin(unwanted)]
        else:
            warnings.warn(
                f"Ellucian CRM Advance gift export has no transaction type "
                f"column, so exclude_transaction_types="
                f"{tuple(exclude_transaction_types)!r} could not be applied: "
                f"pledge commitments, if present, are summed alongside "
                f"payments made against them. Add the 'Type' field to the "
                f"export.",
                stacklevel=2,
            )

    if df.empty:
        return _empty_feature_frame()
    return civicrm_contributions_to_features(
        df, reference_date=reference_date, statuses=None
    )


def read_ellucian_advance_gifts(path: str | Path) -> pd.DataFrame:
    """Read Ellucian CRM Advance gift export CSV(s) into one frame."""
    return read_civicrm_contributions(path)


def _canonical_ellucian_advance(header: str) -> str:
    key = _NON_ALNUM.sub("_", str(header).strip().lower()).strip("_")
    return _HEADER_ALIASES.get(key) or _canonical(header)


def _transaction_type_key(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return _NON_ALNUM_ALL.sub("", str(value).strip().lower())
