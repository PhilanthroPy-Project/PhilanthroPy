"""
philanthropy.ingest._ellucian_advance
=====================================
Bridge from an Ellucian CRM Advance transaction export to a PhilanthroPy
donor-level feature table.

Advance's Transaction List shows one row per transaction with ``Date``,
``Legal`` (the hard credit amount), ``Credit`` (the soft credit amount) and
``Type``. ``Type`` holds either a transaction type (gift, pledge payment) or
a pledge type (straight pledge, grant, bequest intention), followed by the
donor type in parentheses, so a real pledge row reads
``Straight Pledge (Primary)``. ``Date`` is the date of record followed by
``#`` and the receipt number, e.g. ``01/10/2025 #0001234``.

This bridge strips both suffixes, drops pledge commitments by default (the
pledge payments made against them are kept), sums ``Legal`` only so soft
credits are not counted as money the donor gave, and delegates the donor
roll-up to the shared CiviCRM aggregator.

Sources (Advance help pages published by the University of Mississippi):

* Transaction List, for the ``Date``, ``Legal``, ``Credit`` and ``Type``
  columns: https://advance.olemiss.edu/Help/Advance/Transaction_List.htm
* Primary Pledge Detail, for the pledge types:
  https://advance.olemiss.edu/Help/Advance/Primary_Pledge_Detail.htm
* Primary Gift Detail, for the transaction types:
  https://advance.olemiss.edu/Help/Advance/Primary_Gift_Detail.htm
* Entity Profile, for the entity ID / ID number:
  https://advance.olemiss.edu/Help/Advance/Entity_Profile.htm

Both type lists are given as examples and each site validates its own values
against the ``tms_pledge_type`` and ``tms_transaction_type`` tables, so pass
``exclude_transaction_types`` if your site uses other pledge types.
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

#: Pledge types excluded from the roll-up by default: each is a commitment,
#: and the money arrives later as ``Pledge Payment`` rows, which are kept.
#: These are the pledge types the Advance help pages name. ``Grant`` is left
#: out on purpose: Advance uses it both as a pledge type (a promise) and as a
#: transaction type (money received), and the ``Type`` column alone cannot
#: tell the two apart. Matching ignores case, spacing, punctuation and the
#: trailing donor type in parentheses.
DEFAULT_EXCLUDED_ADVANCE_TRANSACTION_TYPES = (
    "Straight Pledge",
    "Bequest Intention",
    "Bequest Expectancy",
    "EFT",
    "Payroll Deduction",
    "Telefund Pledge",
    "Verbal Pledge",
)

_HEADER_ALIASES = {
    # Entity Profile looks constituents up by "entity ID" or "ID number". A
    # bare "ID" is left out: in a transaction export it may be the receipt.
    "entity_id": "contact_id",
    "entityid": "contact_id",
    "id_number": "contact_id",
    # Transaction List column names.
    "date": "receive_date",
    "legal": "total_amount",
    "legal_amount": "total_amount",
    "type": "gift_type",
    "transaction_type": "gift_type",
    "allocation": "financial_type",
}

_NON_ALNUM_ALL = re.compile(r"[^a-z0-9]+")
_DONOR_TYPE_SUFFIX = re.compile(r"\s*\([^)]*\)\s*$")
_RECEIPT_SUFFIX = re.compile(r"\s*#.*$")


def ellucian_advance_gifts_to_features(
    gifts: Iterable[Mapping] | pd.DataFrame,
    *,
    reference_date: str | pd.Timestamp | None = None,
    exclude_transaction_types: Sequence[str] | None = (
        DEFAULT_EXCLUDED_ADVANCE_TRANSACTION_TYPES
    ),
) -> pd.DataFrame:
    """Aggregate an Ellucian CRM Advance transaction export into donor features.

    Pledge commitment rows are dropped first, then the surviving gift and
    pledge-payment rows are handed to
    :func:`~philanthropy.ingest.civicrm_contributions_to_features`, which
    emits the one-row-per-donor frame.

    Parameters
    ----------
    gifts : iterable of mapping, or DataFrame
        Advance transaction rows. ``Entity ID`` (or ``ID Number``), ``Date``
        and ``Legal`` are required after normalisation; ``Type`` and
        ``Allocation`` are used when present. A ``Date`` that carries the
        receipt number (``01/10/2025 #0001234``) is read as the date alone.
        ``Credit`` (soft credit) is ignored.
    reference_date : str or datetime-like, optional
        Anchor for recency features. If ``None``, the latest surviving
        transaction date in the batch is used.
    exclude_transaction_types : sequence of str or None, \
default=DEFAULT_EXCLUDED_ADVANCE_TRANSACTION_TYPES
        Values of ``Type`` to drop before aggregation, compared after
        removing the parenthesised donor type, so ``Straight Pledge`` also
        drops ``Straight Pledge (Primary)`` and ``Straight Pledge (Joint)``.
        ``None`` or an empty sequence disables the filter and sums every
        row, pledges included.

    Returns
    -------
    pandas.DataFrame
        One row per entity, indexed by ``contact_id``, with the columns
        :func:`~philanthropy.ingest.civicrm_contributions_to_features`
        produces. Empty input, or input whose every row is excluded, gives
        the typed empty frame.

    Raises
    ------
    KeyError
        If the entity ID, ``Date`` or ``Legal`` column is missing after
        normalisation.

    Warns
    -----
    UserWarning
        If a filter is requested but the export has no ``Type`` column, so
        pledge commitments cannot be told apart from payments.

    Examples
    --------
    >>> from philanthropy.ingest import ellucian_advance_gifts_to_features
    >>> rows = [
    ...     {"Entity ID": "88", "Date": "01/10/2025 #0001234",
    ...      "Legal": "1200.00", "Type": "Straight Pledge (Primary)"},
    ...     {"Entity ID": "88", "Date": "02/10/2025 #0001301",
    ...      "Legal": "100.00", "Type": "Pledge Payment (Primary)"},
    ... ]
    >>> features = ellucian_advance_gifts_to_features(rows)
    >>> float(features.loc["88", "total_gift_amount"])
    100.0
    """
    df = _normalise_headers(_to_frame(gifts), _canonical_ellucian_advance)
    if df.empty:
        return _empty_feature_frame()

    missing = [col for col in _REQUIRED if col not in df.columns]
    if missing:
        raise KeyError(
            f"Ellucian CRM Advance transaction export is missing {missing}. "
            f"Export the 'Entity ID', 'Date' and 'Legal' fields; got "
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
                f"Ellucian CRM Advance transaction export has no Type column, "
                f"so exclude_transaction_types="
                f"{tuple(exclude_transaction_types)!r} could not be applied: "
                f"pledge commitments, if present, are summed alongside the "
                f"payments made against them. Add the 'Type' field to the "
                f"export.",
                stacklevel=2,
            )

    if df.empty:
        return _empty_feature_frame()
    df = df.assign(
        receive_date=df["receive_date"].map(
            lambda v: _RECEIPT_SUFFIX.sub("", v) if isinstance(v, str) else v
        )
    )
    return civicrm_contributions_to_features(
        df, reference_date=reference_date, statuses=None
    )


def read_ellucian_advance_gifts(path: str | Path) -> pd.DataFrame:
    """Read Ellucian CRM Advance transaction export CSV(s) into one frame.

    Reading is delegated to the shared CiviCRM CSV reader, which accepts a
    file or a directory of them. Header normalisation, the pledge filter and
    the receipt-number strip happen in
    :func:`ellucian_advance_gifts_to_features`.
    """
    return read_civicrm_contributions(path)


def _canonical_ellucian_advance(header: str) -> str:
    """Return the canonical name for an Advance export header."""
    key = _NON_ALNUM.sub("_", str(header).strip().lower()).strip("_")
    return _HEADER_ALIASES.get(key) or _canonical(header)


def _transaction_type_key(value: object) -> str:
    """Drop the parenthesised donor type, then collapse case and punctuation."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return _NON_ALNUM_ALL.sub("", _DONOR_TYPE_SUFFIX.sub("", str(value)).lower())
