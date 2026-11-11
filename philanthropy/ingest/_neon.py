"""
philanthropy.ingest._neon
=========================
Bridge from a Neon CRM donation export to a PhilanthroPy donor-level feature
table.

Neon CRM keeps donations, pledges and pledge payments in one donation table,
and classifies every row by a ``Donation Type`` of ``Donation``, ``Pledge``,
``Pledge Payment`` or ``Match Pledge``. A pledge (and a match pledge) is a
promise to pay later; a pledge payment is money received against it. This
bridge normalises Neon's field names to the canonical ``contact_id`` /
``receive_date`` / ``total_amount`` columns, drops the two commitment types by
default, and delegates the donor roll-up to the shared CiviCRM aggregator.

Field names come from two places. The API v2 ``Donation`` object names
``accountId``, ``date``, ``amount`` and ``fund``, so a donation search or API
result loads as is. A report export labels its columns in the report
builder; ``Donation Type`` is the label Neon's support articles use, and
``Account ID``, ``Donation Date`` and ``Donation Amount`` are accepted as the
same fields under the report builder's naming. A column under any other name
can be renamed first with :func:`~philanthropy.ingest.map_columns`.

Sources:

Neon CRM API v2 specification (``Donation`` schema; ``POST
/donations/search``, "Search for donations, pledges, and pledge payments"):
https://developer.neoncrm.com/api-v2/docs/v2.12.yaml

Neon CRM Developer Center, *Transactions* (donations, pledges and pledge
payments share one table): https://developer.neoncrm.com/transactions/

Neon CRM Support Center, *Reporting on Different Donation Types* (the four
donation types, the ``Donation Type`` search field and output column):
https://support.neoncrm.com/hc/en-us/articles/218528777-Reporting-on-Different-Donation-Types

Neon CRM Support Center, *Pledges* (pledges and pledge payments are donation
types, reported through the Donation Type filter):
https://support.neoncrm.com/hc/en-us/articles/218528707-Pledges
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

#: Donation types excluded from the roll-up by default: Neon's two commitment
#: types, not the pledge payments made against them. Matching ignores case,
#: spacing and punctuation, so ``Match Pledge`` and ``match_pledge`` collapse
#: to the same value.
DEFAULT_EXCLUDED_NEON_DONATION_TYPES = (
    "Pledge",
    "Match Pledge",
)

_HEADER_ALIASES = {
    # API v2 Donation object field names (``accountId`` normalises to
    # ``accountid``; ``amount`` is already canonical).
    "accountid": "contact_id",
    "date": "receive_date",
    "fund": "financial_type",
    # Report-builder labels for the same fields.
    "account_id": "contact_id",
    "donation_date": "receive_date",
    "donation_amount": "total_amount",
    "donation_type": "gift_type",
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

    Pledge and match-pledge rows are dropped first, then the surviving
    donation and pledge-payment rows are handed to
    :func:`civicrm_contributions_to_features`, which emits the one-row-per-
    donor frame.

    Parameters
    ----------
    donations : iterable of mapping, or DataFrame
        Neon donation rows, from a report export or the API. The account id
        (``Account ID`` or ``accountId``), donation date (``Donation Date``
        or ``date``) and amount (``Donation Amount`` or ``amount``) are
        required. ``Donation Type`` and ``Fund`` are used when present.
    reference_date : str or datetime-like, optional
        Anchor for recency features. If ``None``, the latest surviving
        donation date in the batch is used.
    exclude_donation_types : sequence of str or None, default \
DEFAULT_EXCLUDED_NEON_DONATION_TYPES
        Donation types to drop before aggregation, matched ignoring case,
        spacing and punctuation. The default drops ``Pledge`` and
        ``Match Pledge``. ``None`` or an empty sequence keeps every row.

    Returns
    -------
    features : pandas.DataFrame
        One row per account, indexed by ``contact_id``, with the columns of
        :func:`civicrm_contributions_to_features`. An empty input, or one
        whose rows are all excluded, returns an empty frame with the same
        columns and dtypes.

    Raises
    ------
    KeyError
        If the account id, donation date or amount field is missing.

    Warns
    -----
    UserWarning
        If ``exclude_donation_types`` is set but the export has no
        ``Donation Type`` column, so pledge commitments cannot be told apart
        from payments.

    Examples
    --------
    >>> rows = [
    ...     {"Account ID": "88", "Donation Date": "2025-01-10",
    ...      "Donation Amount": "1200.00", "Donation Type": "Pledge"},
    ...     {"Account ID": "88", "Donation Date": "2025-02-10",
    ...      "Donation Amount": "100.00", "Donation Type": "Pledge Payment"},
    ... ]
    >>> feats = neon_donations_to_features(rows)
    >>> float(feats.loc["88", "total_gift_amount"])  # the pledge is excluded
    100.0
    """
    df = _neon_received_rows(
        donations, exclude_donation_types=exclude_donation_types
    )
    if df.empty:
        return _empty_feature_frame()
    return civicrm_contributions_to_features(
        df, reference_date=reference_date, statuses=None
    )


def read_neon_donations(path: str | Path) -> pd.DataFrame:
    """Read Neon CRM donation export CSV(s) into one frame.

    Reading is delegated to
    :func:`~philanthropy.ingest.read_civicrm_contributions`; the Neon header
    aliases and the donation-type filter are applied in
    :func:`neon_donations_to_features`.

    Parameters
    ----------
    path : str or pathlib.Path
        A donation export CSV, or a directory of them (walked recursively and
        concatenated).

    Returns
    -------
    donations : pandas.DataFrame
        The export rows as text, headers lower-cased and underscored. A
        directory holding no CSV returns an empty frame.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    return read_civicrm_contributions(path)


def _canonical_neon(header: str) -> str:
    key = _NON_ALNUM.sub("_", str(header).strip().lower()).strip("_")
    return _HEADER_ALIASES.get(key) or _canonical(header)


def _donation_type_key(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return _NON_ALNUM_ALL.sub("", str(value).strip().lower())


def _neon_received_rows(
    donations: Iterable[Mapping] | pd.DataFrame,
    exclude_donation_types: Sequence[str] | None = (
        DEFAULT_EXCLUDED_NEON_DONATION_TYPES
    ),
) -> pd.DataFrame:
    """Normalise a Neon donation export and keep only the rows recording
    money received, the rows :func:`neon_donations_to_features` counts.

    Shared with :func:`~philanthropy.ingest.read_gift_rows` so both count the
    same gifts.
    """
    df = _normalise_headers(_to_frame(donations), _canonical_neon)
    if df.empty:
        return df

    missing = [col for col in _REQUIRED if col not in df.columns]
    if missing:
        raise KeyError(
            f"Neon donation export is missing {missing}. Export the "
            f"'Account ID', 'Donation Date' and 'Donation Amount' fields; "
            f"got {sorted(df.columns)}."
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
                stacklevel=3,
            )
    return df
