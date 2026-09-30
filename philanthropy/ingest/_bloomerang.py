"""
philanthropy.ingest._bloomerang
================================
Bridge from a Bloomerang transaction export to a PhilanthroPy donor-level
feature table.

`Bloomerang <https://bloomerang.co>`_ is a CRM common among small and
mid-sized nonprofits. Its REST API's ``Transaction`` object keys a gift to
the constituent via ``AccountId``, with ``Date``, ``Amount``, a read-only
``EntryType`` and a ``FundName``; a report or CSV export out of the product
instead carries a human label such as ``Account Number``. Both are accepted
here and normalised onto the canonical ``contact_id`` / ``receive_date`` /
``total_amount`` names.

:func:`read_bloomerang_transactions` loads the CSV(s);
:func:`bloomerang_transactions_to_features` drops the commitment rows and
hands the remaining payments to
:func:`~philanthropy.ingest.civicrm_contributions_to_features`, which already
knows how to roll a gift log up into the one-row-per-donor frame the
estimators consume.

**The trap this exists to prevent is commitment versus payment, same as
Raiser's Edge and NPSP.** Bloomerang's ``Transaction`` API distinguishes a
``Pledge`` (the supporter's up-front commitment to give a total amount over
time, recorded with the full instalment schedule) from each ``PledgePayment``
(one instalment, linked back to the pledge and applied against its
outstanding balance). A ``RecurringDonation`` is the schedule record for an
ongoing gift processed outside Bloomerang, not money received, while each
``RecurringDonationPayment`` is. Summing every transaction's ``Amount``
regardless of ``EntryType`` therefore double-counts a pledged dollar, once as
the promise and again as each payment, and counts a recurring schedule that
was never charged.

The fix is the same allowlist-by-exclusion shape as the other bridges: only
``Pledge`` and ``RecurringDonation`` (the two commitment-only entry types)
are dropped by default; ``Donation``, ``PledgePayment`` and
``RecurringDonationPayment`` are kept, because those are the money. Matching
ignores case, spacing and punctuation, so the API's ``PledgePayment`` and a
report's ``Pledge Payment`` collapse to the same key. Bloomerang exports are
user-configured like the others, so the excluded set is a documented default
parameter rather than a constant: see :data:`DEFAULT_EXCLUDED_ENTRY_TYPES`.

Sources: Bloomerang, *REST API V1* (https://bloomerang.com/api/rest-api-v1/),
the ``Transaction`` object's ``AccountId``, ``Date``, ``Amount``,
``EntryType`` (enum: ``Donation``, ``RecurringDonation``,
``RecurringDonationPayment``, ``Pledge``, ``PledgePayment``) and
``FundName`` fields; Bloomerang Help Center, *Transactions Report*
(https://help.bloomerang.com/en/articles/13382625-transactions-report),
which documents ``Account Number`` as a constituent-report export column and
transaction type as a report filter/column with the ``Pledge Payment`` and
``Recurring Donation Payment`` labels; Bloomerang, *Donations and Pledges*
webinar deck (https://info.bloomerang.com/rs/618-WGI-459/images/11.12.2024%20Donations%20and%20Pledges%20Part%201.pdf),
which describes a Pledge as the supporter's up-front commitment and each
instalment as a separate Pledge Payment applied against its balance.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

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
    "DEFAULT_EXCLUDED_ENTRY_TYPES",
    "bloomerang_transactions_to_features",
    "read_bloomerang_transactions",
]

#: Entry types excluded from the roll-up by default: the commitment rows,
#: whose amount is a promise or a not-yet-charged schedule rather than money
#: received. Matching ignores case, spacing and punctuation, so one spelling
#: here also matches the API's concatenated ``PledgePayment`` (kept, it is a
#: payment, not this set) and a report's spaced ``Pledge Payment``. Payment
#: rows (``Donation``, ``PledgePayment``, ``RecurringDonationPayment``) are
#: *not* in this set and are what the features are built from.
DEFAULT_EXCLUDED_ENTRY_TYPES = (
    "Pledge",
    "Recurring Donation",
)

# Bloomerang header normalisation, applied before the CiviCRM bridge's own.
# Case and punctuation are already collapsed ("Account Number" ->
# account_number), so this maps only the residue onto the canonical names.
# Anything absent here falls through to the CiviCRM canonicaliser, which
# already handles the labels the two systems happen to share ("Amount",
# "Email", "First Name").
_HEADER_ALIASES = {
    # Constituent key: report/export label, then REST API field name.
    "account_number": "contact_id",
    "accountid": "contact_id",
    "account_id": "contact_id",
    # Date: shared by the report column and the API field.
    "date": "receive_date",
    # Amount: shared by the report column and the API field.
    "amount": "total_amount",
    # Entry type: the column this module exists to read.
    "transaction_type": "gift_type",
    "entrytype": "gift_type",
    "entry_type": "gift_type",
    # Fund: Bloomerang's GL-style designation, the analogue of CiviCRM's
    # Financial Type. Feeds distinct_financial_types.
    "fund": "financial_type",
    "fundname": "financial_type",
    "fund_name": "financial_type",
}

# Entry-type matching key: strip everything but letters and digits so
# "Pledge Payment" and "PledgePayment" collapse to one value.
_NON_ALNUM_ALL = re.compile(r"[^a-z0-9]+")


def bloomerang_transactions_to_features(
    transactions: Union[Iterable[Mapping], pd.DataFrame],
    *,
    reference_date: Optional[Union[str, pd.Timestamp]] = None,
    exclude_entry_types: Optional[Sequence[str]] = DEFAULT_EXCLUDED_ENTRY_TYPES,
) -> pd.DataFrame:
    """Aggregate a Bloomerang transaction export into donor-level features.

    Commitment rows (pledges and recurring-donation schedules) are dropped
    first, then the surviving payment rows are handed to
    :func:`civicrm_contributions_to_features`, which produces the donor
    frame. Bloomerang has no contribution-status column analogous to
    CiviCRM's, so no status filter is applied.

    Parameters
    ----------
    transactions : iterable of mapping, or DataFrame
        Bloomerang transaction rows, under a report export label
        (``Account Number``, ``Date``, ``Amount``, ``Transaction Type``,
        ``Fund``) or the REST API field names (``AccountId``, ``Date``,
        ``Amount``, ``EntryType``, ``FundName``). ``contact_id``,
        ``receive_date`` and ``total_amount`` are required after
        normalisation; ``gift_type``, ``financial_type``, ``email``,
        ``first_name`` and ``last_name`` are used when present.
    reference_date : str or datetime-like, optional
        Anchor for the recency features. If ``None``, the latest transaction
        date in the batch is used, which keeps the aggregation free of "now"
        leakage.
    exclude_entry_types : sequence of str or None, default=:data:`DEFAULT_EXCLUDED_ENTRY_TYPES`
        Entry types to drop before aggregating, matched against ``gift_type``
        ignoring case, spacing and punctuation. ``None`` or an empty sequence
        disables the filter and sums **every** row, which double-counts
        pledged dollars and counts a recurring schedule never charged. Rows
        whose entry type is blank are kept either way: an unlabelled row
        cannot be shown to be a commitment.

    Returns
    -------
    features : pandas.DataFrame
        One row per donor, indexed by ``contact_id``, with the same columns
        :func:`civicrm_contributions_to_features` emits.
        ``distinct_financial_types`` counts distinct Funds here.

    Raises
    ------
    KeyError
        If ``contact_id``, ``receive_date`` or ``total_amount`` is absent
        after normalisation.

    Warns
    -----
    UserWarning
        If ``exclude_entry_types`` was requested but the export carries no
        entry type column. Silently summing a pledge together with its
        payments (or an uncharged recurring schedule) is exactly the error
        this bridge exists to prevent, so it is worth a warning rather than a
        quiet wrong total.

    Notes
    -----
    Nothing here deduplicates on a transaction id, so concatenating two
    exports that overlap in time will double-count the overlap; export
    disjoint date ranges.

    Examples
    --------
    >>> rows = [
    ...     {"Account Number": "88", "Date": "2025-01-10",
    ...      "Amount": "1200.00", "Transaction Type": "Pledge"},
    ...     {"Account Number": "88", "Date": "2025-02-10",
    ...      "Amount": "100.00", "Transaction Type": "PledgePayment"},
    ...     {"Account Number": "88", "Date": "2025-03-10",
    ...      "Amount": "100.00", "Transaction Type": "PledgePayment"},
    ... ]
    >>> feats = bloomerang_transactions_to_features(rows)
    >>> float(feats.loc["88", "total_gift_amount"])  # not 1400.0
    200.0
    >>> int(feats.loc["88", "gift_count"])
    2
    """
    df = _normalise_headers(_to_frame(transactions), _canonical_bloomerang)
    if df.empty:
        return _empty_feature_frame()

    missing = [col for col in _REQUIRED if col not in df.columns]
    if missing:
        raise KeyError(
            f"Bloomerang transaction export is missing {missing}. Export the "
            f"'Account Number', 'Date' and 'Amount' fields (REST API: "
            f"AccountId, Date, Amount); got {sorted(df.columns)}."
        )

    if exclude_entry_types:
        if "gift_type" in df.columns:
            unwanted = {_entry_type_key(t) for t in exclude_entry_types}
            keys = df["gift_type"].map(_entry_type_key)
            df = df[~keys.isin(unwanted)]
        else:
            warnings.warn(
                f"Bloomerang transaction export has no entry type column, so "
                f"exclude_entry_types={tuple(exclude_entry_types)!r} could not "
                f"be applied: pledges and recurring-donation schedules (if "
                f"any) are summed alongside the payments made against them, "
                f"which counts every committed dollar twice. Add the "
                f"'Transaction Type' field to the export.",
                stacklevel=2,
            )

    if df.empty:
        return _empty_feature_frame()
    return civicrm_contributions_to_features(
        df, reference_date=reference_date, statuses=None
    )


def read_bloomerang_transactions(path: Union[str, Path]) -> pd.DataFrame:
    """Read Bloomerang transaction export CSV(s) into one frame.

    Accepts a single ``.csv`` or a directory of them, walked recursively and
    concatenated in sorted relative-path order; symlinks are not followed.
    Every column is read as text and nothing is filtered: **commitment rows
    are still present**, and it is
    :func:`bloomerang_transactions_to_features` that drops them.

    Reading a gift export is CRM-agnostic once the headers are normalised, so
    this delegates to :func:`read_civicrm_contributions` rather than
    repeating its BOM handling and path hardening. Bloomerang headers are
    normalised on the way into
    :func:`bloomerang_transactions_to_features`, not here.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV file, or a directory of them.

    Returns
    -------
    transactions : pandas.DataFrame
        The export as written, with text values.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    return read_civicrm_contributions(path)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _canonical_bloomerang(header: str) -> str:
    key = _NON_ALNUM.sub("_", str(header).strip().lower()).strip("_")
    return _HEADER_ALIASES.get(key) or _canonical(header)


def _entry_type_key(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return _NON_ALNUM_ALL.sub("", str(value).strip().lower())
