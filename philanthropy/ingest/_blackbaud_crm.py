"""
philanthropy.ingest._blackbaud_crm
==================================
Bridge from a Blackbaud CRM revenue export to a PhilanthroPy donor-level
feature table.

`Blackbaud CRM <https://www.blackbaud.com/solutions/blackbaud-crm>`_ is the
enterprise-tier product from Blackbaud, distinct from Raiser's Edge, and it
models a gift differently. A gift is modelled as a *revenue record* against
a *constituent*, not a Raiser's Edge-style Gift row. The revenue record
lives in the ``FINANCIALTRANSACTION`` table (Blackbaud's own views alias
``dbo.FINANCIALTRANSACTION`` as ``REVENUE``) and carries the constituent
(``CONSTITUENTID``, a foreign key to ``CONSTITUENT.ID``), the transaction
date (``DATE``), the amount in the transaction currency
(``TRANSACTIONAMOUNT``), and a revenue transaction type (``TYPECODE``, with
a computed ``TYPE`` column exposing the human-readable label).

An export out of Blackbaud CRM carries the front-end labels the query
designer or an acknowledgement export writes (``Constituent ID``,
``Revenue Type``, ``Date``, ``Amount``, ``Application type``), which the
schema reference exposes as computed string columns (``TYPE`` on
``FINANCIALTRANSACTION``, ``APPLICATION`` on ``REVENUESPLIT_EXT``) alongside
their numeric source columns (``TYPECODE``, ``APPLICATIONCODE``). This bridge
reads the string form. An export that carries only the numeric codes is
rejected with a warning rather than translated, because a numeric code this
bridge does not recognise cannot be assumed to mean money received.

:func:`read_blackbaud_crm_revenue` loads the CSV(s);
:func:`blackbaud_crm_revenue_to_features` keeps only the received revenue
types and hands the surviving rows to
:func:`~philanthropy.ingest.civicrm_contributions_to_features`, which already
knows how to roll a revenue log up into the one-row-per-donor frame the
estimators consume.

**The trap this exists to prevent is commitment versus payment.** In
Blackbaud CRM a pledge and the money paid against it are *separate revenue
records*, distinguished by their transaction type (``TYPE``): ``Pledge`` is
the promise, and ``Payment`` is the money. Summing every revenue record's
``TRANSACTIONAMOUNT`` therefore counts each pledged dollar twice (once as
the promise and again as each payment against it). The same is true of
``Recurring gift``, which is a schedule rather than a receipt, and
``Matching gift claim`` and ``Planned gift``, which are expectations instead
of money. None of them represent a received gift, so the allowlist keeps
only ``Payment``.

Unlike Raiser's Edge, Blackbaud CRM's ``TYPECODE`` is a fixed schema
enumeration rather than an org-configurable list, so this bridge uses an
allowlist rather than an exclusion set: only ``Payment`` is included by
default, and an unrecognised type is dropped rather than silently summed.
See :data:`DEFAULT_INCLUDED_REVENUE_TYPES`. A numeric-only export (a raw
``TYPECODE`` column with no ``TYPE`` label) is not supported: the label
columns are the documented export shape, and the missing-column warning
fires in that case.

**Application versus transaction type:** ``FINANCIALTRANSACTION.TYPE`` and
``REVENUESPLIT_EXT.APPLICATION`` overlap in vocabulary (``Pledge`` and
``Recurring gift`` appear in both), but mean different things. The
transaction type describes the *structure* (commitment or a payment) of the
record. The application describes what the money is *applied to*. A ``Payment``
transaction applied to a ``Pledge`` is money received against a pledge and is
kept by this bridge. A ``Pledge`` transaction is the promise itself and is
dropped. The filter operates on the transaction type, so the distinction is
preserved.

Sources:

Blackbaud, *Blackbaud CRM Database Schema (FINANCIALTRANSACTION)*
(https://webfiles-sc1.blackbaud.com/files/support/helpfiles/crm-db/content/financialtransaction.html),
which documents ``CONSTITUENTID`` (FK to CONSTITUENT), ``DATE``,
``TRANSACTIONAMOUNT``, ``TYPECODE`` (with its full computed ``TYPE`` label
list), and ``USERDEFINEDID``.

Blackbaud, *Blackbaud CRM Database Schema (REVENUESPLIT_EXT)*
(https://webfiles-sc1.blackbaud.com/files/support/helpfiles/crm-db/content/revenuesplit_ext.html),
which documents ``APPLICATIONCODE`` (with its computed ``APPLICATION`` label
list) and ``DESIGNATIONID`` (FK to DESIGNATION).

Blackbaud, *Blackbaud CRM Database Schema (CONSTITUENT)*
(https://webfiles-sc1.blackbaud.com/files/support/helpfiles/crm-db/content/constituent.html),
which documents the ``ID`` primary key and the computed ``LOOKUPID`` column
(always unique; falls back to a sequence-based system identifier when no
custom identifier is set).

Blackbaud, *Blackbaud CRM Database Schema (BATCHREVENUE)*
(https://webfiles-sc1.blackbaud.com/files/support/helpfiles/crm-db/content/batchrevenue.html),
which documents the batch-entry path and its ``REVENUEID`` foreign key to
``FINANCIALTRANSACTION.ID``.

Blackbaud, *Blackbaud CRM Database Schema (V_SEARCH_OUTSTANDING_PLEDGE)*
(https://webfiles-sc1.blackbaud.com/files/support/helpfiles/crm-db/content/v_search_outstanding_pledge.html),
which aliases ``FINANCIALTRANSACTION`` as ``REVENUE``, joins through
``FINANCIALTRANSACTIONLINEITEM`` and ``REVENUESPLIT_EXT``, and confirms
``TRANSACTIONAMOUNT`` as the aggregation field.

Blackbaud SKY API, *CreateRecurringGift* resource
(https://developer.sky.blackbaud.com/api#api=crm-revmg&operation=CreateRecurringGift),
which documents the wire-format field names ``constituent_id``, ``date``,
``amount``, and the nested split fields ``designation_id``,
``application_code``, and ``type_code``.

Blackbaud, *Revenue Transaction Types*
(https://webfiles-sc1.blackbaud.com/files/support/helpfiles/crm/us/40/Content/RPTransactionTypes.html),
which names the four primary revenue transaction types (payments, pledges,
recurring gifts, and matching gift claims) and describes pledges, recurring
gifts, and matching gift claims as promises rather than receipts. This is
the source for the allowlist defaulting to ``Payment``.

Blackbaud, *Extended Output Formats* for acknowledgement exports
(https://webfiles-sc1.blackbaud.com/files/support/helpfiles/crm/us/40/Content/RPAckOutputFormatsExtended.html),
which shows the export column labels ``Constituent ID``, ``Revenue Type``,
``Date``, ``Amount``, ``Application type``, ``Designation name``,
``Designation amount``, and ``Designation ID``.
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
    "DEFAULT_INCLUDED_REVENUE_TYPES",
    "blackbaud_crm_revenue_to_features",
    "read_blackbaud_crm_revenue",
]

#: Revenue transaction types included in the roll-up by default, matched
#: against the ``gift_type`` column (the export's ``Revenue Type`` label that
#: Blackbaud CRM calls "Revenue Transaction Types"). Only ``Payment`` is listed:
#: it is the sole transaction type that unambiguously represents money
#: received. Blackbaud's own Revenue Transaction Types reference names four
#: primary types (payments, pledges, recurring gifts, and matching gift
#: claims) and describes pledges, recurring gifts, and matching gift claims
#: as promises rather than receipts. ``Planned gift``, ``Grant award``,
#: ``Auction donation``, ``Donor challenge claim``, ``Pending Gift``, and
#: ``Membership installment plan`` are the same class of commitment. The
#: remaining types are system or accounting entries (``Invoice``,
#: ``Credit memo``, ``Journal entry batch``, ``Reversal``, ...), ledger
#: corrections (``Write off``, ``Refund``, the ``Adjustment *`` family), cash
#: movements (``Deposit``, ``Payout``), non-cash dispositions (``Sold stock``,
#: ``Sold property``, ``Sold Gift In Kind``), or purchases (``Order``). None
#: are gifts that a correct roll-up counts. Since the vocabulary is a fixed
#: schema enumeration instead of an org-configurable list, an allowlist is
#: the safer default: an unrecognised type is dropped rather than summed, and
#: an org whose exports carry a received type this bridge does not yet know
#: can override with ``include_revenue_types=``. Matching ignores case,
#: spacing and punctuation, so one spelling here also matches ``"payment"``
#: and ``"  PAYMENT  "``.
DEFAULT_INCLUDED_REVENUE_TYPES = ("Payment",)

# Blackbaud CRM header normalisation, applied before the CiviCRM bridge's own.
# Case and punctuation are already collapsed ("Constituent ID" ->
# constituent_id), so this maps only the residue onto the canonical names.
# Anything absent here falls through to the CiviCRM canonicaliser, which
# already handles the labels the two systems happen to share ("Amount",
# "Email", "First Name"). Only labels and column names with a documented
# source are listed: the Extended Output Formats reference, the database
# schema, and the SKY API CreateRecurringGift payload.
_HEADER_ALIASES = {
    # Donor key: export label "Constituent ID" or "Constituent";
    # physical column CONSTITUENTID (FK to CONSTITUENT.ID); API field
    # constituent_id. CONSTITUENT.LOOKUPID is the human-facing unique
    # identifier and is also accepted.
    "constituent": "contact_id",
    "constituent_id": "contact_id",
    "constituentid": "contact_id",
    "lookup_id": "contact_id",
    "lookupid": "contact_id",
    "constituent_system_record_id": "contact_id",
    "constituent_lookup_id": "contact_id",
    # Date: export label "Date"; physical column DATE; API field date.
    "date": "receive_date",
    # Amount: export label "Amount"; physical column TRANSACTIONAMOUNT;
    # API field amount. The batch table (BATCHREVENUE) uses a plain AMOUNT
    # column, which the CiviCRM canonicaliser already handles.
    "amount": "total_amount",
    "transaction_amount": "total_amount",
    "transactionamount": "total_amount",
    # Revenue type: export label "Revenue Type"; physical column TYPECODE;
    # API field type_code. The computed TYPE label is the string form
    # ("Payment", "Pledge", ...). This is the structural transaction type the
    # allowlist filters on.
    "revenue_type": "gift_type",
    "type": "gift_type",
    # Application: export label "Application" or "Application type"; physical
    # column APPLICATIONCODE; API field application_code. The computed
    # APPLICATION label is the string form ("Donation", ...). Designation is
    # the specific fund the split is allocated to, exposed by the export as
    # "Designation" / "Designation ID" / "Designation name".
    "application": "financial_type",
    "application_type": "financial_type",
    # Designation: export label "Designation", "Designation ID",
    # "Designation name"; physical column DESIGNATIONID; API field
    # designation_id.
    "designation": "financial_type",
    "designation_id": "financial_type",
    "designation_name": "financial_type",
}

# Revenue-type matching key: strip everything but letters and digits so
# "Recurring gift", "RecurringGift" and "recurring_gift" collapse to one value.
_NON_ALNUM_ALL = re.compile(r"[^a-z0-9]+")


def blackbaud_crm_revenue_to_features(
    revenue: Union[Iterable[Mapping], pd.DataFrame],
    *,
    reference_date: Optional[Union[str, pd.Timestamp]] = None,
    include_revenue_types: Optional[Sequence[str]] = DEFAULT_INCLUDED_REVENUE_TYPES,
) -> pd.DataFrame:
    """Aggregate a Blackbaud CRM revenue export into donor-level features.

    Only rows whose transaction type is a documented received revenue type
    (``Payment`` by default) are kept, then handed to
    :func:`civicrm_contributions_to_features`, which produces the donor frame.
    Blackbaud CRM's revenue record carries no separate commitment flag: whether
    a row is a promise or a receipt is encoded entirely in its transaction
    type, so no other filter is applied.

    Parameters
    ----------
    revenue : iterable of mapping, or DataFrame
        Blackbaud CRM revenue rows, under the acknowledgement export's labels
        (``Constituent ID``, ``Revenue Type``, ``Date``, ``Amount``,
        ``Application type``, ``Designation name``), the underlying database
        column names (``CONSTITUENTID``, ``TYPE``, ``DATE``,
        ``TRANSACTIONAMOUNT``, ``APPLICATION```), or the
        constituent's ``LOOKUPID`` as an alternative donor key.
        ``contact_id``, ``receive_date`` and ``total_amount`` are required
        after normalisation. ``gift_type`` and ``financial_type`` are read
        from the transaction type and application columns when present; the
        export's ``Designation name`` label (from ``DESIGNATION.NAME`` via a
        join through ``REVENUESPLIT_EXT.DESIGNATIONID``) also maps to
        ``financial_type``. The canonicaliser will also pick up ``email``,
        ``first_name`` and ``last_name`` if the input frame happens to carry
        them, though the acknowledgement export does not write separate name
        columns (instead it writes ``Constituent name``).
    reference_date : str or datetime-like, optional
        Anchor for the recency features. If ``None``, the latest transaction
        date in the batch is used, which keeps the aggregation free of "now"
        leakage.
    include_revenue_types : sequence of str or None, default=:data:`DEFAULT_INCLUDED_REVENUE_TYPES`
        Revenue types to keep before aggregating, matched against the
        canonical ``gift_type`` column (populated from the export's
        ``Revenue Type`` label) ignoring case, spacing and punctuation. An
        allowlist is used rather than an exclusion set because Blackbaud
        CRM's ``TYPECODE`` is a fixed schema enumeration: a type this bridge
        does not recognise is dropped rather than summed, and an org whose
        exports carry a received type not in the default can pass it
        explicitly. ``None`` or an empty sequence disables the filter and
        sums **every** row, which double-counts a pledge against the
        payments that satisfy it. Rows whose transaction type is blank are
        kept either way: an unlabelled row cannot be shown to be a
        commitment.

    Returns
    -------
    features : pandas.DataFrame
        One row per donor, indexed by ``contact_id``, with the same columns
        :func:`civicrm_contributions_to_features` emits.
        ``distinct_financial_types`` counts distinct applications (or
        designations, whichever the export carries) here.

    Raises
    ------
    KeyError
        If ``contact_id``, ``receive_date`` or ``total_amount`` is absent
        after normalisation. In particular, an export that carries only a
        name column and no constituent ID raises here rather than silently
        merging donors who share a name.

    Warns
    -----
    UserWarning
        If ``include_revenue_types`` was requested but the export carries no
        revenue type column. Silently summing pledges together with their
        payments is exactly the error this bridge exists to prevent, so it is
        worth a warning rather than a quiet wrong total.

    Notes
    -----
    A Blackbaud CRM revenue record that has been split across multiple
    applications writes one ``REVENUESPLIT_EXT`` row per split, all sharing
    the parent revenue's ``ID``, and their amounts are meant to be summed.
    Nothing here deduplicates on a revenue id for that reason, so
    concatenating two exports that overlap in time will double-count the
    overlap; export disjoint date ranges.

    Examples
    --------
    >>> rows = [
    ...     {"Constituent ID": "88", "Date": "2025-01-10",
    ...      "Amount": "1200.00", "Revenue Type": "Pledge"},
    ...     {"Constituent ID": "88", "Date": "2025-02-10",
    ...      "Amount": "100.00", "Revenue Type": "Payment"},
    ...     {"Constituent ID": "88", "Date": "2025-03-10",
    ...      "Amount": "100.00", "Revenue Type": "Payment"},
    ... ]
    >>> feats = blackbaud_crm_revenue_to_features(rows)
    >>> float(feats.loc["88", "total_gift_amount"])  # not 1400.0
    200.0
    >>> int(feats.loc["88", "gift_count"])
    2
    """
    df = _normalise_headers(_to_frame(revenue), _canonical_blackbaud_crm)
    if df.empty:
        return _empty_feature_frame()

    missing = [col for col in _REQUIRED if col not in df.columns]
    if missing:
        raise KeyError(
            f"Blackbaud CRM revenue export is missing {missing}. Export the "
            f"'Constituent ID', 'Date' and 'Amount' fields (database columns: "
            f"CONSTITUENTID, DATE, TRANSACTIONAMOUNT); got {sorted(df.columns)}."
        )

    if include_revenue_types:
        if "gift_type" in df.columns:
            wanted = {_revenue_type_key(t) for t in include_revenue_types}
            keys = df["gift_type"].map(_revenue_type_key)
            df = df[(keys == "") | keys.isin(wanted)]
        else:
            warnings.warn(
                f"Blackbaud CRM revenue export has no revenue type "
                f"column, so include_revenue_types="
                f"{tuple(include_revenue_types)!r} could not be applied: "
                f"a pledge or a recurring gift schedule (if any) is summed "
                f"alongside the payments made against it, which counts every "
                f"committed dollar twice. Add the 'Revenue Type' field to the "
                f"export.",
                stacklevel=2,
            )

    if df.empty:
        return _empty_feature_frame()
    return civicrm_contributions_to_features(
        df, reference_date=reference_date, statuses=None
    )


def read_blackbaud_crm_revenue(path: Union[str, Path]) -> pd.DataFrame:
    """Read Blackbaud CRM revenue export CSV(s) into one frame.

    Accepts a single ``.csv`` or a directory of them, walked recursively and
    concatenated in sorted relative-path order; symlinks are not followed.
    Every column is read as text and nothing is filtered: **commitment rows
    are still present**, and it is :func:`blackbaud_crm_revenue_to_features`
    that drops them.

    Reading a revenue export is CRM-agnostic once the headers are normalised, so
    this delegates to :func:`read_civicrm_contributions` rather than repeating
    its BOM handling and path hardening. Blackbaud CRM headers are normalised
    on the way into :func:`blackbaud_crm_revenue_to_features`, not here.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV file, or a directory of them.

    Returns
    -------
    revenue : pandas.DataFrame
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
def _canonical_blackbaud_crm(header: str) -> str:
    key = _NON_ALNUM.sub("_", str(header).strip().lower()).strip("_")
    return _HEADER_ALIASES.get(key) or _canonical(header)


def _revenue_type_key(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return _NON_ALNUM_ALL.sub("", str(value).strip().lower())