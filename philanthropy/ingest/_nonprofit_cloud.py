"""
philanthropy.ingest._nonprofit_cloud
====================================
Bridge from a Salesforce Nonprofit Cloud (NPC) GiftTransaction export to a
PhilanthroPy donor-level feature table.

`Nonprofit Cloud <https://www.salesforce.com/nonprofit/>`_ is Salesforce's
newer nonprofit data model, distinct from the Nonprofit Success Pack (NPSP)
package the sibling :mod:`philanthropy.ingest._npsp` bridge reads. Where NPSP
folds a gift and its payment onto one ``Opportunity`` record distinguished by
stage, Nonprofit Cloud splits the two across separate objects:

* ``GiftCommitment`` (the promise): a pledge, recurring donation agreement,
  and grant agreement; it holds the total amount and a schedule.
* ``GiftTransaction`` (the money): one row per actual payment containing a
  one-time gift, one installment of a pledge, and one charge of a recurring
  donation.

This module reads ``GiftTransaction`` exports, not ``GiftCommitment``. The
amount received, the date received and the donor key all live on the
transaction. A ``GiftCommitment`` has no ``TransactionDate`` to aggregate by.

An export out of a report or the Data Loader carries either the report column
label (``Donor ID``, ``Transaction Date``, ``Current Amount``, ``Status``)
or the raw API field name (``DonorId``, ``TransactionDate``,
``CurrentAmount``, ``Status``). Both are accepted here and normalised onto the
canonical ``contact_id`` / ``receive_date`` / ``total_amount`` names.

:func:`read_nonprofit_cloud_gifts` loads the CSV(s);
:func:`nonprofit_cloud_gifts_to_features` only keeps paid transactions and hands
the surviving receipts to
:func:`~philanthropy.ingest.civicrm_contributions_to_features`, which already
knows how to roll a gift log up into the one-row-per-donor frame the
estimators consume.

**The trap this exists to prevent is the commitment versus payment split,
expressed through the GiftTransaction Status field.** When a ``GiftCommitment``
is created, Salesforce automatically and immediately generates the
``GiftTransaction`` records its schedule implies. Each carries a ``Status``
field that starts at ``Unpaid`` and only moves to ``Paid`` once the money
actually arrives. Salesforce's own Trailhead module is explicit: *"Gift
transactions automatically created by a gift commitment, for example, exist
with a Status value of Unpaid until you receive the payment."* Summing every
``GiftTransaction.CurrentAmount`` regardless of status therefore counts money
that has not yet been given. A $1,200 pledge paid monthly becomes twelve
``GiftTransaction`` rows the moment the commitment is saved, and the two that
have actually been received cannot be distinguished from the ten that have
not without reading ``Status``.

This is the same problem the NPSP bridge solves by excluding ``Pledged``
stage Opportunities, and the same problem the Raiser's Edge bridge solves by
excluding ``Pledge`` gift types. Here it is solved by an allowlist set on
``Status``, defaulting to ``("Paid",)``. See
:data:`DEFAULT_INCLUDED_GIFT_TRANSACTION_STATUSES`.

The allowlist is a documented default parameter rather than a constant
because Salesforce's GiftTransaction status vocabulary can be extended by an
org (custom statuses such as ``In Review`` are possible). The default matches
Salesforce's out-of-the-box lifecycle; an org with additional statuses should
pass ``include_statuses=`` explicitly. An allowlist is the safer default as a
status this module has never heard of is dropped rather than silently summed.
An unrecognized status is not proof that the money arrived.

**Donor key precedence:** Nonprofit Cloud keys a ``GiftTransaction`` to a
donor account (a person Account for individuals, a household Account, or an
organization Account) via ``DonorId``, and to the individual contact
associated with the transaction via ``ContactId``. ``DonorId`` points to the
Account DMO and is the primary donor reference; ``ContactId`` points to the
Individual DMO and is the contact associated with the gift, which may be a
household member or an organizational contact. When an export carries both,
``DonorId`` is used as the donor key regardless of column order. This is the
same precedence NPSP's bridge applies when both an Account and a Primary Contact
are present. See ``_prefer_donor_key`` below.

Notes
-----
``distinct_financial_types`` is not populated by this bridge. NPC's
designation data (the analog of NPSP's General Accounting Unit or Raiser's
Edge's Fund) lives on the related ``GiftTransactionDesignation`` and
``GiftDesignation`` objects, not as a direct field on ``GiftTransaction``. A
simple GiftTransaction CSV export does not include designation columns unless
the report is explicitly configured to join to them. Supporting that would
require a multi-object reader, which is out of scope for this bridge.

``GiftType`` on the transaction is a restricted picklist whose documented
values are ``Individual`` and ``Organizational``, describing the type of
donor, not the fund the gift is designated to. It is not mapped to
``financial_type`` and does not feed ``distinct_financial_types``.

Sources:

Salesforce, *Gift Transactions Overview*
(https://help.salesforce.com/s/articleView?id=sfdo.fundraising_gift_transactions_overview.htm),
which documents the full Status lifecycle (``Unpaid`` -> ``Paid``, ``Canceled``,
``Failed``, ``Fully Refunded``, ``Written Off``) and the rule that a commitment's
generated transactions begin at ``Unpaid``.

Trailhead, "Track Gift Transactions"
(https://trailhead.salesforce.com/content/learn/modules/nonprofit-cloud-fundraising-operations/track-gift-transactions),
which walks through a commitment generating unpaid transactions and marks them paid
on receipt.

Salesforce, *GiftTransaction Object Reference*
(https://developer.salesforce.com/docs/atlas.en-us.nonprofit_cloud.meta/nonprofit_cloud/npc_fundraising_api_objects_gifttransaction.htm),
which lists ``DonorId``, ``TransactionDate``, ``CurrentAmount``, ``GiftType`` and
``Status`` as the object's fields and documents ``GiftType``'s possible values
(``Individual``, ``Organizational``).

Salesforce, *Data Cloud GiftTransaction DLO Mappings*
(https://developer.salesforce.com/docs/data/data-cloud-dmo-mapping/guide/c360dm-gifttransaction_dmo_mappings.html),
which confirms the API field names a report export would surface.

Salesforce, *GiftTransaction DMO Object Reference*
(https://developer.salesforce.com/docs/data/data-cloud-dmo-mapping/guide/c360dm-si-gifttransactiondmo-dmo.html),
which documents the FOREIGNKEY relationship from ``DonorId`` to the Account
DMO and from ``ContactId`` to the Individual DMO. It proves the basis for the
DonorId-over-ContactId donor-key precedence this bridge applies.

Salesforce, *Fundraising Performance Insights Dashboard in Nonprofit*
(https://help.salesforce.com/s/articleView?id=sfdo.fundraising_fundraising_performance_insights_dashboard.htm&type=5),
which confirms the object fields used via its dashboard SQL commands.

Salesforce, *Gifts Transactions (POST)*
(https://developer.salesforce.com/docs/atlas.en-us.nonprofit_cloud.meta/nonprofit_cloud/connect_resources_gift_transaction_post.htm),
which documents the Connect API's treatment of the gift transaction object.

Salesforce, *Nonprofit Cloud UML Diagram*
(https://developer.salesforce.com/docs/resources/img/en-us/264.0?doc_id=nonprofit%2Ffundraising%2Fimages%2Fnpc_fundraising_datamodel.png&folder=nonprofit_cloud),
which shows the relationships between GiftTransaction, GiftCommitment, and
the donor Account.
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
    "DEFAULT_INCLUDED_GIFT_TRANSACTION_STATUSES",
    "nonprofit_cloud_gifts_to_features",
    "read_nonprofit_cloud_gifts",
]

#: GiftTransaction statuses included from the roll-up by default. Only
#: ``Paid`` is listed: a ``GiftTransaction`` created automatically from a
#: ``GiftCommitment`` schedule starts at ``Unpaid`` and only moves to ``Paid``
#: when the money is received, so an ``Unpaid`` row is a promise, not a gift.
#: Every other documented status (``Unpaid``, ``Canceled``, ``Failed``,
#: ``Fully Refunded``, ``Written Off``, ``Pending``) is not part of the default
#: inclusion set. Those represent money that was never received, received and
#: unwound, or unreceived--and none are gift transactions that a correct roll-up
#: should count. Override to add org-specific statuses (see
#: :func:`nonprofit_cloud_gifts_to_features`). Matching ignores case, spacing and
#: punctuation, so one spelling here also matches ``"PAID"`` and ``"  paid  "``.
DEFAULT_INCLUDED_GIFT_TRANSACTION_STATUSES = ("Paid",)

# Donor-key precedence: Nonprofit Cloud keys a GiftTransaction to a
# Donor (individual, household, or organization account that represents
# the donor) via DonorId. ContactId is secondary, pointing to the Contact
# (the individual) associated with the transaction, via ContactId.
# When an export carries both, prefer the Donor ID (see
# _prefer_donor_key below).
_DONOR_KEY_ALIASES = frozenset({"donor_id", "donorid"})
_CONTACT_KEY_ALIASES = frozenset({"contact_id", "contactid"})

# Nonprofit Cloud header normalisation, applied before the CiviCRM bridge's
# own. Case and punctuation are already collapsed ("Transaction Date" ->
# transaction_date), so this maps only the residue onto the canonical names.
# Anything absent here falls through to the CiviCRM canonicaliser, which
# already handles the labels the two systems happen to share ("Amount",
# "Email", "First Name").
_HEADER_ALIASES = {
    # Donor: report label "Donor ID", raw API field DonorId. ContactId is
    # the individual-level key associated with the transaction and is
    # accepted as a fallback.
    "donor_id": "contact_id",
    "donorid": "contact_id",
    "contact_id": "contact_id",
    "contactid": "contact_id",
    # Date: report label "Transaction Date", raw API field TransactionDate.
    "transaction_date": "receive_date",
    "transactiondate": "receive_date",
    # Amount: report label "Current Amount", raw API field CurrentAmount. The
    # "Current" prefix is material: GiftTransaction also carries
    # OriginalAmount (the pre-adjustment figure) and RefundedAmount (what has
    # been returned), and CurrentAmount is the net that Salesforce's own
    # reports aggregate by.
    "current_amount": "total_amount",
    "currentamount": "total_amount",
    # Gift type: report label "Gift Type", raw API field GiftType.
    # NPC's GiftType is a restricted picklist (Individual, Organizational). It
    # describes the type of donor, not the fund the gift is designated to. It
    # is not mapped to `financial_type` and does not feed `distinct_financial_types`
    # (see the module docstring's Notes section).
    "gift_type": "gift_type",
    "gifttype": "gift_type",
    # Status: the column this module exists to read. Mapped to the neutral
    # `status` canonical name rather than to `gift_type`, because NPC carries
    # both a GiftType (Individual, Organizational) and a Status (Unpaid, Paid,
    # ...), and collapsing them onto one column would lose information the
    # inclusion filter needs to act on separately.
    "status": "status",
    "gift_transaction_status": "status",
    "gifttransactionstatus": "status",
}

# Status-matching key: strip everything but letters and digits so "Paid"
# and " PAID " collapse to one value, matching the NPSP bridge's
# stage-matching and Raiser's Edge bridge's gift-type matching.
_NON_ALNUM_ALL = re.compile(r"[^a-z0-9]+")


def nonprofit_cloud_gifts_to_features(
    gifts: Union[Iterable[Mapping], pd.DataFrame],
    *,
    reference_date: Optional[Union[str, pd.Timestamp]] = None,
    include_statuses: Optional[Sequence[str]] = DEFAULT_INCLUDED_GIFT_TRANSACTION_STATUSES
) -> pd.DataFrame:
    """Aggregate a Nonprofit Cloud GiftTransaction export into donor features.

    Transactions whose status is not a documented received status (``Paid``) are
    dropped first, then the surviving receipts are handed to
    :func:`civicrm_contributions_to_features`, which produces the donor frame.
    Nonprofit Cloud's GiftTransaction carries no separate pledge flag beyond
    ``Status``, so no other filter is applied.

    Parameters
    ----------
    gifts : iterable of mapping, or DataFrame
        Nonprofit Cloud ``GiftTransaction`` rows, under a report's column
        labels (``Donor ID``, ``Transaction Date``, ``Current Amount``,
        ``Status``) or the raw API field names (``DonorId``,
        ``TransactionDate``, ``CurrentAmount``, ``Status``). ``contact_id``,
        ``receive_date`` and ``total_amount`` are required after
        normalisation; ``gift_type``, ``status``, ``email``, ``first_name``
        and ``last_name`` are used when present. If both a Donor column
        (``DonorId``) and a Contact column (``ContactId``) are present, the
        Donor ID is used as the donor key. The Account / Party reference donor
        is the aggregation unit, and the individual contact reference is a
        transaction-level fallback used only when the Donor is absent.
    reference_date : str or datetime-like, optional
        Anchor for the recency features. If ``None``, the latest
        Transaction Date in the batch is used, which keeps the aggregation
        free of "now" leakage.
    include_statuses : sequence of str or None, default=:data:`DEFAULT_INCLUDED_GIFT_TRANSACTION_STATUSES`
        Statuses to keep before aggregating, matched against ``status``
        ignoring case, spacing and punctuation. ``None`` or an empty sequence
        disables the filter and sums **every** row, which double-counts a
        commitment's full schedule against the receipts that have actually
        landed: a ``GiftCommitment`` for a monthly pledge generates every
        scheduled ``GiftTransaction`` up front, and the two that have been
        received cannot be told apart from the ten that have not without
        reading Status. Rows whose status is blank are kept either way: an
        unlabelled row cannot be shown to be an expected-but-unpaid
        commitment. A partial refund does not change a transaction's ``Status``
        (it stays ``Paid``), so a partially refunded gift is kept by default
        and contributes its net ``CurrentAmount``; only a full refund moves the
        row to ``Fully Refunded``, which the default drops.

    Returns
    -------
    features : pandas.DataFrame
        One row per donor, indexed by ``contact_id``, with the same columns
        :func:`civicrm_contributions_to_features` emits.
        ``distinct_financial_types`` is not populated by this bridge (see Notes).

    Raises
    ------
    KeyError
        If ``contact_id``, ``receive_date`` or ``total_amount`` is absent
        after normalisation.

    Warns
    -----
    UserWarning
        If ``include_statuses`` was requested but the export carries no
        Status column. Silently summing a commitment's unpaid and other non-paid
        schedules together with the receipts that have actually landed is exactly
        the error this bridge exists to prevent, so it is worth a warning rather
        than a quiet wrong total.

    Notes
    -----
    Nothing here deduplicates on a GiftTransaction id, so concatenating two exports
    that overlap in time will double-count the overlap; export disjoint date ranges.

    Examples
    --------
    >>> rows = [
    ...     {"Donor ID": "88", "Transaction Date": "2025-01-10",
    ...      "Current Amount": "100.00", "Status": "Paid"},
    ...     {"Donor ID": "88", "Transaction Date": "2025-02-10",
    ...      "Current Amount": "100.00", "Status": "Unpaid"},
    ...     {"Donor ID": "88", "Transaction Date": "2025-03-10",
    ...      "Current Amount": "100.00", "Status": "Paid"},
    ... ]
    >>> feats = nonprofit_cloud_gifts_to_features(rows)
    >>> float(feats.loc["88", "total_gift_amount"])  # not 300.0
    200.0
    >>> int(feats.loc["88", "gift_count"])
    2
    """
    df = _normalise_headers(_prefer_donor_key(_to_frame(gifts)), _canonical_nonprofit_cloud)
    if df.empty:
        return _empty_feature_frame()

    missing = [col for col in _REQUIRED if col not in df.columns]
    if missing:
        raise KeyError(
            f"Nonprofit Cloud GiftTransaction export is missing {missing}. "
            f"Export the 'Donor ID' (or 'Contact ID'), 'Transaction Date' "
            f"and 'Current Amount' fields (Salesforce API: DonorId, "
            f"TransactionDate, CurrentAmount); got {sorted(df.columns)}."
        )

    if include_statuses:
        if "status" in df.columns:
            wanted = {_status_key(s) for s in include_statuses}
            keys = df["status"].map(_status_key)
            df = df[(keys == "") | keys.isin(wanted)]
        else:
            warnings.warn(
                f"Nonprofit Cloud GiftTransaction export has no Status column, "
                f"so include_statuses={tuple(include_statuses)!r} could not be "
                f"applied: a GiftCommitment's unpaid and non-paid schedules (if "
                f"present) are summed alongside the receipts that have actually "
                f"landed, which counts money that has not yet been given. Add "
                f"the 'Status' field to the export.",
                stacklevel=2,
            )

    if df.empty:
        return _empty_feature_frame()
    return civicrm_contributions_to_features(
        df, reference_date=reference_date, statuses=None
    )


def read_nonprofit_cloud_gifts(path: Union[str, Path]) -> pd.DataFrame:
    """Read Nonprofit Cloud GiftTransaction export CSV(s) into one frame.

    Accepts a single ``.csv`` or a directory of them, walked recursively and
    concatenated in sorted relative-path order; symlinks are not followed.
    Every column is read as text and nothing is filtered: **unpaid and other
    non-paid transactions are still present**, and it is
    :func:`nonprofit_cloud_gifts_to_features` that drops them.

    Reading a gift export is CRM-agnostic once the headers are normalised,
    so this delegates to :func:`read_civicrm_contributions` rather than
    repeating its BOM handling and path hardening. Nonprofit Cloud headers
    are normalised on the way into
    :func:`nonprofit_cloud_gifts_to_features`, not here.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV file, or a directory of them.

    Returns
    -------
    gifts : pandas.DataFrame
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
def _canonical_nonprofit_cloud(header: str) -> str:
    key = _NON_ALNUM.sub("_", str(header).strip().lower()).strip("_")
    return _HEADER_ALIASES.get(key) or _canonical(header)


def _prefer_donor_key(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve which column is the donor key when an export has both.

    Nonprofit Cloud keys a ``GiftTransaction`` to the donor account (a person
    Account, household Account, or an organization Account) via ``DonorId`` and
    to the Contact associated with the transaction via ``ContactId``. For the
    donor-level aggregation this frame produces, the donor account is the intended
    key. Without this, an export carrying both columns would have them collapse onto
    the same ``contact_id`` name and ``_normalise_headers`` would silently keep
    whichever one happens to come first in the export's column order. Drop the Contact
    column(s) first instead, so the donor key is always the Donor when both are present.
    """
    if df.empty:
        return df
    raw_keys = {c: _NON_ALNUM.sub("_", str(c).strip().lower()).strip("_") for c in df.columns}
    has_donor = any(k in _DONOR_KEY_ALIASES for k in raw_keys.values())
    has_contact = any(k in _CONTACT_KEY_ALIASES for k in raw_keys.values())
    if has_donor and has_contact:
        drop_cols = [c for c, k in raw_keys.items() if k in _CONTACT_KEY_ALIASES]
        df = df.drop(columns=drop_cols)
    return df


def _status_key(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return _NON_ALNUM_ALL.sub("", str(value).strip().lower())