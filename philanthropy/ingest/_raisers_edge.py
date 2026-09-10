"""
philanthropy.ingest._raisers_edge
=================================
Bridge from a Blackbaud Raiser's Edge gift export to a PhilanthroPy
donor-level feature table.

`The Raiser's Edge <https://www.blackbaud.com>`_ is the CRM most academic
medical centres and large nonprofits run their advancement shop on, and a gift
export out of it is the file a prospect researcher actually has to hand.  Like
CiviCRM it surfaces the same table under more than one spelling: the desktop
Export module writes human labels (``Constituent ID``, ``Gift Date``, ``Gift
Amount``, ``Gift Type``), the underlying database columns are terser
(``CONSTIT_ID``, ``DTE``, ``TYPE``), and the RE NXT SKY API returns
``constituent_id`` / ``date`` / ``amount`` / ``type``.  All three are accepted
here and normalised onto the canonical ``contact_id`` / ``receive_date`` /
``total_amount`` names.

:func:`read_raisers_edge_gifts` loads the CSV(s);
:func:`raisers_edge_gifts_to_features` drops the commitment rows and hands the
remaining payments to :func:`~philanthropy.ingest.civicrm_contributions_to_features`,
which already knows how to roll a gift log up into the one-row-per-donor frame
the estimators consume.

**The trap this exists to prevent is commitment versus payment.** In Raiser's
Edge a pledge and the money paid against it are *separate gift records*.  The
Gift Records Guide is explicit: "When you receive a pledge payment from the
constituent, you must create a separate gift record for the pledge payment and
apply the payment toward the pledge to reduce the balance."  A ``Pledge`` row's
``Pledge amt`` is "the total amount the constituent agreed to give", so summing
gift amounts naively counts every pledged dollar twice, once as the promise and
again as each payment.  A ``Recurring Gift`` row is worse: it is a template, and
its ``Amount`` is "the amount the constituent wants to pay at a recurring
interval", not a sum ever received.  Both are excluded by default; the payment
rows they generate (``Pay-Cash``, ``PledgePayment``, ``RecurringGiftPayment``,
``MG Pay-Cash``, ...) are kept, because those are the money.

Raiser's Edge exports are user-configured, so the excluded set is a documented
default parameter rather than a constant: see
:data:`DEFAULT_EXCLUDED_GIFT_TYPES`.

Sources: Blackbaud, *The Raiser's Edge Gift Records Guide*
(https://help.blackbaud.com/docs/0/assets/guides/re/gifts.pdf), "Understanding
Gift Types" and "Gift Types and Subtypes"; Blackbaud Knowledgebase article
46484, "What are gift types?".
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
    "DEFAULT_EXCLUDED_GIFT_TYPES",
    "raisers_edge_gifts_to_features",
    "read_raisers_edge_gifts",
]

#: Gift types excluded from the roll-up by default: the commitment rows, whose
#: amount is a promise rather than money received, and the ledger corrections,
#: which are not gifts at all. Matching ignores case, spacing and punctuation,
#: so one spelling here covers every dialect the same type is written in:
#: ``"Recurring Gift"`` also matches the SKY API's ``RecurringGift`` and a
#: hand-renamed ``recurring_gift``. Payment rows are *not* in this set and are
#: what the features are built from.
DEFAULT_EXCLUDED_GIFT_TYPES = (
    "Pledge",
    "Matching Gift Pledge",
    # The desktop abbreviates the matching-gift types, and the abbreviation is
    # not punctuation away from the long form, so it needs its own entry. It is
    # the commitment half of the "MG Pay-Cash" the payment set keeps.
    "MG Pledge",
    "Recurring Gift",
    "Amendment",
    "Adjustment",
    "General Ledger Reversal",
    "Write Off",
    "Pledge Write Off",
    "Matching Gift Write Off",
    "MG Write Off",
)

# Raiser's Edge header normalisation, applied before the CiviCRM bridge's own.
# Case and punctuation are already collapsed ("Gift Date" -> gift_date), so this
# maps only the residue onto the canonical names. Anything absent here falls
# through to the CiviCRM canonicaliser, which already handles the labels the two
# systems happen to share ("Amount", "Email", "First Name").
_HEADER_ALIASES = {
    # Constituent key: export label, RE7 database column, SKY API field.
    "constituent_id": "contact_id",
    "constituent_system_record_id": "contact_id",
    "constituent_lookup_id": "contact_id",
    "constit_id": "contact_id",
    "donor_id": "contact_id",
    # Date. The guide renames this field per gift type: a Pledge row carries
    # "Pledged on" where a Cash row carries "Gift date".
    "gift_date": "receive_date",
    "pledged_on": "receive_date",
    "date": "receive_date",
    "dte": "receive_date",
    # Amount, renamed per gift type the same way: "Pledge amt" on a Pledge,
    # "Value" on a Gift-in-Kind or Stock/Property.
    "gift_amount": "total_amount",
    "pledge_amt": "total_amount",
    "pledge_amount": "total_amount",
    "value": "total_amount",
    # Gift type: the column this module exists to read.
    "type": "gift_type",
    "gift_type_name": "gift_type",
    # Fund is the Raiser's Edge analogue of CiviCRM's Financial Type: the GL
    # classification a gift is designated to. It feeds distinct_financial_types.
    "fund": "financial_type",
    "fund_description": "financial_type",
    "fund_id": "financial_type",
}

# Gift-type matching key: strip everything but letters and digits so
# "Recurring Gift", "RecurringGift" and "recurring_gift" collapse to one value.
_NON_ALNUM_ALL = re.compile(r"[^a-z0-9]+")


def raisers_edge_gifts_to_features(
    gifts: Union[Iterable[Mapping], pd.DataFrame],
    *,
    reference_date: Optional[Union[str, pd.Timestamp]] = None,
    exclude_gift_types: Optional[Sequence[str]] = DEFAULT_EXCLUDED_GIFT_TYPES,
) -> pd.DataFrame:
    """Aggregate a Raiser's Edge gift export into donor-level features.

    Commitment rows (pledges, matching gift pledges, recurring gift templates)
    and ledger corrections are dropped first, then the surviving payment rows
    are handed to :func:`civicrm_contributions_to_features`, which produces the
    donor frame. Raiser's Edge has no contribution-status column, so no status
    filter is applied.

    Parameters
    ----------
    gifts : iterable of mapping, or DataFrame
        Raiser's Edge gift rows, under the desktop export labels
        (``Constituent ID``, ``Gift Date``, ``Gift Amount``, ``Gift Type``), the
        RE7 database columns (``CONSTIT_ID``, ``DTE``, ``TYPE``) or the RE NXT
        SKY API field names (``constituent_id``, ``date``, ``amount``,
        ``type``). ``contact_id``, ``receive_date`` and ``total_amount`` are
        required after normalisation; ``gift_type``, ``fund``, ``email``,
        ``first_name`` and ``last_name`` are used when present.
    reference_date : str or datetime-like, optional
        Anchor for the recency features. If ``None``, the latest gift date in
        the batch is used, which keeps the aggregation free of "now" leakage.
    exclude_gift_types : sequence of str or None, default=:data:`DEFAULT_EXCLUDED_GIFT_TYPES`
        Gift types to drop before aggregating, matched against ``gift_type``
        ignoring case, spacing and punctuation. ``None`` or an empty sequence
        disables the filter and sums **every** row, which double-counts pledged
        dollars. Rows whose gift type is blank are kept either way: an unlabelled
        row cannot be shown to be a commitment.

    Returns
    -------
    features : pandas.DataFrame
        One row per donor, indexed by ``contact_id``, with the same columns
        :func:`civicrm_contributions_to_features` emits.
        ``distinct_financial_types`` counts distinct Funds here.

    Raises
    ------
    KeyError
        If ``contact_id``, ``receive_date`` or ``total_amount`` is absent after
        normalisation.

    Warns
    -----
    UserWarning
        If ``exclude_gift_types`` was requested but the export carries no gift
        type column. Silently summing pledges together with their payments is
        exactly the error this bridge exists to prevent, so it is worth a
        warning rather than a quiet wrong total.

    Notes
    -----
    A Raiser's Edge export of *split* gifts writes one row per split, all
    sharing the gift's system record id, and their amounts are meant to be
    summed. Nothing here deduplicates on a gift id for that reason, so
    concatenating two exports that overlap in time will double-count the
    overlap; export disjoint date ranges.

    Examples
    --------
    >>> rows = [
    ...     {"Constituent ID": "88", "Gift Date": "2025-01-10",
    ...      "Gift Amount": "1200.00", "Gift Type": "Pledge"},
    ...     {"Constituent ID": "88", "Gift Date": "2025-02-10",
    ...      "Gift Amount": "100.00", "Gift Type": "Pay-Cash"},
    ...     {"Constituent ID": "88", "Gift Date": "2025-03-10",
    ...      "Gift Amount": "100.00", "Gift Type": "Pay-Cash"},
    ... ]
    >>> feats = raisers_edge_gifts_to_features(rows)
    >>> float(feats.loc["88", "total_gift_amount"])  # not 1400.0
    200.0
    >>> int(feats.loc["88", "gift_count"])
    2
    """
    df = _normalise_headers(_to_frame(gifts), _canonical_raisers_edge)
    if df.empty:
        return _empty_feature_frame()

    missing = [col for col in _REQUIRED if col not in df.columns]
    if missing:
        raise KeyError(
            f"Raiser's Edge gift export is missing {missing}. Export the "
            f"'Constituent ID', 'Gift Date' and 'Gift Amount' fields (SKY API: "
            f"constituent_id, date, amount); got {sorted(df.columns)}."
        )

    if exclude_gift_types:
        if "gift_type" in df.columns:
            unwanted = {_gift_type_key(t) for t in exclude_gift_types}
            keys = df["gift_type"].map(_gift_type_key)
            df = df[~keys.isin(unwanted)]
        else:
            warnings.warn(
                f"Raiser's Edge gift export has no gift type column, so "
                f"exclude_gift_types={tuple(exclude_gift_types)!r} could not be "
                f"applied: pledges and recurring gift templates (if any) are "
                f"summed alongside the payments made against them, which counts "
                f"every committed dollar twice. Add the 'Gift Type' field to the "
                f"export.",
                stacklevel=2,
            )

    if df.empty:
        return _empty_feature_frame()
    return civicrm_contributions_to_features(
        df, reference_date=reference_date, statuses=None
    )


def read_raisers_edge_gifts(path: Union[str, Path]) -> pd.DataFrame:
    """Read Raiser's Edge gift export CSV(s) into one frame.

    Accepts a single ``.csv`` or a directory of them, walked recursively and
    concatenated in sorted relative-path order; symlinks are not followed. Every
    column is read as text and nothing is filtered: **commitment rows are still
    present**, and it is :func:`raisers_edge_gifts_to_features` that drops them.

    Reading a gift export is CRM-agnostic once the headers are normalised, so
    this delegates to :func:`read_civicrm_contributions` rather than repeating
    its BOM handling and path hardening. Raiser's Edge headers are normalised on
    the way into :func:`raisers_edge_gifts_to_features`, not here.

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
def _canonical_raisers_edge(header: str) -> str:
    key = _NON_ALNUM.sub("_", str(header).strip().lower()).strip("_")
    return _HEADER_ALIASES.get(key) or _canonical(header)


def _gift_type_key(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return _NON_ALNUM_ALL.sub("", str(value).strip().lower())
