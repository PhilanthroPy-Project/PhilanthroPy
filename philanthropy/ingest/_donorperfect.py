"""
philanthropy.ingest._donorperfect
==================================
Bridge from a DonorPerfect gift export to a PhilanthroPy donor-level feature
table.

`DonorPerfect <https://www.donorperfect.com>`_ is a CRM common among small
and mid-sized nonprofits. Its ``dp_savegift`` XML API and its underlying gift
table key a gift to the donor via ``donor_id`` / ``DONOR_ID``, with
``gift_date``, ``amount`` and a ``record_type`` field; a report or "Gift/
Pledge Transactions" export out of the product instead carries the human
label ``Donor ID``. Both are accepted here and normalised onto the canonical
``contact_id`` / ``receive_date`` / ``total_amount`` names.

:func:`read_donorperfect_gifts` loads the CSV(s);
:func:`donorperfect_gifts_to_features` drops the commitment and split-summary
rows and hands the remaining gift rows to
:func:`~philanthropy.ingest.civicrm_contributions_to_features`, which already
knows how to roll a gift log up into the one-row-per-donor frame the
estimators consume.

**The trap this exists to prevent is commitment versus payment, but
DonorPerfect's data model splits that signal differently from Raiser's Edge,
NPSP or Bloomerang.** Those three each write the commitment-versus-payment
distinction into one type-like column. DonorPerfect instead uses
``record_type`` for the gift's *structural* role, separately from its own
``gift_type`` field, which is a payment method / thank-you-letter
descriptor (e.g. "Cash", "Check") unrelated to whether money changed hands.
The API documentation is explicit: ``record_type`` is set to ``'G'`` "for a
regular gift or for an individual split gift entry within a split gift
(i.e.; not the Main top level split)", ``'P'`` "for Pledge", or ``'M'`` "for
the Main gift in a split gift". A pledge's own record (``'P'``) carries the
promised total, not money received, exactly like a Raiser's Edge ``Pledge``
or an NPSP ``Pledged`` Opportunity; a payment against it is a separate
``'G'`` gift record (with a ``pledge_payment`` flag and a ``plink`` back to
the pledge's ``gift_id``). A split gift's own ``'M'`` row is the same trap
in a different shape: it is the split total, and its ``'G'`` split entries
(linked back to it via ``glink``) are the same money recorded again. Summing
every record regardless of ``record_type`` therefore double-counts a pledged
dollar (as the promise and as its payments) and double-counts a split gift
(as the total and as its parts).

The fix is the same exclusion-by-default shape as the other bridges: only
``record_type`` values ``P`` (Pledge) and ``M`` (split-gift Main total) are
dropped; ``G`` (a regular gift, a pledge payment, or a split entry) is kept,
because those are the money. DonorPerfect exports are user-configured like
the others, so the excluded set is a documented default parameter rather
than a constant: see :data:`DEFAULT_EXCLUDED_RECORD_TYPES`.

Sources: SofterWare, *DonorPerfect Online XML API Documentation* (Version
7.2, June 19, 2026)
(https://softerware-uploads.s3.amazonaws.com/community/donorperfect/DPO_XML_API_Documentation.pdf),
the ``dp_savegift`` procedure's ``@donor_id``, ``@gift_date``, ``@amount``,
``@record_type`` (values ``G``, ``P``, ``M``, and elsewhere ``S`` for soft
credit) and ``@gift_type`` parameters, and its "Split Gifts" and "Pledge
Notes" sections describing the Main/split and Pledge/payment relationships;
DonorPerfect Software, *Export the Data You Want* and *Financial Reporting*
webinar transcripts (https://www.donorperfect.com/video/export-the-data-you-want/,
https://www.donorperfect.com/video/financial-reporting/), which name
"Gift/Pledge Transactions" as an export/listing type and the gift type
field as a payment-method descriptor in transaction listings.
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
    "DEFAULT_EXCLUDED_RECORD_TYPES",
    "donorperfect_gifts_to_features",
    "read_donorperfect_gifts",
]

#: Record types excluded from the roll-up by default: ``P`` (Pledge), whose
#: amount is a promise rather than money received, and ``M`` (the Main gift
#: of a split gift), whose amount is the same money its own splits already
#: record. Matching ignores case, spacing and punctuation. ``G`` (a regular
#: gift, a pledge payment, or a split gift entry) is *not* in this set and is
#: what the features are built from.
DEFAULT_EXCLUDED_RECORD_TYPES = (
    "P",
    "M",
)

# DonorPerfect header normalisation, applied before the CiviCRM bridge's own.
# Case and punctuation are already collapsed ("Donor ID" -> donor_id), so
# this maps only the residue onto the canonical names. Anything absent here
# falls through to the CiviCRM canonicaliser, which already handles the
# labels the two systems happen to share ("Amount", "Email", "First Name").
_HEADER_ALIASES = {
    # Donor key: export/report label, then DB column and XML API field.
    "donor_id": "contact_id",
    "donorid": "contact_id",
    # Date: shared by the export label and the DB column/API field.
    "gift_date": "receive_date",
    "giftdate": "receive_date",
    # Amount: DonorPerfect's own field is just "amount"; already handled by
    # the CiviCRM canonicaliser this falls through to, so no entry needed,
    # but the export/report label is spelled out.
    "gift_amount": "total_amount",
    "giftamount": "total_amount",
    # Record type: the field this module exists to read. Not the same as
    # DonorPerfect's own "gift_type" (a payment-method descriptor), so that
    # name is deliberately not aliased here.
    "record_type": "gift_type",
    "recordtype": "gift_type",
    # GL code: DonorPerfect's fund-like designation, the analogue of
    # CiviCRM's Financial Type. Feeds distinct_financial_types.
    "gl_code": "financial_type",
    "glcode": "financial_type",
}

# Record-type matching key: strip everything but letters and digits so
# "P", "p" and " P " collapse to one value.
_NON_ALNUM_ALL = re.compile(r"[^a-z0-9]+")


def donorperfect_gifts_to_features(
    gifts: Union[Iterable[Mapping], pd.DataFrame],
    *,
    reference_date: Optional[Union[str, pd.Timestamp]] = None,
    exclude_record_types: Optional[Sequence[str]] = DEFAULT_EXCLUDED_RECORD_TYPES,
) -> pd.DataFrame:
    """Aggregate a DonorPerfect gift export into donor-level features.

    Pledge records (``record_type='P'``) and split-gift Main totals
    (``record_type='M'``) are dropped first, then the surviving gift rows
    (regular gifts, pledge payments, and split gift entries, all
    ``record_type='G'``) are handed to :func:`civicrm_contributions_to_features`,
    which produces the donor frame. DonorPerfect has no contribution-status
    column analogous to CiviCRM's, so no status filter is applied.

    Parameters
    ----------
    gifts : iterable of mapping, or DataFrame
        DonorPerfect gift rows, under a "Gift/Pledge Transactions" export
        label (``Donor ID``, ``Gift Date``, ``Gift Amount``, ``Record
        Type``) or the DB column / XML API field names (``DONOR_ID``,
        ``GIFT_DATE``, ``AMOUNT``, ``RECORD_TYPE``). ``contact_id``,
        ``receive_date`` and ``total_amount`` are required after
        normalisation; ``gift_type``, ``financial_type``, ``email``,
        ``first_name`` and ``last_name`` are used when present.
    reference_date : str or datetime-like, optional
        Anchor for the recency features. If ``None``, the latest gift date
        in the batch is used, which keeps the aggregation free of "now"
        leakage.
    exclude_record_types : sequence of str or None, default=:data:`DEFAULT_EXCLUDED_RECORD_TYPES`
        Record types to drop before aggregating, matched against
        ``gift_type`` ignoring case, spacing and punctuation. ``None`` or an
        empty sequence disables the filter and sums **every** row, which
        double-counts pledged dollars and split-gift totals. Rows whose
        record type is blank are kept either way: an unlabelled row cannot
        be shown to be a commitment.

    Returns
    -------
    features : pandas.DataFrame
        One row per donor, indexed by ``contact_id``, with the same columns
        :func:`civicrm_contributions_to_features` emits.
        ``distinct_financial_types`` counts distinct GL codes here.

    Raises
    ------
    KeyError
        If ``contact_id``, ``receive_date`` or ``total_amount`` is absent
        after normalisation.

    Warns
    -----
    UserWarning
        If ``exclude_record_types`` was requested but the export carries no
        record type column. Silently summing a pledge together with its
        payments (or a split gift's Main total together with its splits) is
        exactly the error this bridge exists to prevent, so it is worth a
        warning rather than a quiet wrong total.

    Notes
    -----
    Nothing here deduplicates on a gift id, so concatenating two exports
    that overlap in time will double-count the overlap; export disjoint
    date ranges. Soft credits (``record_type='S'``) are out of scope for
    this filter and are summed if present; DonorPerfect creates a soft
    credit under a *different* donor than the one who actually gave, so
    including them attributes the same dollar to two donors by design of
    that feature, not a bug this bridge is meant to catch.

    Examples
    --------
    >>> rows = [
    ...     {"Donor ID": "88", "Gift Date": "2025-01-10",
    ...      "Gift Amount": "1200.00", "Record Type": "P"},
    ...     {"Donor ID": "88", "Gift Date": "2025-02-10",
    ...      "Gift Amount": "100.00", "Record Type": "G"},
    ...     {"Donor ID": "88", "Gift Date": "2025-03-10",
    ...      "Gift Amount": "100.00", "Record Type": "G"},
    ... ]
    >>> feats = donorperfect_gifts_to_features(rows)
    >>> float(feats.loc["88", "total_gift_amount"])  # not 1400.0
    200.0
    >>> int(feats.loc["88", "gift_count"])
    2
    """
    df = _normalise_headers(_to_frame(gifts), _canonical_donorperfect)
    if df.empty:
        return _empty_feature_frame()

    missing = [col for col in _REQUIRED if col not in df.columns]
    if missing:
        raise KeyError(
            f"DonorPerfect gift export is missing {missing}. Export the "
            f"'Donor ID', 'Gift Date' and 'Gift Amount' fields (DB/API: "
            f"DONOR_ID, GIFT_DATE, AMOUNT); got {sorted(df.columns)}."
        )

    if exclude_record_types:
        if "gift_type" in df.columns:
            unwanted = {_record_type_key(t) for t in exclude_record_types}
            keys = df["gift_type"].map(_record_type_key)
            df = df[~keys.isin(unwanted)]
        else:
            warnings.warn(
                f"DonorPerfect gift export has no record type column, so "
                f"exclude_record_types={tuple(exclude_record_types)!r} could "
                f"not be applied: pledges and split-gift Main totals (if "
                f"any) are summed alongside the payments/splits made "
                f"against them, which counts every committed or split "
                f"dollar twice. Add the 'Record Type' field to the export.",
                stacklevel=2,
            )

    if df.empty:
        return _empty_feature_frame()
    return civicrm_contributions_to_features(
        df, reference_date=reference_date, statuses=None
    )


def read_donorperfect_gifts(path: Union[str, Path]) -> pd.DataFrame:
    """Read DonorPerfect gift export CSV(s) into one frame.

    Accepts a single ``.csv`` or a directory of them, walked recursively and
    concatenated in sorted relative-path order; symlinks are not followed.
    Every column is read as text and nothing is filtered: **pledge and
    split-Main rows are still present**, and it is
    :func:`donorperfect_gifts_to_features` that drops them.

    Reading a gift export is CRM-agnostic once the headers are normalised, so
    this delegates to :func:`read_civicrm_contributions` rather than
    repeating its BOM handling and path hardening. DonorPerfect headers are
    normalised on the way into :func:`donorperfect_gifts_to_features`, not
    here.

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
def _canonical_donorperfect(header: str) -> str:
    key = _NON_ALNUM.sub("_", str(header).strip().lower()).strip("_")
    return _HEADER_ALIASES.get(key) or _canonical(header)


def _record_type_key(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return _NON_ALNUM_ALL.sub("", str(value).strip().lower())
