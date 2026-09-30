"""
philanthropy.ingest._npsp
==========================
Bridge from a Salesforce Nonprofit Success Pack (NPSP) Opportunity export to a
PhilanthroPy donor-level feature table.

`NPSP <https://www.salesforce.org/nonprofit/nonprofit-success-pack/>`_ is the
Salesforce managed package most nonprofits running on Salesforce build their
donor database on, and the ``Opportunity`` object is where a gift lives: a
donation is an Opportunity record, keyed to the donor's ``Account`` (NPSP's
default Household Account model) or its ``Primary Contact``, with ``Amount``,
``CloseDate`` and ``StageName``. An export out of a report or the Data Loader
therefore carries the report column label (``Account Name``, ``Amount``,
``Close Date``, ``Stage``), the raw API field name (``AccountId``,
``CloseDate``, ``StageName``), or NPSP's own Data Import template header
(``Donation Amount``, ``Donation Date``, ``Donation Stage``). All three are
accepted here and normalised onto the canonical ``contact_id`` /
``receive_date`` / ``total_amount`` names.

:func:`read_npsp_opportunities` loads the CSV(s);
:func:`npsp_opportunities_to_features` keeps only the closed/won rows and
hands them to :func:`~philanthropy.ingest.civicrm_contributions_to_features`,
which already knows how to roll a gift log up into the one-row-per-donor frame
the estimators consume.

**The trap this exists to prevent is the same commitment-versus-payment split
Raiser's Edge writes as separate gift records, expressed instead through
Opportunity stage.** An Opportunity's ``StageName`` is one of an org-defined
sales process's stages, and only some of them mean the money actually
arrived. NPSP's Recurring Donations feature, for instance, creates one
Opportunity per instalment; the upcoming instalment is created with
``StageName`` ``Pledged`` (the "we expect this money" stage NPSP selects
automatically), and is only moved to a closed/won stage such as ``Closed Won``
once the gift is actually received. Every sales process also carries open
pipeline stages (``Prospecting``, ``Qualification``, ...) that are
cultivation work, not gifts, and a ``Closed Lost`` stage for an Opportunity
that never closed. Summing every Opportunity's ``Amount`` regardless of stage
therefore both double-counts a ``Pledged`` instalment already recorded again
as ``Closed Won``, and counts pipeline that was never given at all.

The fix is an allowlist, not a blocklist: only rows whose stage is a
documented closed/won equivalent are counted, so an org's own open- or
lost-stage names (which this module cannot enumerate) are excluded by
default instead of silently summed. NPSP's Donation and Major Gift record
types close won at ``Closed Won``; the Grant record type closes won at
``Awarded``; and ``Posted`` is the closed/won stage name a payment
processor's NPSP integration (e.g. Click & Pledge) commonly writes instead.
NPSP's stage vocabulary is otherwise org-configured (custom sales processes
can rename or add stages), so the allowlist is a documented default
parameter rather than a constant: see :data:`DEFAULT_INCLUDED_STAGES`.

Sources: Salesforce, *Standard NPSP Data Import Fields*
(https://help.salesforce.com/s/articleView?id=sfdo.npsp_standard_di_fields.htm),
which lists the ``Donation Amount`` / ``Donation Date`` / ``Donation Stage`` /
``Donation Record Type Name`` headers NPSP's own Data Import tool uses;
*NPSP Logic for Creating Opportunity Contact Roles*
(https://help.salesforce.com/s/articleView?id=sfdo.NPSP_Logic_for_Creating_OCRs.htm),
which documents the Opportunity's Primary Contact; the Trailhead modules
"Managing Recurring Donations with Nonprofit Success Pack"
(https://trailhead.salesforce.com/content/learn/modules/donation-management-basics-with-nonprofit-success-pack/create-recurring-donations),
which walks through an instalment Opportunity moving from ``Pledged`` to
``Closed Won``, "Customize Sales Processes and Paths for Nonprofit Success"
(https://trailhead.salesforce.com/content/learn/modules/opportunity-settings-in-nonprofit-success-pack/understand-and-customize-sales-process-and-path-npsp),
which documents that the stage list is a per-org sales process, and "Create
and Manage Stages and Sales Processes"
(https://trailhead.salesforce.com/content/learn/projects/create-an-opportunity-record-type-for-npsp/create-and-manage-stages-and-sales-processes),
which shows the default sales process's open stages (``Prospecting``,
``Qualification``, ...) closing at ``Closed Won`` / ``Closed Lost``; and
Salesforce, *Manage Grantseeking Opportunities*
(https://help.salesforce.com/s/articleView?id=sfdo.NPSP_Create_Manage_Grants.htm),
which documents ``Awarded`` as the Grant record type's closed/won stage.
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
    "DEFAULT_INCLUDED_STAGES",
    "npsp_opportunities_to_features",
    "read_npsp_opportunities",
]

#: Closed/won-equivalent Opportunity stages counted by default: NPSP's own
#: ``Closed Won``, the Grant record type's ``Awarded``, and ``Posted`` (the
#: stage name a payment processor's NPSP integration commonly writes).
#: Matching ignores case, spacing and punctuation, so one spelling here also
#: matches ``"closed won"`` and ``"  CLOSED_WON  "``. Everything else,
#: including ``Pledged``, every open pipeline stage (``Prospecting``,
#: ``Qualification``, ...) and ``Closed Lost``, is excluded by default,
#: because this module cannot enumerate an org's own stage names and an
#: unrecognised stage is not proof that the money arrived.
DEFAULT_INCLUDED_STAGES = (
    "Closed Won",
    "Awarded",
    "Posted",
)

# Donor-key precedence: NPSP's default Household Account model keys a gift to
# the Account, so when an export carries both the Account and the Opportunity's
# Primary Contact, the Account is the intended donor key (see
# _prefer_account_donor_key below).
_ACCOUNT_KEY_ALIASES = frozenset({"account_id", "accountid", "account_name"})
_CONTACT_KEY_ALIASES = frozenset({"primary_contact", "npsp_primary_contact_c"})

# NPSP header normalisation, applied before the CiviCRM bridge's own. Case and
# punctuation are already collapsed ("Close Date" -> close_date), so this maps
# only the residue onto the canonical names. Anything absent here falls
# through to the CiviCRM canonicaliser, which already handles the labels the
# two systems happen to share ("Amount", "Email", "First Name").
_HEADER_ALIASES = {
    # Donor key: NPSP's default Household Account model keys a gift to the
    # Account; "Primary Contact" is the Opportunity's contact-level donor.
    # Report label, then raw API field name, for each.
    "account_id": "contact_id",
    "accountid": "contact_id",
    "account_name": "contact_id",
    "primary_contact": "contact_id",
    "npsp_primary_contact_c": "contact_id",
    # Date: report label "Close Date", API field CloseDate, Data Import
    # template header "Donation Date".
    "close_date": "receive_date",
    "closedate": "receive_date",
    "donation_date": "receive_date",
    # Amount: Data Import template header. "Amount" itself is already handled
    # by the CiviCRM canonicaliser this falls through to.
    "donation_amount": "total_amount",
    # Stage: the column this module exists to read.
    "stage": "gift_type",
    "stagename": "gift_type",
    "donation_stage": "gift_type",
    # Record Type: NPSP's Opportunity classification (Donation, Grant,
    # In-Kind, Major Gift, ...), the NPSP analogue of Raiser's Edge's Fund.
    "record_type": "financial_type",
    "recordtype_name": "financial_type",
    "donation_record_type_name": "financial_type",
}

# Stage-matching key: strip everything but letters and digits so "Closed Won"
# and "closed_won" collapse to one value, matching the Raiser's Edge bridge's
# gift-type matching.
_NON_ALNUM_ALL = re.compile(r"[^a-z0-9]+")


def npsp_opportunities_to_features(
    opportunities: Union[Iterable[Mapping], pd.DataFrame],
    *,
    reference_date: Optional[Union[str, pd.Timestamp]] = None,
    include_stages: Optional[Sequence[str]] = DEFAULT_INCLUDED_STAGES,
) -> pd.DataFrame:
    """Aggregate an NPSP Opportunity export into donor-level features.

    Only rows whose stage is a closed/won equivalent are kept, then handed to
    :func:`civicrm_contributions_to_features`, which produces the donor
    frame. NPSP's stage vocabulary carries no test-mode flag, so none is
    applied.

    Parameters
    ----------
    opportunities : iterable of mapping, or DataFrame
        NPSP Opportunity export rows, under a report's column labels
        (``Account Name`` or ``Primary Contact``, ``Close Date``, ``Amount``,
        ``Stage``), the raw API field names (``AccountId``, ``CloseDate``,
        ``StageName``), or NPSP's Data Import template headers (``Donation
        Date``, ``Donation Amount``, ``Donation Stage``). ``contact_id``,
        ``receive_date`` and ``total_amount`` are required after
        normalisation; ``gift_type``, ``financial_type``, ``email``,
        ``first_name`` and ``last_name`` are used when present. If both an
        Account column (``Account Name``, ``AccountId``) and ``Primary
        Contact`` are present, the Account is used as the donor key, per
        NPSP's default Household Account model.
    reference_date : str or datetime-like, optional
        Anchor for the recency features. If ``None``, the latest Opportunity
        date in the batch is used, which keeps the aggregation free of "now"
        leakage.
    include_stages : sequence of str or None, default=:data:`DEFAULT_INCLUDED_STAGES`
        Allowlist of closed/won-equivalent stages to keep before aggregating,
        matched against ``gift_type`` (the normalised Stage column) ignoring
        case, spacing and punctuation. Every other named stage, including
        ``Pledged``, open pipeline stages and ``Closed Lost``, is dropped:
        an org's own stage vocabulary cannot be enumerated here, so an
        unrecognised stage is treated as not yet money rather than summed.
        ``None`` or an empty sequence disables the filter and sums **every**
        row, which double-counts an instalment recorded in both its
        ``Pledged`` and closed/won stages, and counts open pipeline that was
        never given. Rows whose stage is blank are kept either way: an
        unlabelled row cannot be shown to be anything but a gift.

    Returns
    -------
    features : pandas.DataFrame
        One row per donor, indexed by ``contact_id``, with the same columns
        :func:`civicrm_contributions_to_features` emits.
        ``distinct_financial_types`` counts distinct Record Types here.

    Raises
    ------
    KeyError
        If ``contact_id``, ``receive_date`` or ``total_amount`` is absent
        after normalisation.

    Warns
    -----
    UserWarning
        If ``include_stages`` was requested but the export carries no stage
        column. Silently summing a ``Pledged`` instalment together with the
        ``Closed Won`` row recording its receipt (or an open pipeline row)
        is exactly the error this bridge exists to prevent, so it is worth a
        warning rather than a quiet wrong total.

    Notes
    -----
    Nothing here deduplicates on an Opportunity id, so concatenating two
    exports that overlap in time will double-count the overlap; export
    disjoint date ranges.

    Examples
    --------
    >>> rows = [
    ...     {"Account ID": "88", "Close Date": "2025-01-10",
    ...      "Amount": "100.00", "Stage": "Pledged"},
    ...     {"Account ID": "88", "Close Date": "2025-02-10",
    ...      "Amount": "100.00", "Stage": "Closed Won"},
    ...     {"Account ID": "88", "Close Date": "2025-03-10",
    ...      "Amount": "100.00", "Stage": "Closed Won"},
    ... ]
    >>> feats = npsp_opportunities_to_features(rows)
    >>> float(feats.loc["88", "total_gift_amount"])  # not 300.0
    200.0
    >>> int(feats.loc["88", "gift_count"])
    2
    """
    df = _normalise_headers(_prefer_account_donor_key(_to_frame(opportunities)), _canonical_npsp)
    if df.empty:
        return _empty_feature_frame()

    missing = [col for col in _REQUIRED if col not in df.columns]
    if missing:
        raise KeyError(
            f"NPSP Opportunity export is missing {missing}. Export the "
            f"'Account ID' (or 'Primary Contact'), 'Close Date' and 'Amount' "
            f"fields (Salesforce API: AccountId, CloseDate, Amount); got "
            f"{sorted(df.columns)}."
        )

    if include_stages:
        if "gift_type" in df.columns:
            wanted = {_stage_key(s) for s in include_stages}
            keys = df["gift_type"].map(_stage_key)
            df = df[(keys == "") | keys.isin(wanted)]
        else:
            warnings.warn(
                f"NPSP Opportunity export has no Stage column, so "
                f"include_stages={tuple(include_stages)!r} could not be "
                f"applied: a Pledged instalment or an open-pipeline row (if "
                f"any) is summed alongside the Closed Won row recording a "
                f"gift's receipt, which counts money that was never given. "
                f"Add the 'Stage' field to the export.",
                stacklevel=2,
            )

    if df.empty:
        return _empty_feature_frame()
    return civicrm_contributions_to_features(
        df, reference_date=reference_date, statuses=None
    )


def read_npsp_opportunities(path: Union[str, Path]) -> pd.DataFrame:
    """Read NPSP Opportunity export CSV(s) into one frame.

    Accepts a single ``.csv`` or a directory of them, walked recursively and
    concatenated in sorted relative-path order; symlinks are not followed.
    Every column is read as text and nothing is filtered: **Pledged rows are
    still present**, and it is :func:`npsp_opportunities_to_features` that
    drops them.

    Reading a gift export is CRM-agnostic once the headers are normalised, so
    this delegates to :func:`read_civicrm_contributions` rather than repeating
    its BOM handling and path hardening. NPSP headers are normalised on the
    way into :func:`npsp_opportunities_to_features`, not here.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV file, or a directory of them.

    Returns
    -------
    opportunities : pandas.DataFrame
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
def _canonical_npsp(header: str) -> str:
    key = _NON_ALNUM.sub("_", str(header).strip().lower()).strip("_")
    return _HEADER_ALIASES.get(key) or _canonical(header)


def _prefer_account_donor_key(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve which column is the donor key when an export has both.

    NPSP's default Household Account model keys a gift to the Account, not
    the Opportunity's ``Primary Contact`` (see module docstring). Without
    this, an export carrying both columns would have them collapse onto the
    same ``contact_id`` name and ``_normalise_headers`` would silently keep
    whichever one happens to come first in the export's column order. Drop
    the Primary Contact column(s) first instead, so the donor key is always
    the Account when both are present.
    """
    if df.empty:
        return df
    raw_keys = {c: _NON_ALNUM.sub("_", str(c).strip().lower()).strip("_") for c in df.columns}
    has_account = any(k in _ACCOUNT_KEY_ALIASES for k in raw_keys.values())
    has_contact = any(k in _CONTACT_KEY_ALIASES for k in raw_keys.values())
    if has_account and has_contact:
        drop_cols = [c for c, k in raw_keys.items() if k in _CONTACT_KEY_ALIASES]
        df = df.drop(columns=drop_cols)
    return df


def _stage_key(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return _NON_ALNUM_ALL.sub("", str(value).strip().lower())
