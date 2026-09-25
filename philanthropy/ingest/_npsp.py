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
:func:`npsp_opportunities_to_features` drops the ``Pledged`` rows and hands the
remaining rows to :func:`~philanthropy.ingest.civicrm_contributions_to_features`,
which already knows how to roll a gift log up into the one-row-per-donor frame
the estimators consume.

**The trap this exists to prevent is the same commitment-versus-payment split
Raiser's Edge writes as separate gift records, expressed instead through
Opportunity stage.** NPSP's Recurring Donations feature creates one
Opportunity per instalment; the upcoming instalment is created with
``StageName`` ``Pledged`` (the "we expect this money" stage NPSP selects
automatically), and is only moved to a closed/won stage such as ``Closed Won``
or a site's own ``Posted`` once the gift is actually received. Depending on
the org's "Installment Opportunity Auto-Creation" setting, the ``Pledged``
Opportunity for an instalment and the ``Closed Won`` Opportunity recording its
receipt can both exist, on the same close date, for the same amount, so
summing every Opportunity's ``Amount`` naively counts that instalment twice:
once as the promise, again as the money. ``Pledged`` is excluded by default;
closed/won rows (``Closed Won``, ``Posted``, ...) are kept, because those are
the money.

NPSP's stage vocabulary is org-configured (custom sales processes can rename
or add stages), so the excluded set is a documented default parameter rather
than a constant: see :data:`DEFAULT_EXCLUDED_STAGES`.

Sources: Salesforce, *Standard NPSP Data Import Fields*
(https://help.salesforce.com/s/articleView?id=sfdo.npsp_standard_di_fields.htm),
which lists the ``Donation Amount`` / ``Donation Date`` / ``Donation Stage`` /
``Donation Record Type Name`` headers NPSP's own Data Import tool uses;
*NPSP Logic for Creating Opportunity Contact Roles*
(https://help.salesforce.com/s/articleView?id=sfdo.NPSP_Logic_for_Creating_OCRs.htm),
which documents the Opportunity's Primary Contact; and the Trailhead modules
"Managing Recurring Donations with Nonprofit Success Pack"
(https://trailhead.salesforce.com/content/learn/modules/donation-management-basics-with-nonprofit-success-pack/create-recurring-donations),
which walks through an instalment Opportunity moving from ``Pledged`` to
``Closed Won``/``Posted``, and "Customize Sales Processes and Paths for
Nonprofit Success"
(https://trailhead.salesforce.com/content/learn/modules/opportunity-settings-in-nonprofit-success-pack/understand-and-customize-sales-process-and-path-npsp),
which documents that the stage list is a per-org sales process.
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
    "DEFAULT_EXCLUDED_STAGES",
    "npsp_opportunities_to_features",
    "read_npsp_opportunities",
]

#: Opportunity stages excluded from the roll-up by default: the stage NPSP
#: selects for a Recurring Donation instalment that has not been received yet.
#: Matching ignores case, spacing and punctuation, so this one spelling also
#: matches ``"pledged"`` and ``"  PLEDGED  "``. Closed/won rows (``Closed
#: Won``, a site's own ``Posted``, ...) are *not* in this set and are what the
#: features are built from.
DEFAULT_EXCLUDED_STAGES = (
    "Pledged",
)

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
    exclude_stages: Optional[Sequence[str]] = DEFAULT_EXCLUDED_STAGES,
) -> pd.DataFrame:
    """Aggregate an NPSP Opportunity export into donor-level features.

    ``Pledged`` instalment rows are dropped first, then the surviving rows are
    handed to :func:`civicrm_contributions_to_features`, which produces the
    donor frame. NPSP's stage vocabulary carries no test-mode flag, so none is
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
        ``first_name`` and ``last_name`` are used when present.
    reference_date : str or datetime-like, optional
        Anchor for the recency features. If ``None``, the latest Opportunity
        date in the batch is used, which keeps the aggregation free of "now"
        leakage.
    exclude_stages : sequence of str or None, default=:data:`DEFAULT_EXCLUDED_STAGES`
        Stages to drop before aggregating, matched against ``gift_type``
        (the normalised Stage column) ignoring case, spacing and punctuation.
        ``None`` or an empty sequence disables the filter and sums **every**
        row, which double-counts an instalment recorded in both its
        ``Pledged`` and closed/won stages. Rows whose stage is blank are kept
        either way: an unlabelled row cannot be shown to be a pledge.

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
        If ``exclude_stages`` was requested but the export carries no stage
        column. Silently summing a ``Pledged`` instalment together with the
        ``Closed Won`` row recording its receipt is exactly the error this
        bridge exists to prevent, so it is worth a warning rather than a quiet
        wrong total.

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
    df = _normalise_headers(_to_frame(opportunities), _canonical_npsp)
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

    if exclude_stages:
        if "gift_type" in df.columns:
            unwanted = {_stage_key(s) for s in exclude_stages}
            keys = df["gift_type"].map(_stage_key)
            df = df[~keys.isin(unwanted)]
        else:
            warnings.warn(
                f"NPSP Opportunity export has no Stage column, so "
                f"exclude_stages={tuple(exclude_stages)!r} could not be "
                f"applied: a Pledged instalment (if any) is summed alongside "
                f"the Closed Won row recording its receipt, which counts "
                f"every pledged dollar twice. Add the 'Stage' field to the "
                f"export.",
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


def _stage_key(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return _NON_ALNUM_ALL.sub("", str(value).strip().lower())
