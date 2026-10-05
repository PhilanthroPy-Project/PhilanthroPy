"""
philanthropy.ingest._little_green_light
=======================================

Bridge from a Little Green Light gift export to a PhilanthroPy donor-level
feature table.

Little Green Light (LGL) gift reports can include ``LGL Constituent ID``,
``Gift date``, ``Gift type``, ``Amount``, ``Campaign`` and ``Fund``. These
fields are normalised onto the canonical ``contact_id`` / ``receive_date`` /
``total_amount`` / ``gift_type`` names used by the shared CiviCRM aggregator.

The important distinction is between a pledge and money actually received.
LGL documents a pledge as a formal commitment that can be fulfilled by
subsequent payments. Therefore ``Pledge`` rows are excluded from the
received-gift roll-up by default, while payment/gift rows remain.

Sources:
Little Green Light Knowledge Base, "Mapping gift fields":
https://help.littlegreenlight.com/article/288-mapping-gift-fields

Little Green Light Knowledge Base, "Mapping pledges":
https://help.littlegreenlight.com/article/295-mapping-pledges

Little Green Light Knowledge Base,
"Generate a lump sum report to record donations into QuickBooks":
https://help.littlegreenlight.com/article/241-generate-a-lump-sum-report-to-record-into-quickbooks
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
    "DEFAULT_EXCLUDED_LGL_GIFT_TYPES",
    "little_green_light_gifts_to_features",
    "read_little_green_light_gifts",
]


#: LGL's Pledge gift type represents a commitment rather than money received.
#: Payments made in fulfilment of the pledge are represented by gift/payment
#: records and should remain in the roll-up.
DEFAULT_EXCLUDED_LGL_GIFT_TYPES = ("Pledge",)


_HEADER_ALIASES = {
    # Constituent identifier used by LGL exports.
    "lgl_constituent_id": "contact_id",
    "constituent_id": "contact_id",

    # Gift date.
    "gift_date": "receive_date",

    # Gift amount.
    "gift_amount": "total_amount",
    "amount": "total_amount",

    # Gift classification.
    "gift_type": "gift_type",

    # Fund is the closest analogue to CiviCRM's financial type.
    "fund": "financial_type",
}


_NON_ALNUM_ALL = re.compile(r"[^a-z0-9]+")


def little_green_light_gifts_to_features(
    gifts: Union[Iterable[Mapping], pd.DataFrame],
    *,
    reference_date: Optional[Union[str, pd.Timestamp]] = None,
    exclude_gift_types: Optional[Sequence[str]] = DEFAULT_EXCLUDED_LGL_GIFT_TYPES,
) -> pd.DataFrame:
    """Aggregate a Little Green Light gift export into donor-level features.

    Pledge rows are excluded by default because LGL defines a pledge as a
    commitment fulfilled by subsequent payments. Surviving gift/payment rows
    are delegated to the shared CiviCRM feature aggregator.

    Parameters
    ----------
    gifts : iterable of mapping, or DataFrame
        Little Green Light gift rows. ``LGL Constituent ID``, ``Gift date``
        and ``Amount`` are required after normalisation. ``Gift type`` and
        ``Fund`` are used when present.
    reference_date : str or datetime-like, optional
        Anchor for recency features. If ``None``, the latest surviving gift
        date in the batch is used.
    exclude_gift_types : sequence of str or None
        Gift types to exclude before aggregation. By default, ``Pledge`` is
        excluded. ``None`` or an empty sequence disables filtering.

    Returns
    -------
    pandas.DataFrame
        One row per constituent, indexed by ``contact_id``.
    """
    df = _normalise_headers(_to_frame(gifts), _canonical_little_green_light)

    if df.empty:
        return _empty_feature_frame()

    missing = [col for col in _REQUIRED if col not in df.columns]
    if missing:
        raise KeyError(
            f"Little Green Light gift export is missing {missing}. Export the "
            f"'LGL Constituent ID', 'Gift date' and 'Amount' fields; "
            f"got {sorted(df.columns)}."
        )

    if exclude_gift_types:
        if "gift_type" in df.columns:
            unwanted = {_gift_type_key(t) for t in exclude_gift_types}
            keys = df["gift_type"].map(_gift_type_key)
            df = df[~keys.isin(unwanted)]
        else:
            warnings.warn(
                "Little Green Light gift export has no gift type column, so "
                f"exclude_gift_types={tuple(exclude_gift_types)!r} could not "
                "be applied: pledge commitments (if present) may be summed "
                "alongside payments. Add the 'Gift type' field to the export.",
                stacklevel=2,
            )

    if df.empty:
        return _empty_feature_frame()

    return civicrm_contributions_to_features(
        df,
        reference_date=reference_date,
        statuses=None,
    )


def read_little_green_light_gifts(path: Union[str, Path]) -> pd.DataFrame:
    """Read Little Green Light gift export CSV(s) into one frame.

    Reading is delegated to the shared CiviCRM CSV reader. Filtering and
    Little Green Light header normalisation occur in
    :func:`little_green_light_gifts_to_features`.
    """
    return read_civicrm_contributions(path)


def _canonical_little_green_light(name: object) -> str:
    """Return the canonical name for a Little Green Light export header."""
    key = _NON_ALNUM.sub("_", str(name).strip().lower()).strip("_")
    return _HEADER_ALIASES.get(key, _canonical(str(name)))


def _gift_type_key(value: object) -> str:
    """Collapse case, whitespace and punctuation for gift-type matching."""
    return _NON_ALNUM_ALL.sub("", str(value).strip().lower())