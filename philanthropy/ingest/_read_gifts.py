"""
philanthropy.ingest._read_gifts
================================
One call over the CiviCRM, Raiser's Edge, Blackbaud CRM, NPSP, Nonprofit
Cloud, Bloomerang, DonorPerfect, Little Green Light and Ellucian CRM
Advance gift bridges.

Each bridge module pairs its own ``read_<source>_...`` loader with a
``<source>_..._to_features`` aggregator, because each CRM's export needs its
own header aliases and commitment-versus-payment exclusion filter before the
rows can be handed to the shared aggregator in
:mod:`philanthropy.ingest._civicrm`. A caller working with more than one CRM
export format otherwise has to import three differently-named pairs for what
is, from the outside, the same read-then-aggregate operation.
:func:`read_gifts` is that pair looked up by a ``source`` name instead, and
:func:`read_gift_rows` returns the gift rows that pair counts, before they
are rolled up per donor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Union

import pandas as pd

from ._blackbaud_crm import (
    _blackbaud_crm_received_rows,
    blackbaud_crm_revenue_to_features,
    read_blackbaud_crm_revenue,
)
from ._bloomerang import (
    _bloomerang_received_rows,
    bloomerang_transactions_to_features,
    read_bloomerang_transactions,
)
from ._civicrm import (
    _civicrm_received_rows,
    _normalise_headers,
    _require_civicrm_columns,
    _to_frame,
    _typed_gift_rows,
    civicrm_contributions_to_features,
    read_civicrm_contributions,
)
from ._donorperfect import (
    _donorperfect_received_rows,
    donorperfect_gifts_to_features,
    read_donorperfect_gifts,
)
from ._ellucian_advance import (
    _ellucian_advance_received_rows,
    ellucian_advance_gifts_to_features,
    read_ellucian_advance_gifts,
)
from ._little_green_light import (
    _little_green_light_received_rows,
    little_green_light_gifts_to_features,
    read_little_green_light_gifts,
)
from ._nonprofit_cloud import (
    _nonprofit_cloud_received_rows,
    nonprofit_cloud_gifts_to_features,
    read_nonprofit_cloud_gifts,
)
from ._npsp import _npsp_received_rows, npsp_opportunities_to_features, read_npsp_opportunities
from ._raisers_edge import _raisers_edge_received_rows, raisers_edge_gifts_to_features, read_raisers_edge_gifts

__all__ = ["GIFT_SOURCES", "read_gifts", "read_gift_rows"]

#: Valid ``source`` names for :func:`read_gifts` and :func:`read_gift_rows`.
GIFT_SOURCES: Tuple[str, ...] = (
    "civicrm",
    "raisers_edge",
    "npsp",
    "nonprofit_cloud",
    "bloomerang",
    "donorperfect",
    "blackbaud_crm",
    "little_green_light",
    "ellucian_advance",
)

# (reader, aggregator) pair per source.
_REGISTRY: "dict[str, tuple[Callable[[Union[str, Path]], pd.DataFrame], Callable[..., pd.DataFrame]]]" = {
    "civicrm": (read_civicrm_contributions, civicrm_contributions_to_features),
    "raisers_edge": (read_raisers_edge_gifts, raisers_edge_gifts_to_features),
    "npsp": (read_npsp_opportunities, npsp_opportunities_to_features),
    "nonprofit_cloud": (read_nonprofit_cloud_gifts, nonprofit_cloud_gifts_to_features),
    "bloomerang": (read_bloomerang_transactions, bloomerang_transactions_to_features),
    "donorperfect": (read_donorperfect_gifts, donorperfect_gifts_to_features),
    "blackbaud_crm": (read_blackbaud_crm_revenue, blackbaud_crm_revenue_to_features),
    "little_green_light": (
        read_little_green_light_gifts,
        little_green_light_gifts_to_features,
    ),
    "ellucian_advance": (
        read_ellucian_advance_gifts,
        ellucian_advance_gifts_to_features,
    ),
}


def read_gifts(
    path_or_df: Union[str, Path, pd.DataFrame, Iterable[Mapping]],
    *,
    source: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read (if given a path) and aggregate a CRM gift export in one call.

    Looks ``source`` up in a small preset registry mapping each name in
    :data:`GIFT_SOURCES` to that CRM's ``read_<source>_...`` loader and
    ``<source>_..._to_features`` aggregator, then runs the pair.
    ``read_gifts("gifts.csv", source="npsp")`` is equivalent to
    ``npsp_opportunities_to_features(read_npsp_opportunities("gifts.csv"))``.

    Parameters
    ----------
    path_or_df : str, pathlib.Path, DataFrame, or iterable of mapping
        A gift export CSV file, a directory of them, or gift rows already in
        memory (a DataFrame or an iterable of mappings, e.g. an API result).
        A path is only meaningful for a single source's own export format;
        already-in-memory rows are handed straight to the aggregator.
    source : str
        Which CRM the export came from. One of :data:`GIFT_SOURCES`.
    **kwargs
        Passed through to the source's aggregator, e.g. ``statuses=`` for
        ``"civicrm"``, ``exclude_gift_types=`` for ``"raisers_edge"``,
        ``include_revenue_types=`` for ``"blackbaud_crm"``, ``include_stages=``
        for ``"npsp"``, ``include_statuses=`` for ``"nonprofit_cloud"``,
        ``exclude_entry_types=`` for ``"bloomerang"``,
        ``exclude_record_types=`` for ``"donorperfect"``,
        ``exclude_gift_types=`` for ``"little_green_light"``,
        ``exclude_transaction_types=`` for ``"ellucian_advance"``, or the
        ``reference_date=`` every preset accepts.

    Returns
    -------
    features : pandas.DataFrame
        One row per donor, indexed by ``contact_id``, identical to calling
        the source's own reader and aggregator directly.

    Raises
    ------
    ValueError
        If ``source`` is not one of :data:`GIFT_SOURCES`.

    Examples
    --------
    >>> rows = [
    ...     {"Constituent ID": "88", "Gift Date": "2025-01-10",
    ...      "Gift Amount": "1200.00", "Gift Type": "Pledge"},
    ...     {"Constituent ID": "88", "Gift Date": "2025-02-10",
    ...      "Gift Amount": "100.00", "Gift Type": "Pay-Cash"},
    ... ]
    >>> feats = read_gifts(rows, source="raisers_edge")
    >>> float(feats.loc["88", "total_gift_amount"])  # the pledge is excluded
    100.0
    """
    try:
        read, to_features = _REGISTRY[source]
    except KeyError:
        raise ValueError(
            f"Unknown gift source {source!r}; expected one of {GIFT_SOURCES}."
        ) from None

    if isinstance(path_or_df, (str, Path)):
        path_or_df = read(path_or_df)
    return to_features(path_or_df, **kwargs)


# (reader, received-rows filter) pair per source: the filter each source's
# aggregator applies before handing its rows to the CiviCRM aggregator.
# CiviCRM's own filter is the status filter applied in read_gift_rows.
_ROW_FILTERS: "dict[str, tuple[Callable[[Union[str, Path]], pd.DataFrame], Optional[Callable[..., pd.DataFrame]]]]" = {
    "civicrm": (read_civicrm_contributions, None),
    "raisers_edge": (read_raisers_edge_gifts, _raisers_edge_received_rows),
    "npsp": (read_npsp_opportunities, _npsp_received_rows),
    "nonprofit_cloud": (read_nonprofit_cloud_gifts, _nonprofit_cloud_received_rows),
    "bloomerang": (read_bloomerang_transactions, _bloomerang_received_rows),
    "donorperfect": (read_donorperfect_gifts, _donorperfect_received_rows),
    "blackbaud_crm": (read_blackbaud_crm_revenue, _blackbaud_crm_received_rows),
    "little_green_light": (read_little_green_light_gifts, _little_green_light_received_rows),
    "ellucian_advance": (read_ellucian_advance_gifts, _ellucian_advance_received_rows),
}


def read_gift_rows(
    path_or_df: Union[str, Path, pd.DataFrame, Iterable[Mapping]],
    *,
    source: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a CRM gift export and return one row per counted gift.

    Applies exactly the filtering :func:`read_gifts` applies for the same
    ``source`` (pledges and other commitments, open stages, unpaid schedules,
    duplicate CiviCRM ids, test-mode rows, non-``Completed`` CiviCRM
    statuses, rows with no donor id or no parseable date), but stops before
    rolling the gifts up per donor. The result is valid
    :func:`~philanthropy.ingest.build_snapshots` input, so one export can
    feed both the donor-level features and the donor x fiscal-year
    snapshots without reaching into the source's private helpers.

    Parameters
    ----------
    path_or_df : str, pathlib.Path, DataFrame, or iterable of mapping
        A gift export CSV file, a directory of them, or gift rows already in
        memory, as for :func:`read_gifts`.
    source : str
        Which CRM the export came from. One of :data:`GIFT_SOURCES`.
    **kwargs
        The source's row filter option, as for :func:`read_gifts`:
        ``statuses=`` for ``"civicrm"``, ``exclude_gift_types=`` for
        ``"raisers_edge"`` and ``"little_green_light"``,
        ``include_stages=`` for ``"npsp"``, and so on. ``reference_date=``
        does not apply to rows and is not accepted.

    Returns
    -------
    rows : pandas.DataFrame
        Columns ``donor_id`` (string), ``gift_date`` (datetime) and
        ``gift_amount`` (float, an unparseable amount counted as 0, as
        :func:`read_gifts` counts it), in export order.

    Raises
    ------
    ValueError
        If ``source`` is not one of :data:`GIFT_SOURCES`.
    KeyError
        If the export is missing the donor id, date or amount field.
    TypeError
        If a keyword is not the source's filter option.

    Examples
    --------
    >>> rows = [
    ...     {"Constituent ID": "88", "Gift Date": "2025-01-10",
    ...      "Gift Amount": "1200.00", "Gift Type": "Pledge"},
    ...     {"Constituent ID": "88", "Gift Date": "2025-02-10",
    ...      "Gift Amount": "100.00", "Gift Type": "Pay-Cash"},
    ... ]
    >>> read_gift_rows(rows, source="raisers_edge")  # the pledge is excluded
      donor_id  gift_date  gift_amount
    0       88 2025-02-10        100.0
    """
    try:
        read, row_filter = _ROW_FILTERS[source]
    except KeyError:
        raise ValueError(
            f"Unknown gift source {source!r}; expected one of {GIFT_SOURCES}."
        ) from None

    if isinstance(path_or_df, (str, Path)):
        path_or_df = read(path_or_df)
    # Called directly, not through a wrapper, so the filters' missing-column
    # warnings (stacklevel=3) still point at the caller.
    if row_filter is None:
        df = _normalise_headers(_to_frame(path_or_df))
        if not df.empty:
            _require_civicrm_columns(df)
            df = _civicrm_received_rows(df, **kwargs)
    else:
        df = row_filter(path_or_df, **kwargs)
        if not df.empty:
            df = _civicrm_received_rows(df, None)
    if df.empty:
        return pd.DataFrame({
            "donor_id": pd.Series(dtype="string"),
            "gift_date": pd.Series(dtype="datetime64[ns]"),
            "gift_amount": pd.Series(dtype="float64"),
        })
    df = _typed_gift_rows(df)
    return pd.DataFrame({
        "donor_id": df["_contact_id"].to_numpy(),
        "gift_date": df["_ts"].to_numpy(),
        "gift_amount": df["_amount"].to_numpy("float64"),
    }).astype({"donor_id": "string"})
