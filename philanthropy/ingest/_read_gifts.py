"""
philanthropy.ingest._read_gifts
================================
One call over the CiviCRM, Raiser's Edge and NPSP gift bridges.

Each bridge module pairs its own ``read_<source>_...`` loader with a
``<source>_..._to_features`` aggregator, because each CRM's export needs its
own header aliases and commitment-versus-payment exclusion filter before the
rows can be handed to the shared aggregator in
:mod:`philanthropy.ingest._civicrm`. A caller working with more than one CRM
export format otherwise has to import three differently-named pairs for what
is, from the outside, the same read-then-aggregate operation.
:func:`read_gifts` is that pair looked up by a ``source`` name instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Tuple, Union

import pandas as pd

from ._civicrm import civicrm_contributions_to_features, read_civicrm_contributions
from ._npsp import npsp_opportunities_to_features, read_npsp_opportunities
from ._raisers_edge import raisers_edge_gifts_to_features, read_raisers_edge_gifts

__all__ = ["GIFT_SOURCES", "read_gifts"]

#: Valid ``source`` names for :func:`read_gifts`, in the order the CLI's
#: `--source` choices already list them.
GIFT_SOURCES: Tuple[str, ...] = ("civicrm", "raisers_edge", "npsp")

# (reader, aggregator) pair per source, the same shape as the preset dispatch
# in cli.py's _cmd_features.
_REGISTRY: "dict[str, tuple[Callable[[Union[str, Path]], pd.DataFrame], Callable[..., pd.DataFrame]]]" = {
    "civicrm": (read_civicrm_contributions, civicrm_contributions_to_features),
    "raisers_edge": (read_raisers_edge_gifts, raisers_edge_gifts_to_features),
    "npsp": (read_npsp_opportunities, npsp_opportunities_to_features),
}


def read_gifts(
    path_or_df: Union[str, Path, pd.DataFrame, Iterable[Mapping]],
    *,
    source: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read (if given a path) and aggregate a CRM gift export in one call.

    Looks ``source`` up in a small preset registry mapping each of
    ``"civicrm"``, ``"raisers_edge"`` and ``"npsp"`` to that CRM's
    ``read_<source>_...`` loader and ``<source>_..._to_features`` aggregator,
    then runs the pair. ``read_gifts("gifts.csv", source="npsp")`` is
    equivalent to
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
        ``exclude_stages=`` for ``"npsp"``, or the ``reference_date=`` every
        preset accepts.

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
