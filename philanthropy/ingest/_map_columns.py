"""
philanthropy.ingest._map_columns
=================================
Rename a user-supplied CRM export's headers to the canonical names a
PhilanthroPy ingest function expects.

A no-code upload flow lets a user pick, per file, which of their own column
headers means "donor ID" or "gift date"; the library never sees the user's
original header names, only the mapping the user chose. :func:`map_columns`
applies that mapping and checks the result actually has every column the next
step requires, so a missing or mistyped mapping fails loudly at the upload
step instead of surfacing as a cryptic ``KeyError`` deep inside a feature
function.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

__all__ = ["map_columns"]


def map_columns(
    df: pd.DataFrame,
    mapping: Mapping[str, str],
    *,
    required: Sequence[str] = (),
) -> pd.DataFrame:
    """Rename a DataFrame's columns per a user-supplied mapping.

    Parameters
    ----------
    df : pandas.DataFrame
        The uploaded export, with whatever headers the source system wrote.
    mapping : mapping of str to str
        User header -> canonical name, e.g. ``{"Constituent ID":
        "contact_id", "Gift Date": "activity_date"}``. Keys not present in
        ``df.columns`` are ignored, so a mapping built from a superset of
        known headers is safe to reuse across files.
    required : sequence of str, default=()
        Canonical column names that must be present after renaming.

    Returns
    -------
    mapped : pandas.DataFrame
        ``df`` with the mapped columns renamed. Unmapped columns are kept
        as-is.

    Raises
    ------
    ValueError
        If, after renaming, any name in ``required`` is missing. The message
        lists every missing column at once, so a user fixing the mapping in a
        UI does not have to resubmit once per missing field.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"CnID": [1], "Gift Date": ["2025-01-01"]})
    >>> map_columns(
    ...     df,
    ...     {"CnID": "contact_id", "Gift Date": "activity_date"},
    ...     required=["contact_id", "activity_date"],
    ... ).columns.tolist()
    ['contact_id', 'activity_date']
    >>> map_columns(df, {"CnID": "contact_id"}, required=["contact_id", "amount"])
    Traceback (most recent call last):
        ...
    ValueError: missing required column(s) after mapping: amount
    """
    mapping = dict(mapping)
    targets: "dict[str, list[str]]" = {}
    for source, target in mapping.items():
        if source in df.columns:
            targets.setdefault(target, []).append(source)
    collisions = {target: sources for target, sources in targets.items() if len(sources) > 1}
    if collisions:
        detail = "; ".join(
            f"{target!r} <- {sources}" for target, sources in collisions.items()
        )
        raise ValueError(
            "mapping assigns multiple source columns to the same target: " + detail
        )

    renamed = df.rename(columns=mapping)
    missing = [col for col in required if col not in renamed.columns]
    if missing:
        raise ValueError(
            "missing required column(s) after mapping: " + ", ".join(missing)
        )
    return renamed
