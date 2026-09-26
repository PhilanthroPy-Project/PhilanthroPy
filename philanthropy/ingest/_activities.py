"""
philanthropy.ingest._activities
================================
Turn a long, multi-source "activity" log (event attendance, volunteer
shifts, email clicks, ...) into per-donor engagement features.

A no-code upload flow lets a user hand over as many activity files as they
have, each tagged with a type (``"event"``, ``"volunteer"``, ...) and mapped
onto a shared shape: a donor id, a date, and optionally an amount, hours, or
name. :func:`activities_to_features` is the single aggregator every one of
those types runs through, so a new activity type never needs new model code:
it just yields its own ``<type>_...`` columns the next time the function sees
it.

This generalises the aggregation pattern in
:mod:`philanthropy.ingest._constituent_events` (a fixed handful of event
types rolled into named columns) to an open-ended set of types discovered
from the data itself, while keeping that module's conventions: an explicit,
sorted donor id key, explicit output dtypes, and a reference date that never
moves with "now".

**Leakage.** ``as_of`` is required, not optional: every row dated after it is
dropped before anything is counted, so a gala attended after a cutoff can
never leak into that cutoff's features. The 12- and 36-month windows, the
lifetime distinct count, and the days-since-last figure are all computed
after that cut, from that cut backwards.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Mapping, Optional, Union

import pandas as pd

from philanthropy.ingest._civicrm import _resolve_reference_date, _to_datetime

__all__ = ["activities_to_features"]

_REQUIRED = ("contact_id", "activity_date", "activity_type")


def activities_to_features(
    activities: Union[Iterable[Mapping], pd.DataFrame],
    *,
    as_of: Union[str, pd.Timestamp],
    donors: Optional[Union[pd.DataFrame, pd.Series, Iterable]] = None,
) -> pd.DataFrame:
    """Aggregate a long activity log into per-donor, per-type features.

    Parameters
    ----------
    activities : iterable of mapping, or DataFrame
        Long-format rows, one per activity: ``contact_id``, ``activity_date``,
        ``activity_type`` are required; ``amount`` and ``hours`` are used when
        present (summed, per type, over the trailing 12 months); ``name`` is
        accepted but not aggregated.
    as_of : str or datetime-like
        The cutoff. Rows with ``activity_date`` after ``as_of`` are dropped
        before any feature is computed; every window and recency figure is
        measured back from this date, not from "now" or from the batch's own
        latest date. A tz-aware timestamp (e.g. ``pd.Timestamp("2024-12-31",
        tz="UTC")``) is converted to naive UTC, the same treatment
        ``activity_date`` gets, so the two are always comparable.
    donors : DataFrame, Series, or iterable, optional
        The donor population this activity log is being matched against (its
        index if a DataFrame, its values otherwise). Used only to report the
        match rate: the share of distinct ``contact_id`` values in the
        (cutoff) activity log that are also in ``donors``. Does not affect
        which rows appear in the output; a donor absent from the activity log
        entirely does not get a row here regardless of ``donors``.

    Returns
    -------
    features : pandas.DataFrame
        One row per donor with at least one activity at or before ``as_of``,
        indexed by ``contact_id``. For each activity type present in that
        (cutoff) data, four columns: ``<type>_count_12m``, ``<type>_count_36m``
        (counts in the trailing 12 / 36 months before ``as_of``, exclusive of
        the boundary itself: a row exactly 12 months before ``as_of`` falls
        outside ``<type>_count_12m``, matching the "trailing 12 *full*
        months" reading rather than "12 months or less ago"),
        ``<type>_days_since_last`` (days from the type's most recent activity
        to ``as_of``; ``NaN`` for a donor with no activity of that type at
        all, not 0 -- 0 would read as "did it today"), and ``<type>_distinct``
        (lifetime count of distinct activity dates of that type, unwindowed).
        Plus ``<type>_hours_12m`` / ``<type>_amount_12m`` (trailing-12-month
        sums) when the input carries an ``hours`` / ``amount`` column at all.
        A donor with no rows of a given type gets 0 in that type's count and
        distinct columns (but ``NaN`` in ``days_since_last``, see above); a
        type with no rows anywhere in the (cutoff) data contributes no
        columns at all. Rows are sorted by ``contact_id`` for determinism.

    Raises
    ------
    KeyError
        If ``contact_id``, ``activity_date``, or ``activity_type`` is absent.

    Warns
    -----
    UserWarning
        If ``donors`` is given and fewer than 80% of the distinct
        ``contact_id`` values in the (cutoff) activity log are found in it;
        the usual cause is an activity export keyed on email while the CRM
        export it is being joined against is keyed on an internal id.

    Examples
    --------
    >>> rows = [
    ...     {"contact_id": "1", "activity_date": "2024-03-01",
    ...      "activity_type": "volunteer", "hours": 2},
    ...     {"contact_id": "1", "activity_date": "2023-01-01",
    ...      "activity_type": "volunteer", "hours": 3},
    ...     {"contact_id": "2", "activity_date": "2024-06-01",
    ...      "activity_type": "event"},
    ... ]
    >>> feats = activities_to_features(rows, as_of="2024-12-31")
    >>> int(feats.loc["1", "volunteer_count_12m"])
    1
    >>> int(feats.loc["1", "volunteer_distinct"])
    2
    >>> float(feats.loc["1", "volunteer_hours_12m"])
    2.0
    >>> "event_count_12m" in feats.columns
    True
    >>> int(feats.loc["2", "event_count_12m"])
    1
    """
    df = _to_frame(activities)

    if df.empty:
        return pd.DataFrame(index=pd.Index([], name="contact_id", dtype="object"))

    missing = [col for col in _REQUIRED if col not in df.columns]
    if missing:
        raise KeyError(
            f"Activity log is missing {missing}. Every activity row needs "
            f"'contact_id', 'activity_date' and 'activity_type'; got "
            f"{sorted(df.columns)}."
        )

    df = df.copy()
    df["_contact_id"] = _normalise_contact_id(df["contact_id"]).str.strip()
    df["_ts"] = _to_datetime(df["activity_date"])
    df["_type"] = df["activity_type"].astype("string").str.strip()
    # as_of is required, so this never falls back to the batch's own latest
    # date; it only normalises as_of (naive or tz-aware) to naive UTC, the
    # same treatment activity_date already got via _to_datetime, so the two
    # sides of every comparison below share a tz-awareness.
    as_of_ts = _resolve_reference_date(as_of, df["_ts"])

    # A row we can't place in time or attribute to a donor and a type
    # contributes to nothing; drop it rather than let a NaT or blank type
    # poison a window or split into its own meaningless column.
    df = df[
        df["_contact_id"].notna()
        & (df["_contact_id"].str.len() > 0)
        & df["_ts"].notna()
        & df["_type"].notna()
        & (df["_type"].str.len() > 0)
    ]
    # The cutoff: nothing after as_of is counted anywhere below, including in
    # which types and donors even appear in the output.
    df = df[df["_ts"] <= as_of_ts]
    if df.empty:
        return pd.DataFrame(index=pd.Index([], name="contact_id", dtype="object"))

    has_amount = "amount" in df.columns
    has_hours = "hours" in df.columns
    if has_amount:
        df["_amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    if has_hours:
        df["_hours"] = pd.to_numeric(df["hours"], errors="coerce").fillna(0.0)

    if donors is not None:
        _warn_on_low_match_rate(df["_contact_id"], donors)

    window_12m = as_of_ts - pd.DateOffset(months=12)
    window_36m = as_of_ts - pd.DateOffset(months=36)

    donor_ids = pd.Index(sorted(df["_contact_id"].unique()), name="contact_id")
    out = pd.DataFrame(index=donor_ids)

    for activity_type in sorted(df["_type"].unique()):
        type_rows = df[df["_type"] == activity_type]
        grouped = type_rows.groupby("_contact_id")
        recent_12m = type_rows[type_rows["_ts"] > window_12m].groupby("_contact_id")
        recent_36m = type_rows[type_rows["_ts"] > window_36m].groupby("_contact_id")

        prefix = f"{activity_type}_"
        out[prefix + "count_12m"] = recent_12m.size().reindex(donor_ids, fill_value=0)
        out[prefix + "count_36m"] = recent_36m.size().reindex(donor_ids, fill_value=0)
        last_activity = grouped["_ts"].max().reindex(donor_ids)
        # A donor with no rows of this type has no "last time" to measure
        # from; leave this NaN (rather than 0, which reads as "did it
        # today") -- the whole point of a per-type column is telling the two
        # cases apart.
        out[prefix + "days_since_last"] = (as_of_ts - last_activity).dt.days
        out[prefix + "distinct"] = (
            grouped["_ts"].nunique().reindex(donor_ids, fill_value=0)
        )

        if has_hours:
            hours_12m = type_rows[type_rows["_ts"] > window_12m].groupby("_contact_id")["_hours"].sum()
            out[prefix + "hours_12m"] = hours_12m.reindex(donor_ids, fill_value=0.0)
        if has_amount:
            amount_12m = type_rows[type_rows["_ts"] > window_12m].groupby("_contact_id")["_amount"].sum()
            out[prefix + "amount_12m"] = amount_12m.reindex(donor_ids, fill_value=0.0)

    for col in out.columns:
        if col.endswith(("_count_12m", "_count_36m", "_distinct")):
            out[col] = out[col].astype("int64")
        elif col.endswith("_days_since_last"):
            # float64, not int64: a donor with no rows of this type is NaN
            # here (see above), and int64 has no way to hold that.
            out[col] = out[col].astype("float64")
        else:
            out[col] = out[col].astype("float64")

    return out


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _to_frame(activities: Union[Iterable[Mapping], pd.DataFrame]) -> pd.DataFrame:
    if isinstance(activities, pd.DataFrame):
        return activities
    return pd.DataFrame(list(activities))


def _normalise_contact_id(series: pd.Series) -> pd.Series:
    """Coerce a contact_id column to string, without a spurious ".0".

    ``pd.read_csv`` reads an id column as float64 the moment any cell in it
    is blank (NaN forces the whole column off int64), so "123" round-trips as
    123.0 and, left to plain ``.astype("string")``, becomes the string
    "123.0" -- which then fails to join against the same donor's "123" from
    a column that never had a blank. Format an integral float back to its
    bare digits first; a genuinely fractional id (not a real CRM id, but not
    this function's problem either) is left alone.
    """
    if not pd.api.types.is_float_dtype(series):
        return series.astype("string")
    as_int_str = series.map(
        lambda v: str(int(v)) if pd.notna(v) and float(v).is_integer() else v
    )
    return as_int_str.astype("string")


def _donor_id_index(donors: Union[pd.DataFrame, pd.Series, Iterable]) -> pd.Index:
    if isinstance(donors, pd.DataFrame):
        return pd.Index(donors.index.astype("string").str.strip())
    if isinstance(donors, pd.Series):
        return pd.Index(donors.astype("string").str.strip())
    return pd.Index(pd.Series(list(donors)).astype("string").str.strip())


def _warn_on_low_match_rate(
    activity_contact_ids: pd.Series, donors: Union[pd.DataFrame, pd.Series, Iterable]
) -> None:
    donor_ids = set(_donor_id_index(donors))
    distinct_activity_ids = set(activity_contact_ids.unique())
    matched = len(distinct_activity_ids & donor_ids)
    match_rate = matched / len(distinct_activity_ids)
    if match_rate < 0.8:
        warnings.warn(
            f"Only {match_rate:.0%} of the {len(distinct_activity_ids)} distinct "
            f"contact_id values in this activity log were found in donors "
            f"({matched} matched). A low match rate usually means the activity "
            f"export is keyed on a different id (e.g. email) than the CRM "
            f"export it is being joined against; re-export using the same "
            f"donor id in both files.",
            stacklevel=2,
        )
