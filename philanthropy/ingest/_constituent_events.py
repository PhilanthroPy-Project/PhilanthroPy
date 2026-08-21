"""
philanthropy.ingest._constituent_events
========================================
Bridge from UniSchema's ``ConstituentEvent`` stream to a PhilanthroPy
donor-level feature table.

`UniSchema <https://github.com/PhilanthroPy-Project/UniSchema>`_ normalises
fragmented advancement webhooks (GiveCampus, Slate, NPSP, Cvent, ...) into a
single ``ConstituentEvent`` schema and egresses them as per-event JSON or
newline-delimited JSON (NDJSON) batches.  Those events are the raw material for
PhilanthroPy models, but every estimator expects **one row per donor** with
engineered features, not a raw event log.

:func:`constituent_events_to_features` performs that aggregation.  Its output
columns (``total_gift_amount``, ``years_active``, ``event_attendance_count``,
``last_gift_date``, ...) are exactly the ones the estimators and the Quick Start
example already consume, so a UniSchema feed drops straight into a
``DonorPropensityModel`` or ``LapsePredictor`` with no glue code.

The aggregation is leakage-safe in the same spirit as the transformers: the
recency reference point is either supplied explicitly or fixed to the latest
event timestamp in the batch, never a moving "now".
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterable, Mapping, Optional, Union

import numpy as np
import pandas as pd

__all__ = ["constituent_events_to_features", "read_constituent_events"]

# ConstituentEvent.eventType values (see UniSchema src/schema/master.ts).
_DONATION = "DONATION"
_EVENT_REGISTRATION = "EVENT_REGISTRATION"
_EMAIL_CLICK = "EMAIL_CLICK"

# Ordered (column, dtype) contract for the donor feature table.  Kept explicit
# so an empty batch still yields a correctly-typed, downstream-safe frame.
_FEATURE_DTYPES: "dict[str, str]" = {
    "constituent_email": "object",
    "first_name": "object",
    "last_name": "object",
    "total_gift_amount": "float64",
    "gift_count": "int64",
    "event_attendance_count": "int64",
    "email_click_count": "int64",
    "first_gift_date": "datetime64[ns]",
    "last_gift_date": "datetime64[ns]",
    "years_active": "float64",
    "recency_days": "int64",
    "distinct_source_systems": "int64",
}

DAYS_PER_YEAR = 365.25


def constituent_events_to_features(
    events: Union[Iterable[Mapping], pd.DataFrame],
    *,
    reference_date: Optional[Union[str, pd.Timestamp]] = None,
    deduplicate: bool = True,
) -> pd.DataFrame:
    """Aggregate a UniSchema ``ConstituentEvent`` stream into donor features.

    Parameters
    ----------
    events : iterable of mapping, or DataFrame
        Records following UniSchema's ``ConstituentEvent`` schema.  Each event
        carries at least ``constituentEmail``, ``eventType``, ``sourceSystem``,
        and ``createdAt`` (ISO-8601); ``amount``, ``externalConstituentId``,
        ``eventId``, ``firstName``, and ``lastName`` are optional.  Accepts the
        output of
        :func:`read_constituent_events`, a list of dicts, or a DataFrame of the
        same fields.
    reference_date : str or datetime-like, optional
        Anchor for the recency features (``years_active``, ``recency_days``).
        If ``None``, the latest ``createdAt`` in the batch is used; this keeps
        the aggregation reproducible and free of "now" leakage.  Naive
        timestamps are interpreted as UTC.
    deduplicate : bool, default=True
        Drop repeated ``eventId`` values before aggregating.  Advancement
        webhooks are at-least-once, so this prevents a redelivered donation from
        being counted (and its dollars summed) twice.

    Returns
    -------
    features : pandas.DataFrame
        One row per constituent, indexed by ``constituent_id`` (the
        ``externalConstituentId`` when present, else ``constituentEmail``), with
        the columns declared in ``_FEATURE_DTYPES`` (``first_name`` /
        ``last_name`` are populated from the feed when available, else null).
        Rows are sorted by ``constituent_id`` for determinism.

    Warns
    -----
    UserWarning
        If the batch mixes currencies (more than one distinct ``currency``).
        ``total_gift_amount`` is a plain sum with no FX conversion, so a
        single-currency feed is assumed; normalise upstream if it isn't.

    Examples
    --------
    >>> events = [
    ...     {"constituentEmail": "a@x.edu", "eventType": "DONATION",
    ...      "sourceSystem": "GIVECAMPUS", "amount": 250.0,
    ...      "createdAt": "2025-03-01T12:00:00Z"},
    ...     {"constituentEmail": "a@x.edu", "eventType": "EVENT_REGISTRATION",
    ...      "sourceSystem": "CVENT", "createdAt": "2025-06-01T09:00:00Z"},
    ... ]
    >>> feats = constituent_events_to_features(events)
    >>> float(feats.loc["a@x.edu", "total_gift_amount"])
    250.0
    >>> int(feats.loc["a@x.edu", "event_attendance_count"])
    1
    """
    df = _to_frame(events)
    if df.empty:
        return _empty_feature_frame()

    # total_gift_amount sums raw amounts; UniSchema carries a per-event
    # `currency` but no FX rates, so a mixed-currency feed would sum apples and
    # oranges. Warn rather than convert (rates aren't in the stream) or crash.
    if "currency" in df.columns and df["currency"].dropna().nunique() > 1:
        warnings.warn(
            "ConstituentEvent feed mixes currencies "
            f"({sorted(df['currency'].dropna().unique())}); total_gift_amount "
            "is summed without FX conversion. Normalise to one currency first.",
            stacklevel=2,
        )

    if deduplicate and "eventId" in df.columns:
        # Collapse only rows that share a real eventId.  A missing eventId is not
        # "equal" to another missing one, but pandas' drop_duplicates treats
        # NaN == NaN, which would silently drop every id-less donation (and even
        # whole donors) once any event carries an id.
        is_dup = df["eventId"].notna() & df.duplicated(subset="eventId")
        df = df[~is_dup]

    df = df.copy()
    df["_constituent_id"] = _constituent_id(df)
    df["_ts"] = _to_utc_naive(df["createdAt"])
    # An event we can't place in time can't contribute to time-aware features;
    # drop it rather than let a NaT poison a donor's recency to a crash.
    df = df[df["_ts"].notna()]
    if df.empty:
        return _empty_feature_frame()

    # fillna("") keeps a missing/None eventType from becoming pd.NA in the
    # comparisons below (np.where on a NA-bearing mask raises); such an event
    # simply matches no type and contributes to no typed count.
    event_type = (
        df.get("eventType", pd.Series(index=df.index, dtype="object"))
        .astype("string")
        .fillna("")
    )
    raw_amount = df["amount"] if "amount" in df.columns else pd.Series(np.nan, index=df.index)
    # ponytail: UniSchema guarantees `amount` is a JSON number, so a missing
    # amount coerces to 0 (a real absent gift); an unparseable string would too.
    # Parse currency strings ('$250', '1,250.00') here only if a non-conforming
    # feed is ever fed in directly, bypassing UniSchema's validation.
    amount = pd.to_numeric(raw_amount, errors="coerce")

    is_donation = (event_type == _DONATION).to_numpy()
    df["_gift_amount"] = np.where(is_donation, amount.fillna(0.0).to_numpy(), 0.0)
    df["_is_donation"] = is_donation.astype("int64")
    df["_is_event"] = (event_type == _EVENT_REGISTRATION).astype("int64")
    df["_is_click"] = (event_type == _EMAIL_CLICK).astype("int64")
    df["_donation_ts"] = df["_ts"].where(is_donation)

    ref = _resolve_reference_date(reference_date, df["_ts"])

    grouped = df.groupby("_constituent_id", sort=True)
    out = pd.DataFrame(index=grouped.size().index)
    out["constituent_email"] = grouped["constituentEmail"].first()
    # Optional identity fields: carried through when the feed supplies them
    # (the output is a donor-level table, not a de-identified feature store, so
    # it already holds constituent_email). ``.first()`` skips nulls, so a donor
    # whose name rode in on only some events still resolves. Absent column -> None.
    for out_col, src in (("first_name", "firstName"), ("last_name", "lastName")):
        out[out_col] = grouped[src].first() if src in df.columns else None
    out["total_gift_amount"] = grouped["_gift_amount"].sum()
    out["gift_count"] = grouped["_is_donation"].sum()
    out["event_attendance_count"] = grouped["_is_event"].sum()
    out["email_click_count"] = grouped["_is_click"].sum()
    out["first_gift_date"] = grouped["_donation_ts"].min()
    out["last_gift_date"] = grouped["_donation_ts"].max()

    first_seen = grouped["_ts"].min()
    last_seen = grouped["_ts"].max()
    out["years_active"] = ((ref - first_seen).dt.days / DAYS_PER_YEAR).clip(lower=0.0)
    out["recency_days"] = (ref - last_seen).dt.days.clip(lower=0)

    if "sourceSystem" in df.columns:
        out["distinct_source_systems"] = grouped["sourceSystem"].nunique()
    else:
        out["distinct_source_systems"] = 0

    out.index.name = "constituent_id"
    return _coerce_schema(out)


def read_constituent_events(
    path: Union[str, Path],
) -> "list[dict]":
    """Read UniSchema egress files into a list of ``ConstituentEvent`` dicts.

    Handles the shapes UniSchema's egress writes:

    * a single ``.json`` file holding one event (object) or many (array);
    * a ``.ndjson`` / ``.jsonl`` batch, one event per line;
    * a directory, which is walked **recursively**: every ``*.json``,
      ``*.ndjson``, and ``*.jsonl`` file at any depth is read and concatenated,
      sorted by relative path.  This handles UniSchema's date-partitioned egress
      (``{prefix}/{vendor}/{yyyy}/{mm}/{dd}/{eventId}.json``); a flat directory
      still works too.  ``*.manifest.json`` batch sidecars are skipped.

    Parameters
    ----------
    path : str or pathlib.Path
        File or directory to read.

    Returns
    -------
    events : list of dict
        Parsed events, ready to pass to :func:`constituent_events_to_features`.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.  Without this an absent directory falls
        through to the single-file branch and surfaces as an opaque OSError.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file or directory: {p}")
    if p.is_dir():
        # UniSchema's local egress is date-partitioned: it writes each event to
        # {prefix}/{vendor}/{yyyy}/{mm}/{dd}/{eventId}.json (see UniSchema
        # src/egress/objectKey.ts), so the files sit several levels down and a
        # non-recursive scan of the top dir finds nothing. Walk the whole tree.
        events: "list[dict]" = []
        files = [
            f
            for f in p.rglob("*")
            if f.is_file()
            # Skip symlinks: a symlink inside the egress tree pointing outside
            # it must not be followed and read (path-traversal hardening).
            and not f.is_symlink()
            and f.suffix.lower() in {".json", ".ndjson", ".jsonl"}
            # Skip S3 batch sidecars: batch metadata, not ConstituentEvents.
            and not f.name.lower().endswith(".manifest.json")
        ]
        # Sort by relative path so ordering is deterministic across platforms.
        for child in sorted(files, key=lambda f: f.relative_to(p).as_posix()):
            events.extend(_read_events_file(child))
        return events
    return _read_events_file(p)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _read_events_file(path: Path) -> "list[dict]":
    if path.suffix.lower() in {".ndjson", ".jsonl"}:
        # Stream line-by-line so a large NDJSON batch is never held in memory
        # twice, and name the offending file+line if a record is malformed
        # (instead of aborting the whole ingest with a contextless traceback).
        events: "list[dict]" = []
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON in {path} at line {lineno}: {exc}"
                    ) from exc
        return events
    # .json (or unknown): a single object or an array of objects.
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path}: {exc}") from exc
    return list(data) if isinstance(data, list) else [data]


def _to_frame(events: Union[Iterable[Mapping], pd.DataFrame]) -> pd.DataFrame:
    if isinstance(events, pd.DataFrame):
        return events
    return pd.DataFrame(list(events))


def _constituent_id(df: pd.DataFrame) -> pd.Series:
    """External CRM id when present and non-empty, else the email."""
    email = df["constituentEmail"].astype("string")
    if "externalConstituentId" not in df.columns:
        return email
    ext = df["externalConstituentId"].astype("string")
    has_ext = ext.notna() & (ext.str.strip().str.len() > 0)
    return ext.where(has_ext, email)


def _to_utc_naive(series: pd.Series) -> pd.Series:
    """Parse ISO timestamps to tz-naive UTC (matches the datasets' naive dates).

    ``createdAt`` is guaranteed ISO-8601 by the ConstituentEvent schema, so we
    pin ``format="ISO8601"``; this parses mixed offsets/precisions without the
    slow, warning-emitting per-element dateutil fallback.
    """
    ts = pd.to_datetime(series, utc=True, format="ISO8601", errors="coerce")
    return ts.dt.tz_localize(None)


def _resolve_reference_date(
    reference_date: Optional[Union[str, pd.Timestamp]],
    timestamps: pd.Series,
) -> pd.Timestamp:
    if reference_date is None or pd.isna(reference_date):
        return timestamps.max()
    ref = pd.to_datetime(reference_date, utc=True)
    return ref.tz_localize(None)


def _empty_feature_frame() -> pd.DataFrame:
    out = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in _FEATURE_DTYPES.items()})
    out.index = pd.Index([], name="constituent_id", dtype="object")
    return out


def _coerce_schema(out: pd.DataFrame) -> pd.DataFrame:
    return out[list(_FEATURE_DTYPES)].astype(_FEATURE_DTYPES)
