"""
philanthropy.ingest._civicrm
============================
Bridge from a CiviCRM contribution export to a PhilanthroPy donor-level
feature table.

`CiviCRM <https://civicrm.org>`_ is the CRM most small and mid-sized nonprofits
actually run, and ``civicrm_contribution`` is where their giving history lives.
It surfaces that table two ways and they disagree on spelling: a CSV export
carries human labels (``Contact ID``, ``Total Amount``, ``Contribution Date``),
while APIv4 returns the underlying DB columns (``contact_id``, ``total_amount``,
``receive_date``).  Both are accepted here (headers are normalised to the APIv4
name), so a hand-run export and a scripted API pull take the same code path.

:func:`read_civicrm_contributions` loads the CSV(s);
:func:`civicrm_contributions_to_features` aggregates the gift log into the
one-row-per-donor frame the estimators consume.

Two things a bare ``pd.read_csv`` gets wrong, and this bridge does not:

* **Test-mode rows are real rows in the export.** CiviCRM writes payment-processor
  test transactions to the same table (``is_test`` / ``Test Mode``).  Summing
  them inflates lifetime giving with money nobody gave, so they are always
  dropped.
* **A contribution is not a payment.** ``contribution_status`` separates
  ``Completed`` from ``Pending``, ``Failed``, ``Refunded``, ``Cancelled`` and
  ``Chargeback``.  Only ``Completed`` is counted by default.

The aggregation is leakage-safe in the same spirit as the transformers: the
recency reference point is either supplied explicitly or fixed to the latest
gift in the batch, never a moving "now".
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import pandas as pd

__all__ = ["civicrm_contributions_to_features", "read_civicrm_contributions"]

DAYS_PER_YEAR = 365.25

# Columns without which there is no gift log to aggregate.
_REQUIRED = ("contact_id", "receive_date", "total_amount")

_NON_ALNUM = re.compile(r"[^0-9a-z]+")

# Header normalisation (lower-case, non-alphanumerics to "_") already collapses
# most of the export-label vs. APIv4-name gap: "Total Amount" -> total_amount,
# "Contact ID" -> contact_id.  This maps the handful that still disagree onto
# the APIv4 name, which is the canonical spelling in this module.
_HEADER_ALIASES = {
    "contribution_id": "id",
    "contribution_date": "receive_date",  # the schema title for receive_date
    "date_received": "receive_date",
    "received_date": "receive_date",
    "amount": "total_amount",
    "contribution_amount": "total_amount",
    "contribution_source": "source",
    "test_mode": "is_test",
    "primary_email": "email",
    "email_address": "email",
    # APIv4 pseudo-fields: `Contribution.get` returns the numeric id under
    # `contribution_status_id` and the readable value under
    # `contribution_status_id:name` / `financial_type_id:label`.  Without these
    # two an API pull would silently arrive with no status column at all.
    "contribution_status_id_name": "contribution_status",
    "financial_type_id_label": "financial_type",
}

# Ordered (column, dtype) contract for the donor feature table.  Kept explicit
# so an empty export still yields a correctly-typed, downstream-safe frame.
_FEATURE_DTYPES: "dict[str, str]" = {
    "constituent_email": "object",
    "first_name": "object",
    "last_name": "object",
    "total_gift_amount": "float64",
    "gift_count": "int64",
    "largest_gift_amount": "float64",
    "first_gift_date": "datetime64[ns]",
    "last_gift_date": "datetime64[ns]",
    "years_active": "float64",
    "recency_days": "int64",
    "distinct_financial_types": "int64",
}

_TRUTHY = {"1", "true", "yes", "y", "t"}


def civicrm_contributions_to_features(
    contributions: Union[Iterable[Mapping], pd.DataFrame],
    *,
    reference_date: Optional[Union[str, pd.Timestamp]] = None,
    statuses: Optional[Sequence[str]] = ("Completed",),
) -> pd.DataFrame:
    """Aggregate a CiviCRM contribution log into donor-level features.

    Parameters
    ----------
    contributions : iterable of mapping, or DataFrame
        CiviCRM contribution rows. Accepts the output of
        :func:`read_civicrm_contributions`, an APIv4 ``Contribution.get``
        result, or a DataFrame of the same fields, under either the export
        labels or the APIv4 column names. ``contact_id``, ``receive_date`` and
        ``total_amount`` are required; ``id``, ``currency``,
        ``contribution_status``, ``financial_type``, ``is_test``, ``email``,
        ``first_name`` and ``last_name`` are used when present.
    reference_date : str or datetime-like, optional
        Anchor for the recency features (``years_active``, ``recency_days``).
        If ``None``, the latest ``receive_date`` in the batch is used; this
        keeps the aggregation reproducible and free of "now" leakage. Naive
        timestamps are interpreted as UTC.
    statuses : sequence of str or None, default=``("Completed",)``
        Contribution statuses to count, matched case-insensitively against
        ``contribution_status``. ``None`` disables the filter and counts every
        row, including ``Failed`` and ``Refunded`` ones. Test-mode rows are
        dropped either way.

    Returns
    -------
    features : pandas.DataFrame
        One row per donor, indexed by ``contact_id``, with the columns declared
        in ``_FEATURE_DTYPES``. ``recency_days``, ``gift_count`` and
        ``total_gift_amount`` are the R, F and M of an RFM model. Rows are
        sorted by ``contact_id`` for determinism.

    Raises
    ------
    KeyError
        If ``contact_id``, ``receive_date`` or ``total_amount`` is absent. An
        export missing one of them cannot be aggregated at all, and failing
        here names the field instead of surfacing a bare column lookup later.

    Warns
    -----
    UserWarning
        If ``statuses`` was requested but the batch carries no
        ``contribution_status`` column: the filter silently counting refunded
        and failed gifts is exactly the error this bridge exists to prevent.
    UserWarning
        If the batch mixes currencies. ``total_gift_amount`` is a plain sum with
        no FX conversion, so a single-currency export is assumed.

    Examples
    --------
    >>> rows = [
    ...     {"Contact ID": "101", "Contribution Date": "2025-01-15",
    ...      "Total Amount": "250.00", "Contribution Status": "Completed"},
    ...     {"Contact ID": "101", "Contribution Date": "2025-06-01",
    ...      "Total Amount": "1,000.00", "Contribution Status": "Completed"},
    ...     {"Contact ID": "101", "Contribution Date": "2025-06-02",
    ...      "Total Amount": "99.00", "Contribution Status": "Failed"},
    ... ]
    >>> feats = civicrm_contributions_to_features(rows)
    >>> float(feats.loc["101", "total_gift_amount"])
    1250.0
    >>> int(feats.loc["101", "gift_count"])
    2
    """
    df = _normalise_headers(_to_frame(contributions))
    if df.empty:
        return _empty_feature_frame()

    missing = [col for col in _REQUIRED if col not in df.columns]
    if missing:
        raise KeyError(
            f"CiviCRM contribution log is missing {missing}. Export the "
            f"'Contact ID', 'Contribution Date' and 'Total Amount' fields "
            f"(APIv4: contact_id, receive_date, total_amount); got "
            f"{sorted(df.columns)}."
        )

    df = df.copy()

    # Two rows sharing a CiviCRM contribution id are the same contribution;
    # concatenating overlapping monthly exports is how that happens. Collapse
    # only rows with a real id: pandas' duplicated() treats NaN == NaN, which
    # would drop every id-less gift once any row carries one.
    if "id" in df.columns:
        df = df[~(df["id"].notna() & df.duplicated(subset="id"))]

    if "is_test" in df.columns:
        df = df[~_to_bool(df["is_test"])]

    if statuses is not None:
        if "contribution_status" in df.columns:
            wanted = {str(s).strip().casefold() for s in statuses}
            status = df["contribution_status"].astype("string").str.strip().str.casefold()
            df = df[status.isin(wanted)]
        else:
            warnings.warn(
                f"CiviCRM contribution log has no contribution_status column, so "
                f"statuses={tuple(statuses)!r} could not be applied: Pending, "
                f"Failed, Refunded and Cancelled gifts (if any) are counted in "
                f"total_gift_amount. Add the 'Contribution Status' field to the "
                f"export, or pass statuses=None to accept every row.",
                stacklevel=2,
            )

    # total_gift_amount is a plain sum; CiviCRM carries a per-row `currency` but
    # the export has no FX rates, so a mixed-currency batch would add apples to
    # oranges. Warn rather than convert (rates aren't there) or crash.
    if "currency" in df.columns and df["currency"].dropna().nunique() > 1:
        warnings.warn(
            "CiviCRM contribution log mixes currencies "
            f"({sorted(df['currency'].dropna().unique())}); total_gift_amount is "
            "summed without FX conversion. Normalise to one currency first.",
            stacklevel=2,
        )

    df["_contact_id"] = df["contact_id"].astype("string").str.strip()
    df["_ts"] = _to_datetime(df["receive_date"])
    df["_amount"] = _to_amount(df["total_amount"]).fillna(0.0)
    # A gift we cannot attribute to a donor, or place in time, contributes to no
    # feature; drop it rather than let a NaT poison a donor's recency.
    df = df[
        df["_contact_id"].notna()
        & (df["_contact_id"].str.len() > 0)
        & df["_ts"].notna()
    ]
    if df.empty:
        return _empty_feature_frame()

    ref = _resolve_reference_date(reference_date, df["_ts"])

    grouped = df.groupby("_contact_id", sort=True)
    out = pd.DataFrame(index=grouped.size().index)
    # Optional identity fields: carried through when the export supplies them
    # (this is a donor-level table keyed by CRM id, not a de-identified feature
    # store). ``.first()`` skips nulls, so a donor whose name rode in on only
    # some rows still resolves. Absent column -> None.
    for out_col, src in (
        ("constituent_email", "email"),
        ("first_name", "first_name"),
        ("last_name", "last_name"),
    ):
        out[out_col] = grouped[src].first() if src in df.columns else None
    out["total_gift_amount"] = grouped["_amount"].sum()
    out["gift_count"] = grouped.size()
    out["largest_gift_amount"] = grouped["_amount"].max()
    out["first_gift_date"] = grouped["_ts"].min()
    out["last_gift_date"] = grouped["_ts"].max()
    out["years_active"] = (
        (ref - out["first_gift_date"]).dt.days / DAYS_PER_YEAR
    ).clip(lower=0.0)
    out["recency_days"] = (ref - out["last_gift_date"]).dt.days.clip(lower=0)

    # Financial Type is CiviCRM's gift classification (Donation, Member Dues,
    # Event Fee, ...); breadth across them is the CiviCRM analogue of the
    # UniSchema bridge's distinct_source_systems.
    type_col = next(
        (c for c in ("financial_type", "financial_type_id") if c in df.columns), None
    )
    out["distinct_financial_types"] = grouped[type_col].nunique() if type_col else 0

    out.index.name = "contact_id"
    return _coerce_schema(out)


def read_civicrm_contributions(path: Union[str, Path]) -> pd.DataFrame:
    """Read CiviCRM contribution export CSV(s) into one normalised frame.

    Accepts either a single ``.csv`` file or a directory, which is walked
    **recursively** and whose ``*.csv`` files are concatenated in sorted
    relative-path order, the shape you get from keeping a folder of monthly
    exports. Symlinks are not followed.

    Every column is read as text and the headers are normalised to the APIv4
    spelling, so ``"Total Amount"``, ``"Contact ID"`` and ``"Contribution Date"``
    arrive as ``total_amount``, ``contact_id`` and ``receive_date``. Nothing else
    is done to the rows: **test-mode and non-``Completed`` contributions are
    still present**, and it is
    :func:`civicrm_contributions_to_features` that drops them and types the
    values. Reading is deliberately lossless so the raw export stays inspectable.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV file, or a directory of them.

    Returns
    -------
    contributions : pandas.DataFrame
        The export as written, with normalised column names and text values.
        A directory holding no CSV returns an empty frame.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist. Without this an absent directory falls
        through to the single-file branch and surfaces as an opaque OSError.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file or directory: {p}")
    if p.is_dir():
        files = [
            f
            for f in p.rglob("*.csv")
            # Skip symlinks: one inside the export folder pointing outside it
            # must not be followed and read (path-traversal hardening).
            if f.is_file() and not f.is_symlink()
        ]
        if not files:
            return pd.DataFrame()
        # Sort by relative path so ordering is deterministic across platforms.
        frames = [
            _read_csv(f) for f in sorted(files, key=lambda f: f.relative_to(p).as_posix())
        ]
        return pd.concat(frames, ignore_index=True)
    return _read_csv(p)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _read_csv(path: Path) -> pd.DataFrame:
    # utf-8-sig: CiviCRM writes UTF-8, but an export round-tripped through Excel
    # comes back with a BOM that would otherwise ride along inside the first
    # header. dtype=str keeps a numeric-looking Contact ID from becoming an int
    # in one file and a float in another once a blank appears.
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=True)
    return _normalise_headers(df)


def _to_frame(contributions: Union[Iterable[Mapping], pd.DataFrame]) -> pd.DataFrame:
    if isinstance(contributions, pd.DataFrame):
        return contributions
    return pd.DataFrame(list(contributions))


def _canonical(header: str) -> str:
    key = _NON_ALNUM.sub("_", str(header).strip().lower()).strip("_")
    return _HEADER_ALIASES.get(key, key)


def _normalise_headers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty and df.columns.empty:
        return df
    out = df.rename(columns={c: _canonical(c) for c in df.columns})
    # A wide export can carry the same field twice ("Amount" and "Total
    # Amount"), which would collapse to two identically named columns and make
    # df["total_amount"] a DataFrame. Keep the first.
    return out.loc[:, ~out.columns.duplicated()]


def _to_amount(series: pd.Series) -> pd.Series:
    """Coerce a CiviCRM amount column to float.

    ponytail: strips grouping separators and currency symbols so an
    Excel-formatted ``"$1,250.00"`` parses. That assumes ``.`` is the decimal
    point; a site exporting ``1.250,00`` needs normalising upstream.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype("string").str.replace(r"[^\d.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _to_datetime(series: pd.Series) -> pd.Series:
    """Parse CiviCRM dates to tz-naive UTC (matches the datasets' naive dates).

    ponytail: CiviCRM stores ``receive_date`` as ``YYYY-MM-DD HH:MM:SS``, but a
    CSV export writes it in the site's configured date format, so a real export
    can carry ``03/01/2025``. ``format="mixed"`` parses both rather than
    rejecting one; it also resolves that example as *March 1* under pandas'
    ``dayfirst=False`` default. A site configured ``dd/mm/yyyy`` should pass an
    already-parsed datetime column.
    """
    ts = pd.to_datetime(series, format="mixed", errors="coerce", utc=True)
    return ts.dt.tz_localize(None)


def _to_bool(series: pd.Series) -> pd.Series:
    """CiviCRM writes is_test as 1/0 over APIv4 and as text in a CSV export."""
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return series.fillna(0).astype(bool)
    return series.astype("string").str.strip().str.casefold().isin(_TRUTHY)


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
    out.index = pd.Index([], name="contact_id", dtype="object")
    return out


def _coerce_schema(out: pd.DataFrame) -> pd.DataFrame:
    return out[list(_FEATURE_DTYPES)].astype(_FEATURE_DTYPES)
