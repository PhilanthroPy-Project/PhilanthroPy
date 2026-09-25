"""
philanthropy.ingest._upgrade_snapshots
=======================================
Build a per-donor, per-fiscal-year training table for an upgrade model: will
a donor currently giving in a mid-level band move up to a leadership level
next year?

``build_upgrade_snapshots`` turns a gift log (optionally joined with an
activity log and static donor attributes) into one row per
``(donor, fiscal year T)`` for every donor whose FY T giving falls inside the
"upgrade band", plus a ``target`` column: did that donor cross ``threshold``
in FY T+1? Everything a row's features touch is dated at or before the end
of FY T; the target alone is read from FY T+1, and nothing later.

This is a plain function, not an estimator: it changes the row count (one
gift log row becomes zero or one donor-year row) and manufactures a label
column, neither of which an ``sklearn`` ``transform()`` can do. It lives in
``philanthropy.ingest`` because it is the same shape as every other bridge in
this subpackage, raw/normalised source tables in, one donor-level feature
table out, and it reuses :func:`~philanthropy.ingest.activities_to_features`
directly when an activity log is supplied.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Tuple, Union

import pandas as pd

from philanthropy.preprocessing import FiscalYearTransformer
from philanthropy.utils._validation import validate_fiscal_year_start

from ._activities import activities_to_features

__all__ = ["build_upgrade_snapshots"]

_REQUIRED = ("donor_id", "gift_date", "gift_amount")


def build_upgrade_snapshots(
    gifts: Union[Iterable[Mapping], pd.DataFrame],
    *,
    fiscal_years: Iterable[int],
    threshold: float = 1000,
    band: Tuple[float, float] = (100, 999),
    fiscal_year_start: int = 7,
    activities: Optional[Union[Iterable[Mapping], pd.DataFrame]] = None,
    donors: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build upgrade-candidate snapshots, one row per (donor, fiscal year).

    For each fiscal year ``T`` in ``fiscal_years``, the population is every
    donor whose FY T total giving falls inside ``band`` (inclusive) and below
    ``threshold``: donors already at or above ``threshold`` in T are not
    "upgrade candidates" and get no row for that T. Every feature is computed
    from gift (and, if given, activity) rows dated at or before the end of FY
    T; ``target`` is 1 if the donor's FY T+1 total reaches ``threshold``, else
    0, including when the donor gave nothing at all in T+1.

    Parameters
    ----------
    gifts : iterable of mapping, or DataFrame
        Gift-level rows, already normalised (not a raw CRM export): each row
        needs ``donor_id``, ``gift_date`` and ``gift_amount``, the same
        columns :class:`~philanthropy.preprocessing.RFMTransformer` and
        :func:`~philanthropy.datasets.make_donor_panel` use. Extra columns
        are ignored.
    fiscal_years : iterable of int
        The snapshot years ``T`` to build. One donor can appear once per year
        it qualified for the band, so passing several years stacks rows.
    threshold : float, default=1000
        The leadership-giving level an upgrade crosses into. Also the upper
        exclusive bound of the candidate population: a donor at or above this
        in FY T already gave at that level and is not a candidate.
    band : (float, float), default=(100, 999)
        Inclusive ``(low, high)`` bounds on FY T total giving that define the
        upgrade-candidate population. ``high`` should sit below ``threshold``
        for the two limits to describe one band; a donor is still excluded
        once the FY T total reaches ``threshold`` even if ``high`` does not.
    fiscal_year_start : int, default=7
        Month (1-12) the organisation's fiscal year begins, the same
        parameter :class:`~philanthropy.preprocessing.FiscalYearTransformer`
        takes; fiscal-year boundaries are computed with that transformer so
        the two stay in agreement.
    activities : iterable of mapping, or DataFrame, optional
        A long activity log in the shape
        :func:`~philanthropy.ingest.activities_to_features` expects
        (``contact_id``, ``activity_date``, ``activity_type``, ...). When
        given, that function is called once per snapshot year with
        ``as_of`` set to the end of FY T, and its columns are joined in.
    donors : DataFrame, optional
        Static donor attributes (constituency, wealth rating, ...), indexed
        by donor id. Joined in as-is; also passed through to
        ``activities_to_features`` for its match-rate check when
        ``activities`` is given. The caller is responsible for not including
        a column that already encodes the answer (e.g. a precomputed giving
        tier); this function does not attempt to detect that.

    Returns
    -------
    snapshots : pandas.DataFrame
        One row per qualifying ``(donor, T)``, indexed by ``donor_id``, sorted
        by ``fiscal_year`` then ``donor_id``. Columns: ``fiscal_year`` (the
        snapshot year T, not T+1); ``fy_total``, ``fy_total_prior1``,
        ``fy_total_prior2`` (summed gift amount in FY T, T-1, T-2); ``fy_trend``
        (``fy_total - fy_total_prior1``); ``largest_gift`` and ``gift_count``
        (within FY T); ``consecutive_years_given`` (count of unbroken prior
        fiscal years, ending at and including T, with positive giving);
        ``months_since_last_gift`` (from the donor's most recent gift on or
        before the end of FY T); any ``activities_to_features`` columns;
        any ``donors`` columns; and ``target``. A fiscal year with no
        qualifying donors contributes no rows. Returns an empty, columnless
        frame (index name ``donor_id``) if no year has any.

    Raises
    ------
    KeyError
        If ``gifts`` is missing ``donor_id``, ``gift_date`` or
        ``gift_amount``.
    ValueError
        If ``fiscal_year_start`` is not between 1 and 12, or ``band[0] >
        band[1]``.

    Examples
    --------
    >>> import pandas as pd
    >>> from philanthropy.ingest import build_upgrade_snapshots
    >>> gifts = pd.DataFrame({
    ...     "donor_id": ["1", "1", "1", "1", "2"],
    ...     "gift_date": ["2017-08-01", "2018-08-01", "2019-08-01",
    ...                   "2020-08-01", "2019-08-01"],
    ...     "gift_amount": [200, 300, 500, 1500, 5000],
    ... })
    >>> snaps = build_upgrade_snapshots(gifts, fiscal_years=[2020])
    >>> list(snaps.index)
    ['1']
    >>> int(snaps.loc["1", "fiscal_year"])
    2020
    >>> float(snaps.loc["1", "fy_total"])
    500.0
    >>> int(snaps.loc["1", "consecutive_years_given"])
    3
    >>> int(snaps.loc["1", "target"])
    1

    Donor "2" gave 5000 in FY2020, at or above ``threshold``, so is excluded
    from the band even though it never appears in ``snaps``:

    >>> "2" in snaps.index
    False
    """
    validate_fiscal_year_start(fiscal_year_start)
    low, high = band
    if low > high:
        raise ValueError(f"band[0] ({low}) must be <= band[1] ({high}).")

    df, pivot_sum, pivot_max, pivot_count = _prepare_gifts(gifts, fiscal_year_start)

    donors_norm = None
    if donors is not None:
        donors_norm = donors.copy()
        donors_norm.index = donors_norm.index.astype("string").str.strip()

    snapshots = []
    for fy_t in fiscal_years:
        fy_t = int(fy_t)
        totals_t = _column(pivot_sum, fy_t)
        candidates = totals_t[(totals_t >= low) & (totals_t <= high) & (totals_t < threshold)]
        if candidates.empty:
            continue
        donor_ids = candidates.index

        snap = _snapshot_features_for_year(
            df, pivot_sum, pivot_max, pivot_count, donor_ids, fy_t,
            fiscal_year_start, activities, donors, donors_norm,
        )

        next_totals = _column(pivot_sum, fy_t + 1).reindex(donor_ids, fill_value=0.0)
        snap["target"] = (next_totals >= threshold).astype("int64")

        snapshots.append(snap)

    if not snapshots:
        return pd.DataFrame(index=pd.Index([], name="donor_id", dtype="object"))

    out = pd.concat(snapshots)
    out = out.reset_index().sort_values(["fiscal_year", "donor_id"], kind="stable")
    return out.set_index("donor_id")


# --------------------------------------------------------------------------- #
# Internals
#
# ``_prepare_gifts`` and ``_snapshot_features_for_year`` are also imported
# directly (via ``philanthropy.ingest._upgrade_snapshots``) by
# ``philanthropy.models.score_upgrade_prospects``, which needs the same
# donor/fiscal-year feature logic for an unlabelled "current" row that this
# function's target computation (reading FY T+1) does not apply to.
# --------------------------------------------------------------------------- #
def _to_frame(gifts: Union[Iterable[Mapping], pd.DataFrame]) -> pd.DataFrame:
    if isinstance(gifts, pd.DataFrame):
        return gifts
    return pd.DataFrame(list(gifts))


def _prepare_gifts(
    gifts: Union[Iterable[Mapping], pd.DataFrame], fiscal_year_start: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalise a raw gift log and pivot it donor x fiscal-year.

    Returns ``(df, pivot_sum, pivot_max, pivot_count)``: ``df`` is the
    cleaned, per-gift frame (with ``donor_id``, ``_date``, ``_amount`` and
    ``_fy`` columns) and the three pivots aggregate ``_amount`` over it.
    """
    df = _to_frame(gifts)
    missing = [col for col in _REQUIRED if col not in df.columns]
    if missing:
        raise KeyError(
            f"gifts is missing {missing}. Every gift row needs 'donor_id', "
            f"'gift_date' and 'gift_amount'; got {sorted(df.columns)}."
        )

    df = df.copy()
    df["donor_id"] = df["donor_id"].astype("string").str.strip()
    df["_date"] = pd.to_datetime(df["gift_date"], errors="coerce")
    df["_amount"] = pd.to_numeric(df["gift_amount"], errors="coerce")
    df = df[df["donor_id"].notna() & df["_date"].notna() & df["_amount"].notna()]

    if not df.empty:
        fy = (
            FiscalYearTransformer(fiscal_year_start=fiscal_year_start)
            .set_output(transform="pandas")
            .fit_transform(df[["_date"]].rename(columns={"_date": "gift_date"}))
        )
        df["_fy"] = fy["fiscal_year"].to_numpy()
        df = df[df["_fy"].notna()]
        df["_fy"] = df["_fy"].astype("int64")

    pivot_sum = _pivot(df, "sum")
    pivot_max = _pivot(df, "max")
    pivot_count = _pivot(df, "count")
    return df, pivot_sum, pivot_max, pivot_count


def _snapshot_features_for_year(
    df: pd.DataFrame,
    pivot_sum: pd.DataFrame,
    pivot_max: pd.DataFrame,
    pivot_count: pd.DataFrame,
    donor_ids: pd.Index,
    fy_t: int,
    fiscal_year_start: int,
    activities: Optional[Union[Iterable[Mapping], pd.DataFrame]],
    donors: Optional[pd.DataFrame],
    donors_norm: Optional[pd.DataFrame],
    as_of: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Gift-derived (and, if given, activity/donor) feature columns for one
    ``(donor_ids, fy_t)`` snapshot, everything ``build_upgrade_snapshots``
    computes except ``target``.

    ``as_of`` clips the recency cutoff (``months_since_last_gift`` and the
    ``activities_to_features`` window) to a date inside a still-open fiscal
    year, for scoring a "current", not-yet-resolved FY; it defaults to the
    end of ``fy_t`` (``build_upgrade_snapshots``'s own, always-resolved case).
    """
    snap = pd.DataFrame(index=donor_ids)
    snap.index.name = "donor_id"
    snap["fiscal_year"] = fy_t
    snap["fy_total"] = _column(pivot_sum, fy_t).reindex(donor_ids, fill_value=0.0)
    snap["fy_total_prior1"] = _column(pivot_sum, fy_t - 1).reindex(donor_ids, fill_value=0.0)
    snap["fy_total_prior2"] = _column(pivot_sum, fy_t - 2).reindex(donor_ids, fill_value=0.0)
    snap["fy_trend"] = snap["fy_total"] - snap["fy_total_prior1"]
    snap["largest_gift"] = _column(pivot_max, fy_t).reindex(donor_ids, fill_value=0.0)
    snap["gift_count"] = _column(pivot_count, fy_t).reindex(donor_ids, fill_value=0).astype("int64")
    snap["consecutive_years_given"] = _consecutive_years_given(pivot_sum, donor_ids, fy_t)

    cutoff = _fy_end(fy_t, fiscal_year_start)
    if as_of is not None and as_of < cutoff:
        cutoff = as_of
    last_gift = df[df["_fy"] <= fy_t].groupby("donor_id")["_date"].max().reindex(donor_ids)
    snap["months_since_last_gift"] = ((cutoff - last_gift).dt.days / 30.0).round(2)

    if activities is not None:
        act_feats = activities_to_features(activities, as_of=cutoff, donors=donors)
        snap = snap.join(act_feats, how="left")

    if donors_norm is not None:
        snap = snap.join(donors_norm, how="left")

    return snap


def _pivot(df: pd.DataFrame, aggfunc: str) -> pd.DataFrame:
    """Donor x fiscal-year table of ``aggfunc`` over gift amounts.

    Empty when there is no valid gift row; a fiscal year with no data at all
    is simply absent as a column, never a column of zeros.
    """
    if df.empty:
        return pd.DataFrame()
    fill = 0
    return (
        df.groupby(["donor_id", "_fy"])["_amount"]
        .agg(aggfunc)
        .unstack(fill_value=fill)
    )


def _column(pivot: pd.DataFrame, fy: int) -> pd.Series:
    """The pivot's column for ``fy``, or an empty float series if absent."""
    if fy not in pivot.columns:
        return pd.Series(dtype="float64")
    return pivot[fy]


def _consecutive_years_given(
    pivot_sum: pd.DataFrame, donor_ids: pd.Index, fy_t: int
) -> pd.Series:
    """Unbroken run of fiscal years with positive giving, ending at ``fy_t``.

    A plain per-donor walk backwards from ``fy_t``: donor counts are small
    (one row per candidate) and streak lengths are bounded by the years on
    file, so this stays a simple loop rather than a vectorised scan.
    """
    counts = []
    for donor_id in donor_ids:
        streak = 0
        year = fy_t
        while True:
            total = float(_column(pivot_sum, year).get(donor_id, 0.0))
            if total <= 0:
                break
            streak += 1
            year -= 1
        counts.append(streak)
    return pd.Series(counts, index=donor_ids, dtype="int64")


def _fy_end(fy: int, fiscal_year_start: int) -> pd.Timestamp:
    """The last calendar day of fiscal year ``fy``.

    Matches :class:`~philanthropy.preprocessing.FiscalYearTransformer`'s
    convention (``fiscal_year = year + 1`` once the month reaches
    ``fiscal_year_start``): FY ``fy`` ends the day before ``fiscal_year_start``
    rolls over in calendar year ``fy``.
    """
    return pd.Timestamp(year=fy, month=fiscal_year_start, day=1) - pd.Timedelta(days=1)
