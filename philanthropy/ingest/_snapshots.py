"""
philanthropy.ingest._snapshots
==============================
One snapshot builder for every question a donor model answers: will this
donor upgrade, lapse, give again, and how much next time.

``build_snapshots`` turns a gift log into one row per ``(donor, period T)``
with a label read from period T+1 and features read from period T and
earlier only. Its core, ``period_snapshots``, works on a donor x period
table instead of a gift log, so a file that is not a gift log (a biennial
household survey, a promotion history) gets exactly the same columns: a
"period" is whatever the file's own unit is, and "prior" always means the
previous period observed in the file, never ``T - 1`` arithmetic.

Like :func:`~philanthropy.ingest.build_leadership_snapshots`, these are
plain functions, not estimators: they change the row count and manufacture
a label, which no ``transform()`` can do.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from philanthropy.utils._validation import validate_fiscal_year_start

from ._upgrade_snapshots import _fy_end, _prepare_gifts

__all__ = ["build_snapshots", "period_snapshots", "SNAPSHOT_KINDS", "CORE_COLUMNS", "SCALE_FREE_COLUMNS"]

SNAPSHOT_KINDS = ("upgrade", "lapse", "response_next_year", "next_amount")

#: Columns every kind emits from any donor x period table.
CORE_COLUMNS = (
    "period_total", "period_total_prior1", "period_total_prior2", "period_trend", "largest_gift",
    "consecutive_periods_given", "gave_prior1", "gave_prior2", "periods_since_first_gift",
)

#: Ratio and rank columns that do not depend on the file's dollar scale.
SCALE_FREE_COLUMNS = (
    "total_over_largest", "largest_over_prior_total", "gifts_per_streak_period", "period_total_pct_rank",
)


def build_snapshots(
    gifts: Union[Iterable[Mapping], pd.DataFrame],
    *,
    kind: str,
    fiscal_years: Optional[Iterable[int]] = None,
    threshold: float = 1000.0,
    band: Tuple[float, float] = (100.0, 999.0),
    min_years_given: int = 1,
    fiscal_year_start: int = 7,
    scale_free: bool = False,
) -> pd.DataFrame:
    """Build labelled donor x fiscal-year snapshots for one question.

    Every feature is computed from gifts dated at or before the end of
    fiscal year ``T``; the label alone is read from ``T + 1``. A fiscal year
    with no following year in the file contributes no rows, so "gave
    nothing next year" and "next year not in the file" are never confused.

    Parameters
    ----------
    gifts : iterable of mapping, or DataFrame
        Gift-level rows with ``donor_id``, ``gift_date`` and ``gift_amount``.
    kind : {"upgrade", "lapse", "response_next_year", "next_amount"}
        Which population and label (see :func:`period_snapshots`).
    fiscal_years : iterable of int, optional
        Snapshot years ``T`` to keep. Default: every year with a following
        year in the file.
    threshold, band : float, (float, float)
        The upgrade question's leadership level and candidate band, as in
        :func:`~philanthropy.ingest.build_leadership_snapshots`. Ignored by
        the other kinds.
    min_years_given : int, default=1
        Keep only donors who gave in at least this many fiscal years up to
        and including ``T``. ``2`` restricts lapse to multi-year donors, the
        donors a retention program can act on.
    fiscal_year_start : int, default=7
        Month (1-12) the fiscal year begins.
    scale_free : bool, default=False
        Also emit :data:`SCALE_FREE_COLUMNS`.

    Returns
    -------
    snapshots : pandas.DataFrame
        Indexed by ``donor_id``, sorted by ``period`` then ``donor_id``:
        ``period`` (the fiscal year T), :data:`CORE_COLUMNS`,
        ``gift_count`` (gifts in T), ``months_since_last_gift`` (to the end
        of T), the scale-free columns when asked, and ``target``.

    Examples
    --------
    >>> import pandas as pd
    >>> from philanthropy.ingest import build_snapshots
    >>> gifts = pd.DataFrame({
    ...     "donor_id": ["1", "1", "2", "2", "2"],
    ...     "gift_date": ["2018-08-01", "2019-08-01", "2018-08-01",
    ...                   "2019-08-01", "2020-08-01"],
    ...     "gift_amount": [50, 60, 20, 30, 40],
    ... })
    >>> snaps = build_snapshots(gifts, kind="lapse", min_years_given=2)
    >>> snaps["period"].tolist(), snaps["target"].tolist()
    ([2020, 2020], [1, 0])

    Donor "1" gave in FY2019 and FY2020 and nothing in FY2021, so it is a
    lapse; donor "2" gave again. FY2019 has no rows: neither donor had two
    giving years by then.
    """
    validate_fiscal_year_start(fiscal_year_start)
    df, pivot_sum, pivot_max, pivot_count = _prepare_gifts(gifts, fiscal_year_start)
    if df.empty:
        return pd.DataFrame(index=pd.Index([], name="donor_id"))
    # Fiscal years are contiguous: a year with no gift at all on file is
    # still a period in which every donor gave $0.
    years = list(range(int(df["_fy"].min()), int(df["_fy"].max()) + 1))
    totals = pivot_sum.reindex(columns=years, fill_value=0.0).fillna(0.0)
    largest = pivot_max.reindex(columns=years, fill_value=0.0).fillna(0.0)
    counts = pivot_count.reindex(columns=years, fill_value=0).fillna(0)

    snap = period_snapshots(
        totals, kind=kind, largest=largest, counts=counts, periods=fiscal_years,
        threshold=threshold, band=band, min_years_given=min_years_given, scale_free=scale_free,
    )
    if snap.empty:
        return snap

    last_date = {}
    for fy in snap["period"].unique():
        fy = int(fy)
        last_date[fy] = (
            (_fy_end(fy, fiscal_year_start) - df[df["_fy"] <= fy].groupby("donor_id")["_date"].max()).dt.days / 30.0
        ).round(2)
    snap["months_since_last_gift"] = [
        last_date[int(p)].get(d, np.nan) for d, p in zip(snap.index, snap["period"])
    ]
    target = snap.pop("target")
    snap["target"] = target
    return snap


def period_snapshots(
    totals: pd.DataFrame,
    *,
    kind: str,
    largest: Optional[pd.DataFrame] = None,
    counts: Optional[pd.DataFrame] = None,
    periods: Optional[Iterable] = None,
    threshold: float = 1000.0,
    band: Tuple[float, float] = (100.0, 999.0),
    min_years_given: int = 1,
    scale_free: bool = False,
) -> pd.DataFrame:
    """Labelled snapshots from a donor x period table of total giving.

    ``totals`` has one row per donor and one column per period, in time
    order; ``NaN`` means the donor was not observed in that period (survey
    attrition), which is different from giving ``0``. A row is only labelled
    from a next period in which the donor was observed. ``largest`` and
    ``counts``, when given, are the same shape: largest single gift and
    number of gifts in the period. Without ``largest`` the period total
    stands in for it.

    Populations and labels, for a snapshot period T:

    - ``"upgrade"``: total in ``band`` and below ``threshold``; target 1 if
      the next period's total reaches ``threshold``.
    - ``"lapse"``: gave (total > 0) in T; target 1 if the next period's
      total is 0.
    - ``"response_next_year"``: gave in T or any earlier period; target 1 if
      the next period's total is above 0.
    - ``"next_amount"``: gave in T and in the next period; target is the
      next period's total.

    Every kind then keeps only donors who gave in at least
    ``min_years_given`` periods up to and including T.

    Returns
    -------
    snapshots : pandas.DataFrame
        Indexed like ``totals``, with ``period``, :data:`CORE_COLUMNS`,
        ``gift_count`` when ``counts`` is given, the scale-free columns when
        ``scale_free`` (``gifts_per_streak_period`` only with ``counts``),
        and ``target``.
    """
    if kind not in SNAPSHOT_KINDS:
        raise ValueError(f"kind must be one of {SNAPSHOT_KINDS}; got {kind!r}.")
    if min_years_given < 1:
        raise ValueError(f"min_years_given must be >= 1; got {min_years_given}.")
    low, high = band
    if low > high:
        raise ValueError(f"band[0] ({low}) must be <= band[1] ({high}).")

    cols = list(totals.columns)
    gave = totals.fillna(0.0) > 0
    n_given = gave.cumsum(axis=1)
    first_pos = gave.to_numpy().argmax(axis=1).astype(float)
    first_pos[~gave.to_numpy().any(axis=1)] = np.nan
    keep_periods = None if periods is None else {p for p in periods}

    rows = []
    for i, period in enumerate(cols[:-1]):
        if keep_periods is not None and period not in keep_periods:
            continue
        cur, nxt = totals[period], totals[cols[i + 1]]
        observed = cur.notna() & nxt.notna()
        if kind == "upgrade":
            cand = (cur >= low) & (cur <= high) & (cur < threshold)
        elif kind == "lapse" or kind == "next_amount":
            cand = cur > 0
        else:
            cand = n_given[period] > 0
        cand &= observed & (n_given[period] >= min_years_given)
        if kind == "next_amount":
            cand &= nxt > 0
        ids = totals.index[cand.to_numpy()]
        if len(ids) == 0:
            continue

        snap = pd.DataFrame(index=ids)
        snap["period"] = period
        snap["period_total"] = cur[ids]
        snap["period_total_prior1"] = totals[cols[i - 1]][ids] if i >= 1 else np.nan
        snap["period_total_prior2"] = totals[cols[i - 2]][ids] if i >= 2 else np.nan
        snap["period_trend"] = snap["period_total"] - snap["period_total_prior1"]
        snap["largest_gift"] = (largest[period][ids] if largest is not None else snap["period_total"])
        snap["consecutive_periods_given"] = _streak(gave, ids, i)
        snap["gave_prior1"] = (snap["period_total_prior1"].fillna(0.0) > 0).astype("int64")
        snap["gave_prior2"] = (snap["period_total_prior2"].fillna(0.0) > 0).astype("int64")
        snap["periods_since_first_gift"] = i - pd.Series(first_pos, index=totals.index)[ids]
        if counts is not None:
            snap["gift_count"] = counts[period][ids].astype("int64")
        if scale_free:
            snap["total_over_largest"] = snap["period_total"] / snap["largest_gift"].where(snap["largest_gift"] > 0)
            snap["largest_over_prior_total"] = (
                snap["largest_gift"] / snap["period_total_prior1"].where(snap["period_total_prior1"] > 0)
            ).clip(upper=10.0)
            if counts is not None:
                snap["gifts_per_streak_period"] = snap["gift_count"] / snap["consecutive_periods_given"].where(
                    snap["consecutive_periods_given"] > 0
                )
            snap["period_total_pct_rank"] = snap["period_total"].rank(pct=True)

        nxt_ids = nxt[ids]
        if kind == "upgrade":
            snap["target"] = (nxt_ids >= threshold).astype("int64")
        elif kind == "lapse":
            snap["target"] = (nxt_ids <= 0).astype("int64")
        elif kind == "response_next_year":
            snap["target"] = (nxt_ids > 0).astype("int64")
        else:
            snap["target"] = nxt_ids.astype("float64")
        rows.append(snap)

    index_name = totals.index.name or "donor_id"
    if not rows:
        return pd.DataFrame(index=pd.Index([], name=index_name))
    out = pd.concat(rows)
    out.index.name = index_name
    out = out.reset_index().sort_values(["period", str(index_name)], kind="stable")
    return out.set_index(index_name)


def _streak(gave: pd.DataFrame, ids: pd.Index, i: int) -> pd.Series:
    """Unbroken run of periods with giving, ending at and including column
    ``i``. An unobserved period breaks the run, the same as a period with
    no gift."""
    block = gave.loc[ids].to_numpy()[:, : i + 1][:, ::-1]
    run = np.where(block.all(axis=1), block.shape[1], (~block).argmax(axis=1))
    return pd.Series(run, index=ids, dtype="int64")
