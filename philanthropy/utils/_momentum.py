"""
philanthropy.utils._momentum
=============================
Shared, as-of trailing-window slope features for an annual (or other
evenly-spaced) time series, used by :class:`~philanthropy.preprocessing.RFMTransformer`,
:func:`~philanthropy.ingest.build_upgrade_snapshots` /
:func:`~philanthropy.models.score_upgrade_prospects`, and
:func:`~philanthropy.ingest.activities_to_features`, so every caller computes
"momentum" the same way.

Bins are trailing 12-month windows anchored at the caller's own cutoff (a
fiscal-year end for a resolved historical row, ``as_of`` for a still-open
"current" row), not calendar fiscal-year columns: this is what makes a
partial, still-open fiscal year behave identically to a resolved one scored
at the same cutoff, and what lets the same code serve a longer "period" (for
example a biennial survey wave) by passing ``period_months=24``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["trailing_slope_features"]


def _ols_slope(values: np.ndarray) -> np.ndarray:
    """Closed-form OLS slope of each column of ``values`` against ``0..k-1``.

    ``values`` has shape ``(k, n_rows)``, oldest period first. Evenly-spaced
    x values make the slope a plain covariance-over-variance ratio, no
    per-row linear-algebra solve needed.
    """
    k = values.shape[0]
    x = np.arange(k, dtype="float64")
    x_centered = x - x.mean()
    denom = float((x_centered ** 2).sum())
    y_centered = values - values.mean(axis=0, keepdims=True)
    return (x_centered[:, None] * y_centered).sum(axis=0) / denom


def trailing_slope_features(
    df: pd.DataFrame,
    donor_ids: pd.Index,
    cutoff: pd.Timestamp,
    *,
    date_col: str,
    value_col: str,
    agg: str,
    ks: Sequence[int] = (3, 5),
    prefix: str,
    donor_col: str = "donor_id",
    period_months: int = 12,
    data_start: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Trailing-period OLS slope and relative slope of one base series.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format rows (gifts or activities) with ``donor_col``,
        ``date_col`` and ``value_col``. Rows after ``cutoff`` are ignored.
    donor_ids : pandas.Index
        The population to return a row for.
    cutoff : pandas.Timestamp
        The as-of date bin 0 (the most recent period) ends at.
    date_col, value_col : str
        Column names in ``df``.
    agg : str
        Aggregation per period bin (``"sum"``, ``"count"``, or ``"max"``).
    ks : sequence of int, default=(3, 5)
        Trailing window lengths, in periods.
    prefix : str
        Output columns are ``f"{prefix}_slope_{k}y"`` and
        ``f"{prefix}_rel_slope_{k}y"`` (the ``y`` suffix names the period as
        "year-like" regardless of ``period_months``, matching the annual
        default every current caller uses).
    period_months : int, default=12
        Width of one period, in months. The default is one year; a future
        caller on a biennial cadence (for example a two-year survey wave)
        passes 24.
    data_start : pandas.Timestamp, optional
        The earliest date this series can be considered observed from (for
        example a frozen training-set minimum). A period whose window ends
        before this date is not "observed" -- it lies entirely before the
        data begins, so a 0 there is unknown, not measured -- and the slope
        over a window with fewer than 2 observed periods is ``NaN``. A
        period whose window ends on or after ``data_start`` is observed and
        contributes its value, 0 if it has no rows (a period with no rows,
        inside the data's own span, is a known zero, not a missing value).
        Defaults to ``df[date_col].min()``.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``donor_ids``, two columns per ``k`` in ``ks``.
    """
    df = df[df[date_col] <= cutoff]
    if data_start is None:
        data_start = df[date_col].min() if len(df) else cutoff

    max_k = max(ks)
    bins = np.zeros((max_k, len(donor_ids)), dtype="float64")
    observed = np.zeros(max_k, dtype=bool)
    for j in range(max_k):
        hi = cutoff - pd.DateOffset(months=period_months * j)
        lo = cutoff - pd.DateOffset(months=period_months * (j + 1))
        window = df[(df[date_col] > lo) & (df[date_col] <= hi)]
        if len(window):
            agg_vals = window.groupby(donor_col)[value_col].agg(agg)
            bins[j] = agg_vals.reindex(donor_ids, fill_value=0.0).to_numpy()
        observed[j] = pd.notna(data_start) and hi >= data_start

    out = {}
    for k in ks:
        # bins[0] is the most recent period; chronological order is oldest first.
        window_vals = bins[k - 1::-1]
        n_observed = int(observed[:k].sum())
        if n_observed < 2:
            slope = np.full(len(donor_ids), np.nan)
            rel_slope = np.full(len(donor_ids), np.nan)
        else:
            slope = _ols_slope(window_vals)
            rel_slope = slope / (window_vals.mean(axis=0) + 1.0)
        out[f"{prefix}_slope_{k}y"] = slope
        out[f"{prefix}_rel_slope_{k}y"] = rel_slope
    return pd.DataFrame(out, index=donor_ids)
