"""
philanthropy.models._upgrade
=============================
Fit-and-score entry point for the mid-level-to-leadership "upgrade" model:
which currently mid-level donors are most likely to cross the leadership
``threshold`` next fiscal year?

``score_upgrade_prospects`` is the one-call version of the workflow
:func:`~philanthropy.ingest.build_upgrade_snapshots` sets up: every
fully-resolved historical ``(donor, fiscal year)`` pair becomes a training
row, a :class:`~philanthropy.models.MajorGiftClassifier` is fit on the stack
with a walk-forward, fiscal-year-aware validation split, and the CURRENT
band-qualifying donors (an unlabelled snapshot as of ``as_of``) are scored
with a model refit on every historical row.

This lives in ``philanthropy.models``, not ``philanthropy.ingest`` where
``build_upgrade_snapshots`` lives: unlike that function, which only reshapes
tables (no estimator involved, by that module's own design), this function's
work is mostly a model-selection split, a classifier fit, and a permutation-
importance call. It still calls ``build_upgrade_snapshots`` directly for the
historical half rather than re-deriving the same target/leakage logic, the
same reasoning that function gives for reusing ``activities_to_features``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from philanthropy.ingest import build_upgrade_snapshots
from philanthropy.ingest._upgrade_snapshots import (
    _column,
    _fy_end,
    _prepare_gifts,
    _snapshot_features_for_year,
)
from philanthropy.inspection import donor_feature_importance
from philanthropy.model_selection import FiscalYearGroupedSplitter

from ._propensity import MajorGiftClassifier

__all__ = ["score_upgrade_prospects"]

_TOP_REASONS = 3
_LOW_DATA_ROWS = 500
# CalibratedClassifierCV (inside MajorGiftClassifier) defaults to a 5-fold
# cross-validation internally, which needs at least this many historical
# rows to split at all; fewer crashes deep inside sklearn ("Cannot have
# number of splits n_splits=5 greater than the number of samples") instead of
# failing with a message that points at the actual cause.
_MIN_TRAINING_ROWS = 5


def score_upgrade_prospects(
    gifts: Union[Iterable[Mapping], pd.DataFrame],
    *,
    activities: Optional[Union[Iterable[Mapping], pd.DataFrame]] = None,
    donors: Optional[pd.DataFrame] = None,
    threshold: float = 1000.0,
    band: Tuple[float, float] = (100.0, 999.0),
    fiscal_year_start: int = 7,
    as_of: Optional[Union[str, pd.Timestamp]] = None,
    top_n: Optional[int] = None,
    baseline_giving_threshold: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fit an upgrade model on history, then score today's band-qualifying donors.

    Every fiscal year ``T`` where both ``T`` and ``T+1`` are fully resolved as
    of ``as_of`` is a labelled training row (via
    :func:`~philanthropy.ingest.build_upgrade_snapshots`); a
    :class:`~philanthropy.models.MajorGiftClassifier` is fit on the stack of
    those years and evaluated on the most recent
    :class:`~philanthropy.model_selection.FiscalYearGroupedSplitter` fold, an
    honest, held-out read. A second copy of the model, refit on every
    historical row (more data, no held-out fold to protect), scores the
    CURRENT band-qualifying donors: one unlabelled row per donor, built the
    same way but for the fiscal year containing ``as_of`` and cut at ``as_of``
    rather than at that year's end, since it may still be in progress.

    Parameters
    ----------
    gifts : iterable of mapping, or DataFrame
        Gift-level rows with ``donor_id``, ``gift_date`` and ``gift_amount``,
        the same shape :func:`~philanthropy.ingest.build_upgrade_snapshots`
        takes.
    activities : iterable of mapping, or DataFrame, optional
        A long activity log, forwarded to
        :func:`~philanthropy.ingest.build_upgrade_snapshots` for the
        historical years and to the same feature logic for the current row.
    donors : DataFrame, optional
        Static donor attributes, indexed by donor id. Forwarded the same way.
        Only its numeric columns become model features (see Notes); it is
        still joined and returned for a non-numeric attribute a caller wants
        to inspect alongside the score.
    threshold : float, default=1000.0
        The leadership-giving level an upgrade crosses into.
    band : (float, float), default=(100.0, 999.0)
        Inclusive bounds on FY giving that define the upgrade-candidate
        population, exactly as in ``build_upgrade_snapshots``.
    fiscal_year_start : int, default=7
        Month (1-12) the fiscal year begins.
    as_of : str or datetime-like, optional
        The scoring cutoff: nothing dated after it is ever read, in either
        the historical training rows or the current scored row. Defaults to
        the latest gift date in ``gifts``, the same leakage-free default
        every other ``reference_date``-style parameter in this package uses.
    top_n : int, optional
        How many of the held-out validation fold's highest-scored donors
        count as "the top" for ``model_upgrade_rate_top_n`` and the "top N
        by FY total" baseline. Defaults to roughly 10% of the validation
        fold (at least 1), rather than a fixed count: a fixed ``top_n`` reads
        as a near-perfect rate on a small fold and an uninformative one on a
        large real-world validation year (e.g. reporting 1.0 on a fixed
        top-10 against a 64,178-row fold whose true top-1% lift was 2.24x).
        Clipped to the fold size if larger.
    baseline_giving_threshold : float, optional
        The FY T giving level the "gave >= X last FY" naive baseline uses.
        Defaults to ``threshold / 2``.
    random_state : int, optional
        Seed forwarded to the classifier fits and to the permutation
        importance call, for reproducible scores and reasons.

    Returns
    -------
    scores : pandas.DataFrame
        One row per currently band-qualifying donor, indexed by ``donor_id``,
        sorted by ``affinity_score`` descending. Columns: ``fiscal_year`` (the
        current, possibly still-open FY, reported for reference only --
        it is not a model feature, see Notes); ``affinity_score`` (0-100, see
        :meth:`~philanthropy.models.MajorGiftClassifier.predict_affinity_score`);
        ``rank`` (1 = highest score); ``decile`` (1 = top 10% by rank, 10 =
        bottom); ``top_reasons`` (a tuple of up to 3 ``(feature_name,
        donor_value)`` pairs, see Notes); ``suggested_ask`` (``NaN`` -- see
        Notes). Empty (but correctly typed) if no donor currently qualifies
        for ``band``.
    report : dict
        ``n_training_rows``, ``n_training_fiscal_years``, ``n_scored``;
        ``low_data_warning`` / ``low_data_message`` (flagged under roughly
        ``500`` training rows); ``activity_id_match_warnings`` (list of str,
        captured from ``activities_to_features``'s own low-match-rate
        warning, not recomputed here); ``validated`` (whether a walk-forward
        held-out fold existed at all), and, when it did:

        - ``validation_fiscal_year``, ``n_validation_rows``: which FY the
          held-out fold is, and how many rows it has.
        - ``top_n``: how many of the fold's highest-scored donors count as
          "the top", see the ``top_n`` parameter.
        - ``model_upgrade_rate_top_n``: the actual upgrade rate among the
          model's own top ``top_n`` donors by predicted score.
        - ``baseline_topn_fy_total_upgrade_rate`` / ``lift_topn_fy_total``:
          the naive "top N by FY total" rule's upgrade rate over the same
          ``top_n``, and the model rate's ratio to it (``None`` if the
          baseline rate is 0).
        - ``baseline_giving_threshold``, ``baseline_gave_threshold_upgrade_rate``
          / ``lift_over_gave_threshold``: the naive "gave >= X last FY" rule
          (``X`` is ``baseline_giving_threshold``), its upgrade rate among
          donors who cleared it, and the model rate's ratio to it (``None``
          if nobody in the fold cleared it, or the resulting rate is 0).
        - ``overall_upgrade_rate``: the fold's overall positive rate.
        - ``deciles``: a list of 10 dicts, one per predicted-score decile (1
          = highest-scored 10% of the fold, 10 = lowest), each with
          ``decile``, ``n`` (rows in that decile), ``actual_rate`` (observed
          upgrade rate) and ``mean_predicted`` (mean predicted probability).
          A finer-grained, fixed-``top_n``-independent view of the same
          ranking; see the ``top_n`` parameter for why a single fixed count
          is misleading on its own.
        - ``roc_auc``, ``average_precision``: from :mod:`sklearn.metrics`
          on the held-out fold; ``None`` if the fold has only one class.

    Raises
    ------
    ValueError
        If ``band[0] > band[1]``; if there is not one historical
        ``(donor, fiscal year)`` row to train on as of ``as_of``; if there
        are fewer than 5 historical rows (too few for
        :class:`~philanthropy.models.MajorGiftClassifier`'s internal 5-fold
        calibration to split at all); or if a walk-forward training fold, or
        the full historical training set when a current donor still needs
        scoring, has only one target class (nothing to learn or nothing to
        score: no donor in it ever upgraded, or every one did).
    KeyError
        If ``gifts`` is missing ``donor_id``, ``gift_date`` or
        ``gift_amount``.

    Notes
    -----
    **Feature columns.** Only the numeric columns of the snapshot (the
    gift-derived features, any numeric ``activities_to_features`` columns,
    and any numeric ``donors`` columns) become model features; a non-numeric
    ``donors`` column (e.g. a wealth-rating letter grade) is still joined and
    returned for reference but dropped before fitting, the simplest rule
    that needs no per-column encoding policy for a feature nobody asked this
    function to build. ``fiscal_year`` is excluded from the model features
    even though it is numeric: it is not a donor-specific signal, and the
    current row's fiscal year always sits outside the range the model was
    trained on (every historical row is a strictly earlier FY), so it can
    only ever generalise as noise or, worse, an ordering artefact. It is
    still returned as an output column and left out of ``top_reasons`` for
    the same reason. A current-row feature column absent from history (or
    vice versa), e.g. an activity type that only shows up in one window, is
    reindexed to 0.0 rather than dropped, matching
    ``activities_to_features``'s own "no rows of that type -> 0" rule.

    **Top reasons.** There is no per-instance explainer in this package
    (SHAP is out of scope for this whole series); this reuses the existing
    GLOBAL permutation importances (:func:`~philanthropy.inspection.donor_feature_importance`,
    computed once on the held-out fold, or in-sample with a warning if there
    was no fold to hold out) as fixed feature weights, then for each scored
    donor ranks their OWN feature columns by ``|z-score within the current
    scored population| * global_importance_weight`` (importance clipped at 0,
    since a negative permutation importance means shuffling that feature
    *helped*, not a reason to report), and keeps the top 3 as
    ``(feature_name, donor_value)`` pairs. This is a heuristic, not a causal
    attribution: it says which features are both unusual for this donor and
    generally predictive, not "if X were different the score would change by
    Y". It is deliberately simple rather than a general explainability
    framework.

    **Suggested ask.** ``AskAmountRecommender`` needs its own ask-amount
    label (what a gift officer actually asked for, or a realistic proxy for
    it); nothing in ``build_upgrade_snapshots``'s output is that. Training it
    on, say, the FY T+1 amount actually given would train on the very
    quantity the upgrade label is derived from, an honesty problem, not a
    convenience one. ``suggested_ask`` is left ``NaN``, out of scope for this
    function until a real ask-amount training signal exists.

    Examples
    --------
    Four donor archetypes across five fiscal years: a flat high-band donor,
    an early riser that crosses the threshold in FY2022, a later riser that
    crosses in FY2024, and a flat low-band donor. Only the two that never
    cross ``threshold`` are still band-qualifying "today" (FY2025):

    >>> import pandas as pd
    >>> from philanthropy.models import score_upgrade_prospects
    >>> years = ["2020-08-01", "2021-08-01", "2022-08-01", "2023-08-01", "2024-08-01"]
    >>> archetypes = {
    ...     "flat_high": [900, 900, 900, 900, 900],
    ...     "early_riser": [600, 1050, 1200, 1400, 1600],
    ...     "late_riser": [300, 500, 700, 1100, 1300],
    ...     "flat_low": [400, 400, 400, 400, 400],
    ... }
    >>> rows = [
    ...     {"donor_id": f"{name}_{i}", "gift_date": year, "gift_amount": amount}
    ...     for name, amounts in archetypes.items()
    ...     for i in range(20)
    ...     for year, amount in zip(years, amounts)
    ... ]
    >>> gifts = pd.DataFrame(rows)
    >>> scores, report = score_upgrade_prospects(gifts, random_state=0)
    >>> list(scores.columns)
    ['fiscal_year', 'affinity_score', 'rank', 'decile', 'top_reasons', 'suggested_ask']
    >>> len(scores)
    40
    >>> bool((scores["affinity_score"] >= 0).all() and (scores["affinity_score"] <= 100).all())
    True
    >>> report["n_training_rows"]
    200
    >>> report["validated"]
    True
    """
    low, high = band
    if low > high:
        raise ValueError(f"band[0] ({low}) must be <= band[1] ({high}).")

    df_all, _, _, _ = _prepare_gifts(gifts, fiscal_year_start)
    if df_all.empty:
        raise ValueError(
            "No usable gift rows (need donor_id, gift_date and gift_amount) "
            "to train or score an upgrade model."
        )
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else df_all["_date"].max()

    # Nothing after as_of is ever read: re-derive the pivots from gifts cut at
    # as_of, so the historical training years and the current row share one
    # cutoff, and appending a future-dated row can never move either.
    cut = df_all[df_all["_date"] <= as_of_ts]
    if cut.empty:
        raise ValueError(f"No gift rows on or before as_of={as_of_ts.date()}.")
    df, pivot_sum, pivot_max, pivot_count = _prepare_gifts(cut, fiscal_year_start)

    current_fy = (
        as_of_ts.year + 1 if as_of_ts.month >= fiscal_year_start else as_of_ts.year
    )
    min_fy, max_fy = int(df["_fy"].min()), int(df["_fy"].max())
    historical_years = [
        t for t in range(min_fy, max_fy + 1)
        if _fy_end(t + 1, fiscal_year_start) <= as_of_ts
    ]

    totals_current = _column(pivot_sum, current_fy)
    current_candidates = totals_current[
        (totals_current >= low) & (totals_current <= high) & (totals_current < threshold)
    ]

    donors_norm = None
    if donors is not None:
        donors_norm = donors.copy()
        donors_norm.index = donors_norm.index.astype("string").str.strip()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        historical_snap = build_upgrade_snapshots(
            df, fiscal_years=historical_years, threshold=threshold, band=band,
            fiscal_year_start=fiscal_year_start, activities=activities, donors=donors,
        )
        current_snap = None
        if not current_candidates.empty:
            current_snap = _snapshot_features_for_year(
                df, pivot_sum, pivot_max, pivot_count, current_candidates.index,
                current_fy, fiscal_year_start, activities, donors, donors_norm,
                as_of=as_of_ts,
            )
    activity_id_match_warnings = [str(w.message) for w in caught]

    if historical_snap.empty:
        raise ValueError(
            f"No historical (donor, fiscal year) rows to train on as of "
            f"{as_of_ts.date()}: need at least one fiscal year T with both T "
            "and T+1 fully resolved by then."
        )

    feature_cols = [
        c for c in historical_snap.columns
        if c not in ("target", "fiscal_year")
        and pd.api.types.is_numeric_dtype(historical_snap[c])
    ]
    X = historical_snap[feature_cols].to_numpy(dtype="float64")
    y = historical_snap["target"].to_numpy()
    fys = historical_snap["fiscal_year"].to_numpy()
    n_unique_fys = int(np.unique(fys).size)

    low_data_warning = len(historical_snap) < _LOW_DATA_ROWS
    low_data_message = None
    if low_data_warning:
        low_data_message = (
            f"Only {len(historical_snap)} historical donor-year rows "
            f"(< {_LOW_DATA_ROWS}); scores and the validation report below "
            "are low-confidence."
        )
        warnings.warn(low_data_message, UserWarning, stacklevel=2)

    # Too few historical rows crashes deep inside CalibratedClassifierCV
    # (every downstream fit needs at least this many rows to split), with a
    # confusing error if left unchecked.
    _check_min_rows(len(historical_snap), as_of_ts)

    gave_threshold = (
        float(baseline_giving_threshold)
        if baseline_giving_threshold is not None else threshold / 2.0
    )

    report: Dict[str, Any] = {
        "n_training_rows": int(len(historical_snap)),
        "n_training_fiscal_years": n_unique_fys,
        "low_data_warning": bool(low_data_warning),
        "low_data_message": low_data_message,
        "activity_id_match_warnings": activity_id_match_warnings,
        "current_fiscal_year": int(current_fy),
        "n_scored": 0,
        "validated": False,
        "validation_fiscal_year": None,
        "n_validation_rows": 0,
        "top_n": None,
        "model_upgrade_rate_top_n": None,
        "baseline_topn_fy_total_upgrade_rate": None,
        "lift_topn_fy_total": None,
        "baseline_giving_threshold": gave_threshold,
        "baseline_gave_threshold_upgrade_rate": None,
        "lift_over_gave_threshold": None,
        "overall_upgrade_rate": None,
        "deciles": None,
        "roc_auc": None,
        "average_precision": None,
    }

    if n_unique_fys >= 2:
        n_splits = min(5, n_unique_fys - 1)
        splitter = FiscalYearGroupedSplitter(n_splits=n_splits, drop_repeat_donors=False)
        train_idx, test_idx = list(splitter.split(X, groups=fys))[-1]

        eval_model = MajorGiftClassifier(random_state=random_state).fit(
            X[train_idx], y[train_idx]
        )
        # A single-class training fold fits fine but its predict_proba comes
        # back with one column, not two: guard the call site rather than let
        # the [:, 1] below raise a confusing IndexError.
        _check_two_classes(eval_model.classes_, as_of_ts, "The walk-forward training fold")
        y_test = y[test_idx]
        proba_test = eval_model.predict_proba(X[test_idx])[:, 1]
        fy_total_test = historical_snap["fy_total"].to_numpy()[test_idx]

        n_val = len(test_idx)
        resolved_top_n = top_n if top_n is not None else max(1, round(0.1 * n_val))
        top_n_eff = min(resolved_top_n, n_val)
        model_top_n = np.argsort(-proba_test)[:top_n_eff]
        baseline_top_n = np.argsort(-fy_total_test)[:top_n_eff]
        baseline_rate = float(y_test[baseline_top_n].mean())
        model_rate = float(y_test[model_top_n].mean())

        gave_threshold_mask = fy_total_test >= gave_threshold
        n_gave_threshold = int(gave_threshold_mask.sum())
        gave_threshold_rate = (
            float(y_test[gave_threshold_mask].mean()) if n_gave_threshold > 0 else None
        )

        multiclass_fold = np.unique(y_test).size >= 2
        roc_auc = float(roc_auc_score(y_test, proba_test)) if multiclass_fold else None
        average_precision = (
            float(average_precision_score(y_test, proba_test)) if multiclass_fold else None
        )

        report.update({
            "validated": True,
            "validation_fiscal_year": int(fys[test_idx][0]),
            "n_validation_rows": int(n_val),
            "top_n": int(top_n_eff),
            "model_upgrade_rate_top_n": model_rate,
            "baseline_topn_fy_total_upgrade_rate": baseline_rate,
            "lift_topn_fy_total": (
                float(model_rate / baseline_rate) if baseline_rate > 0 else None
            ),
            "baseline_gave_threshold_upgrade_rate": gave_threshold_rate,
            "lift_over_gave_threshold": (
                float(model_rate / gave_threshold_rate)
                if gave_threshold_rate else None
            ),
            "overall_upgrade_rate": float(y_test.mean()),
            "deciles": _decile_report(proba_test, y_test),
            "roc_auc": roc_auc,
            "average_precision": average_precision,
        })
        importance_df = donor_feature_importance(
            eval_model, X[test_idx], y[test_idx], feature_names=feature_cols,
            random_state=random_state,
        )
    else:
        warnings.warn(
            "Fewer than 2 distinct historical fiscal years, so no "
            "walk-forward validation fold could be built. Scores are still "
            "produced (fit on all historical rows), but there is no "
            "held-out report, and the feature-importance weights behind "
            "top_reasons are computed in-sample.",
            UserWarning, stacklevel=2,
        )
        in_sample_model = MajorGiftClassifier(random_state=random_state).fit(X, y)
        importance_df = donor_feature_importance(
            in_sample_model, X, y, feature_names=feature_cols,
            random_state=random_state,
        )

    model = MajorGiftClassifier(random_state=random_state).fit(X, y)

    if current_snap is None or current_snap.empty:
        scores = _empty_scores_frame()
    else:
        # Same single-class guard as the walk-forward fold above: only
        # needed here (not unconditionally), since with no current row to
        # score there is nothing that would call predict_proba at all.
        _check_two_classes(model.classes_, as_of_ts, "The historical training set")
        X_current = current_snap.reindex(columns=feature_cols, fill_value=0.0)
        affinity = model.predict_affinity_score(X_current.to_numpy(dtype="float64"))
        reasons = _top_reasons(X_current, importance_df)

        scores = pd.DataFrame(
            {
                "fiscal_year": current_snap["fiscal_year"].to_numpy(),
                "affinity_score": affinity,
                "top_reasons": reasons,
                "suggested_ask": np.nan,
            },
            index=current_snap.index,
        )
        scores = scores.sort_values("affinity_score", ascending=False, kind="stable")
        scores["rank"] = np.arange(1, len(scores) + 1)
        scores["decile"] = np.ceil(scores["rank"] / len(scores) * 10).clip(upper=10).astype("int64")
        scores = scores[
            ["fiscal_year", "affinity_score", "rank", "decile", "top_reasons", "suggested_ask"]
        ]

    report["n_scored"] = int(len(scores))
    return scores, report


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _check_min_rows(n_rows: int, as_of_ts: pd.Timestamp) -> None:
    """Raise a clear ``ValueError`` instead of letting a too-small training
    set crash inside ``MajorGiftClassifier``'s internal
    ``CalibratedClassifierCV(cv=5)``. See ``score_upgrade_prospects``'s
    ``Raises`` section."""
    if n_rows < _MIN_TRAINING_ROWS:
        raise ValueError(
            f"Only {n_rows} historical donor-year row(s) as of "
            f"{as_of_ts.date()} (need at least {_MIN_TRAINING_ROWS}): "
            "MajorGiftClassifier calibrates its probabilities with an "
            "internal 5-fold cross-validation, which needs at least that "
            "many rows to split at all. Gather more historical data, or "
            "widen `band`/lower `threshold` to enlarge the candidate "
            "population."
        )


def _check_two_classes(classes: np.ndarray, as_of_ts: pd.Timestamp, scope: str) -> None:
    """Raise a clear ``ValueError`` instead of letting a single-class fit
    crash a caller's later ``predict_proba(...)[:, 1]`` (a single-class fit
    succeeds, but its ``predict_proba`` only has one column). See
    ``score_upgrade_prospects``'s ``Raises`` section."""
    if classes.size < 2:
        outcome = "an upgrade (target=1)" if classes[0] == 1 else "not an upgrade (target=0)"
        raise ValueError(
            f"{scope} has only one class as of {as_of_ts.date()}: every row "
            f"is {outcome}. There is nothing to learn from a single-class "
            "history; this can happen with too few fiscal years of data, "
            "or a `threshold`/`band` that no historical donor ever crossed "
            "(or that every one did)."
        )


def _decile_report(proba: np.ndarray, y_true: np.ndarray) -> list:
    """Ten dicts, one per predicted-score decile of a held-out fold (1 =
    highest-scored 10%, 10 = lowest), each with ``decile``, ``n``,
    ``actual_rate`` and ``mean_predicted``. See ``score_upgrade_prospects``'s
    ``Returns`` section."""
    n = len(proba)
    ranks = np.empty(n, dtype="int64")
    ranks[np.argsort(-proba, kind="stable")] = np.arange(1, n + 1)
    deciles = np.ceil(ranks / n * 10).clip(max=10).astype("int64")

    rows = []
    for d in range(1, 11):
        mask = deciles == d
        n_d = int(mask.sum())
        rows.append({
            "decile": d,
            "n": n_d,
            "actual_rate": float(y_true[mask].mean()) if n_d > 0 else None,
            "mean_predicted": float(proba[mask].mean()) if n_d > 0 else None,
        })
    return rows


def _top_reasons(
    X: pd.DataFrame, importance_df: pd.DataFrame, top_k: int = _TOP_REASONS
) -> list:
    """Per-donor top-``top_k`` ``(feature, value)`` reasons; see the
    "Top reasons" note on :func:`score_upgrade_prospects`."""
    weights = (
        importance_df.set_index("feature")["importance_mean"]
        .reindex(X.columns)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    z = (X - X.mean()).div(X.std(ddof=0).replace(0.0, np.nan)).fillna(0.0)
    weighted = z.abs().mul(weights, axis=1)

    reasons = []
    for donor_id in X.index:
        top_feats = weighted.loc[donor_id].sort_values(ascending=False, kind="stable").index[:top_k]
        reasons.append(tuple((feat, X.loc[donor_id, feat]) for feat in top_feats))
    return reasons


def _empty_scores_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fiscal_year": pd.Series(dtype="int64"),
            "affinity_score": pd.Series(dtype="float64"),
            "rank": pd.Series(dtype="int64"),
            "decile": pd.Series(dtype="int64"),
            "top_reasons": pd.Series(dtype="object"),
            "suggested_ask": pd.Series(dtype="float64"),
        },
        index=pd.Index([], name="donor_id", dtype="object"),
    )
