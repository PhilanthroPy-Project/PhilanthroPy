"""
philanthropy.metrics._fairness
==============================
Lightweight fairness diagnostics for scored donor cohorts.

Wealth- and capacity-based features (estimated net worth, real-estate value,
geography) can act as proxies for protected characteristics, so a model that
never sees a protected attribute can still produce disparate outcomes. These
functions are **diagnostics, not guarantees**: they surface disparity so a human
can investigate it — they do not certify a model as fair or legally compliant.
See ``docs/explanation/compliance_considerations.md``.
"""

from __future__ import annotations

from typing import Collection, Dict

import numpy as np


def selection_rate_by_group(
    y_pred: Collection,
    sensitive_features: Collection,
    pos_label=1,
) -> Dict[object, float]:
    """Fraction selected (``y_pred == pos_label``) within each protected group.

    Parameters
    ----------
    y_pred : array-like of shape (n_samples,)
        Binary decisions (e.g. "flagged for major-gift outreach"). Threshold
        continuous scores before calling this.
    sensitive_features : array-like of shape (n_samples,)
        Protected-group label per sample (e.g. race, age band, gender).
    pos_label : default=1
        Value in ``y_pred`` that counts as "selected".

    Returns
    -------
    dict
        Mapping of group value -> selection rate in ``[0.0, 1.0]``.
    """
    y_pred = np.asarray(y_pred)
    groups = np.asarray(sensitive_features)
    if y_pred.shape[0] != groups.shape[0]:
        raise ValueError(
            f"y_pred and sensitive_features must be the same length, got "
            f"{y_pred.shape[0]} and {groups.shape[0]}."
        )
    if y_pred.shape[0] == 0:
        raise ValueError("y_pred is empty; nothing to score.")

    selected = y_pred == pos_label
    return {g: float(selected[groups == g].mean()) for g in np.unique(groups)}


def disparate_impact_ratio(
    y_pred: Collection,
    sensitive_features: Collection,
    pos_label=1,
) -> float:
    """Four-fifths-rule disparate-impact ratio across protected groups.

    Computes ``min(selection_rate) / max(selection_rate)`` over the groups in
    ``sensitive_features``. A value of ``1.0`` is exact parity; the US EEOC
    "four-fifths rule" flags a ratio below ``0.8`` as evidence of adverse impact
    that warrants investigation.

    This is a **diagnostic, not a fairness guarantee or legal clearance** — a
    passing ratio does not certify a model as non-discriminatory, and the choice
    of protected groups and decision threshold materially affects the result.

    Parameters
    ----------
    y_pred : array-like of shape (n_samples,)
        Binary decisions. Threshold continuous scores first.
    sensitive_features : array-like of shape (n_samples,)
        Protected-group label per sample.
    pos_label : default=1
        Value in ``y_pred`` that counts as "selected".

    Returns
    -------
    float
        Ratio in ``[0.0, 1.0]``. Returns ``1.0`` when only one group is present
        or when no sample in any group is selected (no disparity to measure).
    """
    rates = selection_rate_by_group(y_pred, sensitive_features, pos_label=pos_label)
    values = np.array(list(rates.values()), dtype=float)
    max_rate = values.max()
    if max_rate == 0.0:
        return 1.0
    return float(values.min() / max_rate)
