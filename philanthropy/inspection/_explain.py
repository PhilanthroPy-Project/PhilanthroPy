"""
philanthropy.inspection._explain
================================
Per-donor score explanations by median substitution.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def explain_donor_scores(
    estimator: Any,
    X: Any,
    method: str = "predict_proba",
    *,
    baseline: Any = None,
    top_k: Optional[int] = 3,
    class_index: Optional[int] = None,
) -> pd.DataFrame:
    """Explain why each donor got their score, one row of ``X`` at a time.

    :func:`donor_feature_importance` says which signals matter to the model
    overall; this says which signals moved *this* donor's score. For every row
    and every feature, the feature is replaced with a typical (baseline) value,
    the row is rescored, and the change is reported:

    ``contribution = score(row) - score(row with feature set to baseline)``

    A positive contribution means the donor's actual value pushed the score up
    relative to a typical donor; a negative one pulled it down. This is a
    one-at-a-time, dependency-free stand-in for SHAP: contributions are exact
    for additive models but do not sum to the score in general, and correlated
    features can share or mask credit. A value already close to the baseline
    gets a contribution near zero, however important the feature.

    Parameters
    ----------
    estimator : fitted estimator
        Any fitted estimator (or :class:`~sklearn.pipeline.Pipeline`) exposing
        ``method``.
    X : array-like or pandas.DataFrame of shape (n_samples, n_features)
        Donors to explain. May contain ``NaN`` if the estimator accepts it; a
        missing value is explained like any other value, so its contribution is
        the effect of being missing rather than typical. DataFrames may carry
        categorical / object columns (for a pipeline that encodes them).
    method : str, default="predict_proba"
        Name of the scoring method, e.g. ``"predict_proba"``,
        ``"predict_affinity_score"``, ``"predict_lapse_score"``,
        ``"decision_function"`` or ``"predict"`` (regressors). Must return one
        value per row, or a 2-D array of per-class scores.
    baseline : array-like, optional
        The typical values substituted in. Either the **training** feature
        matrix (2-D), from which the per-column median is taken (the most
        frequent value for non-numeric columns, ``NaN`` ignored), or a 1-D
        sequence of reference values in column order. When omitted, the
        estimator's frozen ``fill_values_`` (training medians, e.g.
        :class:`~philanthropy.models.FinancialForecastModel`) are used.
        Use training data, not ``X``, so a donor is compared with the
        population the model learned from.
    top_k : int or None, default=3
        Number of features reported per row, largest absolute contribution
        first. ``None`` reports all of them.
    class_index : int, optional
        Column to explain when ``method`` returns per-class scores. Defaults to
        the positive class (column 1) for binary output and, for multiclass
        output, each row's predicted (highest-scoring) class.

    Returns
    -------
    pandas.DataFrame
        Long format, ``top_k`` rows per donor, with columns ``row`` (the
        index label of ``X`` for a DataFrame, else the position), ``rank``
        (1 = most influential), ``feature``, ``value`` (the donor's actual
        value), ``baseline`` and ``contribution``.

    Raises
    ------
    ValueError
        If no baseline is given and the estimator stores none, or if the
        baseline's length does not match the number of columns in ``X``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from philanthropy.inspection import explain_donor_scores
    >>> X = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    >>> model = LinearRegression().fit(X, 2 * X[:, 0])
    >>> out = explain_donor_scores(model, X[[3]], "predict", baseline=X, top_k=1)
    >>> out[["feature", "contribution"]].round(2).to_dict("records")
    [{'feature': 0, 'contribution': 3.0}]
    """
    is_df = hasattr(X, "columns")
    if not is_df:
        X = np.asarray(X)
        if X.dtype.kind in "biu":  # so a fractional median is not truncated
            X = X.astype(float)
    n_samples, n_features = X.shape
    names = list(X.columns) if is_df else list(range(n_features))

    if baseline is None:
        baseline = getattr(estimator, "fill_values_", None)
        if baseline is None:
            raise ValueError(
                "explain_donor_scores needs a baseline: pass baseline=X_train "
                "(the training features) or a 1-D sequence of reference values."
            )
    if np.ndim(baseline) == 2:
        train = pd.DataFrame(baseline)
        ref = [
            train[c].median()
            if pd.api.types.is_numeric_dtype(train[c])
            else train[c].mode().iloc[0]
            for c in train.columns
        ]
    else:
        ref = list(baseline)
    if len(ref) != n_features:
        raise ValueError(
            f"baseline has {len(ref)} columns but X has {n_features} columns."
        )

    raw = np.asarray(getattr(estimator, method)(X), dtype=float)
    if raw.ndim == 2:
        if class_index is not None:
            col: Any = class_index
        elif raw.shape[1] == 2:
            col = 1
        else:
            col = raw.argmax(axis=1)
    rows = np.arange(n_samples)

    def _score(Z: Any) -> np.ndarray:
        s = np.asarray(getattr(estimator, method)(Z), dtype=float)
        return s[rows, col] if s.ndim == 2 else s

    base = raw[rows, col] if raw.ndim == 2 else raw
    contrib = np.empty((n_samples, n_features))
    for j in range(n_features):
        Z = X.copy()
        if is_df:
            Z[names[j]] = ref[j]
        else:
            Z[:, j] = ref[j]
        contrib[:, j] = base - _score(Z)

    k = n_features if top_k is None else min(top_k, n_features)
    order = np.argsort(-np.abs(contrib), axis=1, kind="stable")[:, :k]
    values = X.to_numpy(dtype=object) if is_df else X
    labels = X.index if is_df else rows
    return pd.DataFrame(
        {
            "row": np.repeat(np.asarray(labels), k),
            "rank": np.tile(np.arange(1, k + 1), n_samples),
            "feature": [names[j] for j in order.ravel()],
            "value": values[rows[:, None], order].ravel(),
            "baseline": [ref[j] for j in order.ravel()],
            "contribution": contrib[rows[:, None], order].ravel(),
        }
    )
