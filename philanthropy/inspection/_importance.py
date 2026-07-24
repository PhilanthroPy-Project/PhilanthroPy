"""
philanthropy.inspection._importance
===================================
Model-agnostic feature attribution for fitted donor-scoring estimators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def donor_feature_importance(
    estimator,
    X,
    y,
    *,
    feature_names=None,
    n_repeats: int = 10,
    random_state=None,
    scoring=None,
) -> pd.DataFrame:
    """Permutation feature importance for a fitted donor-scoring estimator.

    Answers "which donor signals move this model's score?" for **any** fitted
    PhilanthroPy / scikit-learn estimator — including calibrated or
    gradient-boosted models (e.g. :class:`~philanthropy.models.MajorGiftClassifier`)
    that do not expose ``feature_importances_`` — by measuring how far a scoring
    metric drops when each feature's values are randomly shuffled. It is a
    dependency-free alternative to SHAP for explaining and auditing donor scores.

    Because attribution is computed by scoring against held-out ``y``, run this on
    a validation split (not the training data) for an honest read.

    Parameters
    ----------
    estimator : fitted estimator
        Any fitted estimator implementing ``predict`` (and the metric implied by
        ``scoring``). Must already be fitted.
    X : array-like or pandas.DataFrame of shape (n_samples, n_features)
        Feature matrix to permute.
    y : array-like of shape (n_samples,)
        True targets used to score each permutation.
    feature_names : sequence of str, optional
        Column labels for ``X``. Inferred from a DataFrame's columns when not
        given; otherwise defaults to ``x0 .. x{n-1}``. Must match the number of
        columns in ``X`` when supplied.
    n_repeats : int, default=10
        Number of times each feature is permuted.
    random_state : int, RandomState instance or None, default=None
        Controls the permutation shuffling for reproducibility.
    scoring : str, callable or None, default=None
        Scorer passed through to
        :func:`sklearn.inspection.permutation_importance` (defaults to the
        estimator's own ``score``).

    Returns
    -------
    pandas.DataFrame
        One row per feature with columns ``feature``, ``importance_mean`` and
        ``importance_std``, sorted by ``importance_mean`` descending (most
        influential first). A larger drop = a more influential feature.

    Raises
    ------
    ValueError
        If ``feature_names`` is supplied but its length does not match the number
        of columns in ``X``.
    """
    n_features = np.asarray(X).shape[1]

    if feature_names is not None:
        names = list(feature_names)
    elif hasattr(X, "columns"):
        names = list(X.columns)
    else:
        names = [f"x{i}" for i in range(n_features)]

    if len(names) != n_features:
        raise ValueError(
            f"feature_names has {len(names)} names but X has {n_features} columns."
        )

    result = permutation_importance(
        estimator,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
    )

    return pd.DataFrame(
        {
            "feature": names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)
