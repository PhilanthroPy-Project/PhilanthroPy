"""
philanthropy.models._planned_giving
===================================
Models for predicting planned giving (bequest) intent.
"""

from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils import Tags
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)

from philanthropy.utils._validation import resolve_columns

_Self = TypeVar("_Self", bound="PlannedGivingIntentScorer")

_CALIBRATION_CV_FOLDS = 2


class PlannedGivingIntentScorer(ClassifierMixin, BaseEstimator):
    """
    Predicts bequest/planned giving intent. Wraps
    HistGradientBoostingClassifier with CalibratedClassifierCV.

    Exposes `.predict_intent_score(X)` returning a 0-100 float array.
    Missing values are accepted and handled natively by the boosting
    backend, so a file with blank age or wealth-screening columns needs no
    imputation first.

    Calibration uses ``cv=2``, so every class in ``y`` must have at least 2
    examples; ``fit`` raises a ``ValueError`` up front if that is not the
    case rather than surfacing scikit-learn's cross-validation error.

    Parameters
    ----------
    n_estimators : int, default=100
        The maximum number of boosting iterations, passed as ``max_iter`` to
        the underlying :class:`HistGradientBoostingClassifier`.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.
    class_weight : dict, "balanced" or None, default=None
        Weight scheme for the classes, useful when bequest intent is rare.
        Applied as ``sample_weight`` (multiplied with any ``sample_weight``
        passed to :meth:`fit`) rather than as the backend's own
        ``class_weight``, for the same reason as
        :class:`~philanthropy.models.MajorGiftClassifier`: the calibration
        folds would otherwise wash the reweighting out. With it set, scores
        reflect the reweighted class balance, so read them as a ranking.
    categorical_features : list of int or str, bool mask, or None, default=None
        Columns the backend should split on as unordered categories (a
        region, constituency or appeal code) instead of as numbers, so they
        need no one-hot step. Positions, a boolean mask, or column names
        when ``X`` is a DataFrame. The columns must hold non-negative
        integer codes below 255, as for ``HistGradientBoostingClassifier``.
        ``None`` treats every column as numeric.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int | None = None,
        class_weight: Any = None,
        categorical_features: Any = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.class_weight = class_weight
        self.categorical_features = categorical_features

    def fit(self: _Self, X: Any, y: Any, sample_weight: Any = None) -> _Self:
        """Fit the calibrated classifier to planned-giving intent labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Binary target labels.
        sample_weight : array-like of shape (n_samples,), default=None
            Per-donor weights, used by both the boosting backend and the
            calibration step.

        Returns
        -------
        self : PlannedGivingIntentScorer
            Fitted estimator. Sets ``classes_``, ``n_features_in_``, and
            ``estimator_``.

        Raises
        ------
        ValueError
            If ``y`` is not a classification target, if it contains fewer
            than 2 classes, or if any class has fewer than 2 examples,
            since calibration uses ``cv=2``.
        """
        X, y = validate_data(self, X, y, ensure_all_finite="allow-nan", reset=True)
        # Reject continuous targets before counting classes, so a regression
        # target gets sklearn's standard "continuous" message instead of
        # being misread as one-example-per-class.
        check_classification_targets(y)

        self.classes_, counts = np.unique(y, return_counts=True)
        self.n_features_in_ = X.shape[1]

        if len(self.classes_) < 2:
            raise ValueError(
                "PlannedGivingIntentScorer requires at least 2 classes in "
                f"y, got {len(self.classes_)} class: {list(self.classes_)}."
            )
        if counts.min() < _CALIBRATION_CV_FOLDS:
            raise ValueError(
                "PlannedGivingIntentScorer calibrates with "
                f"cv={_CALIBRATION_CV_FOLDS}, so every class needs at least "
                f"{_CALIBRATION_CV_FOLDS} examples; the smallest class has "
                f"{counts.min()}."
            )

        base_estimator = HistGradientBoostingClassifier(
            max_iter=self.n_estimators,
            # The old GradientBoostingClassifier default depth: same tree
            # shape, so scores match it on sample data; only NaN handling changes.
            max_depth=3,
            categorical_features=resolve_columns(self, self.categorical_features),
            random_state=self.random_state
        )
        self.estimator_ = CalibratedClassifierCV(
            estimator=base_estimator,
            method="sigmoid",
            cv=_CALIBRATION_CV_FOLDS,
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
        if self.class_weight is not None:
            class_w = compute_sample_weight(self.class_weight, y)
            sample_weight = class_w if sample_weight is None else sample_weight * class_w
        self.estimator_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict bequest/planned-giving intent labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
        return self.estimator_.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return calibrated class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Predicted probabilities for each class.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
        return self.estimator_.predict_proba(X)

    def predict_intent_score(self, X: Any) -> np.ndarray:
        """
        Return P(planned giving intent) × 100, rounded to 2 decimal places.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Values in range [0.0, 100.0].
        """
        proba = self.predict_proba(X)
        scores = np.round(proba[:, 1] * 100.0, 2)
        return scores

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
