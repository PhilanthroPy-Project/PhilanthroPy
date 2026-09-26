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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import Tags
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

_Self = TypeVar("_Self", bound="PlannedGivingIntentScorer")

_CALIBRATION_CV_FOLDS = 2


class PlannedGivingIntentScorer(ClassifierMixin, BaseEstimator):
    """
    Predicts bequest/planned giving intent. Wraps GradientBoostingClassifier
    with CalibratedClassifierCV.

    Exposes `.predict_intent_score(X)` returning a 0-100 float array. NaN
    features are rejected: GradientBoostingClassifier, the backend this
    class calibrates, does not support missing values, so ``fit``/``predict``
    raise on NaN input rather than passing it through.

    Calibration uses ``cv=2``, so every class in ``y`` must have at least 2
    examples; ``fit`` raises a ``ValueError`` up front if that is not the
    case rather than surfacing scikit-learn's cross-validation error.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of boosting stages to perform.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self: _Self, X: Any, y: Any) -> _Self:
        """Fit the calibrated classifier to planned-giving intent labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Binary target labels.

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
        X, y = validate_data(self, X, y, reset=True)
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

        base_estimator = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self.estimator_ = CalibratedClassifierCV(
            estimator=base_estimator,
            method="sigmoid",
            cv=_CALIBRATION_CV_FOLDS,
        )
        self.estimator_.fit(X, y)
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
        X = validate_data(self, X, reset=False)
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
        X = validate_data(self, X, reset=False)
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
        return tags
