"""
philanthropy.models._propensity_baseline
================================
"""

from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils import Tags
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

_Self = TypeVar("_Self", bound="PropensityScorer")


class PropensityScorer(ClassifierMixin, BaseEstimator):
    """Constant-probability baseline that predicts P=0.5 for every donor.

    A deliberately trivial, sklearn-compliant reference point: it fits nothing
    and returns 0.5 for all rows, which makes it equivalent in effect to
    :class:`sklearn.dummy.DummyClassifier` with ``strategy="uniform"``. It
    exists so a domain benchmark has a named floor to beat, not because it
    scores anything. For real
    propensity scoring reach for
    :class:`~philanthropy.models.DonorPropensityModel` or
    :class:`~philanthropy.models.MajorGiftClassifier`.

    Parameters
    ----------
    threshold : float, default=0.5
        Decision threshold on ``predict_proba(X)[:, 1]``.  The comparison is
        **strict** (``proba > threshold``), so at the default the constant 0.5
        score falls *below* the threshold and :meth:`predict` returns
        ``classes_[0]`` for every row.  scikit-learn requires
        ``argmax(predict_proba) == predict``, and ``argmax`` of a tied
        ``[0.5, 0.5]`` row is index 0, so a non-strict comparison here would
        make the estimator self-inconsistent.

    Raises
    ------
    ValueError
        In :meth:`fit`, if ``y`` has more than two classes.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def fit(self: _Self, X: Any, y: Any) -> _Self:
        """Validate input and record the target classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Binary target labels.

        Returns
        -------
        self : PropensityScorer
            Fitted estimator. Sets ``classes_``.

        Raises
        ------
        ValueError
            If ``y`` is not binary.
        """
        X, y = validate_data(self, X, y, reset=True)
        check_classification_targets(y)
        y_type = type_of_target(y, input_name="y", raise_unknown=True)
        if y_type not in ("binary",):
            raise ValueError(
                "Only binary classification is supported. The type of the "
                f"target is {y_type}."
            )
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict binary labels using the constant probability baseline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels. With the default threshold, the constant 0.5
            probability is not above the threshold and ``classes_[0]`` is
            returned for every row.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        if len(self.classes_) == 1:
            return np.full(X.shape[0], self.classes_[0])
        proba = self.predict_proba(X)[:, 1]
        idx = (proba > self.threshold).astype(int)
        return self.classes_[idx]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return the constant probability baseline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            ``[0.5, 0.5]`` for every row, or shape ``(n_samples, 1)`` when
            only one class was seen during fitting.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        n = X.shape[0]
        if len(self.classes_) == 1:
            return np.ones((n, 1))
        prob_pos = np.full(n, 0.5)
        return np.column_stack([1 - prob_pos, prob_pos])

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        tags.classifier_tags.multi_class = False
        return tags
