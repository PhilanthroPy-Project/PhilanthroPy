from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

MOVES_STAGES = ["IDENTIFY", "QUALIFY", "CULTIVATE", "SOLICIT", "STEWARD"]

_Self = TypeVar("_Self", bound="MovesManagementClassifier")


class MovesManagementClassifier(ClassifierMixin, BaseEstimator):
    """
    Predicts the next best moves management stage for a donor.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 200,
        class_weight: str | dict | None = "balanced",
        random_state: int | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self: _Self, X: Any, y: Any) -> _Self:
        """Fit the classifier to labelled moves-stage data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Moves-stage target labels.

        Returns
        -------
        self : MovesManagementClassifier
            Fitted estimator. Sets ``feature_names_in_`` when ``X`` is a
            DataFrame, ``n_features_in_``, ``label_encoder_``, ``classes_``,
            ``estimator_``, and ``n_iter_``.

        Raises
        ------
        ValueError
            If ``y`` is not a classification target.
        """
        X, y = validate_data(self, X, y, reset=True)
        # Reject continuous targets: this is a classifier, so a regression
        # target must not be silently label-encoded into pseudo-classes.
        check_classification_targets(y)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns.tolist(), dtype=object)
        self.n_features_in_ = X.shape[1]

        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        self.estimator_ = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.estimator_.fit(X, y_encoded)
        # Expose n_iter_ (project convention for any estimator taking max_iter;
        # check_estimator requires it).
        self.n_iter_ = self.estimator_.n_iter_
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict the next moves-management stage for each donor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted stage labels.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        y_pred = self.estimator_.predict(X)
        return self.label_encoder_.inverse_transform(y_pred)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return class probabilities for each moves-management stage.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Predicted probabilities for each stage.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self.estimator_.predict_proba(X)

    def action_priority(self, X: Any) -> dict:
        """Predict the next-best stage per donor plus a portfolio rollup.

        Unlike ``predict``/``predict_proba`` (which return ndarrays), this
        returns a dict with keys ``"stage"`` (ndarray of predicted stage
        labels), ``"confidence"`` (ndarray of max class probabilities), and
        ``"portfolio_summary"`` (dict mapping each stage to its donor count).
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        
        probas = self.estimator_.predict_proba(X)
        pred_idx = np.argmax(probas, axis=1)
        confidences = np.max(probas, axis=1)
        
        stages = self.label_encoder_.inverse_transform(pred_idx)
        
        unique_stages, counts = np.unique(stages, return_counts=True)
        portfolio_summary = dict(zip(unique_stages, counts))
        
        return {
            "stage": stages,
            "confidence": confidences,
            "portfolio_summary": portfolio_summary,
        }
