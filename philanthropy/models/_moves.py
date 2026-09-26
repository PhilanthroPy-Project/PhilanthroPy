from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils import Tags
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

MOVES_STAGES = ["IDENTIFY", "QUALIFY", "CULTIVATE", "SOLICIT", "STEWARD"]

_Self = TypeVar("_Self", bound="MovesManagementClassifier")


class MovesManagementClassifier(ClassifierMixin, BaseEstimator):
    """
    Predicts the next best moves management stage for a donor.

    Wraps a :class:`~sklearn.ensemble.HistGradientBoostingClassifier` and
    labels its predictions with the moves-management stage names the donor
    was trained on, rather than requiring the caller to encode stages
    themselves. ``class_weight="balanced"`` is the default because moves
    stages are typically imbalanced (many donors sit in early stages, few in
    ``"STEWARD"``); pass ``class_weight=None`` to disable the reweighting.

    ``predict_proba`` and :meth:`action_priority` pass NaN straight through
    to the HistGradientBoostingClassifier backend, which handles missing
    values natively, so features do not need to be imputed first.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Learning rate of the underlying HistGradientBoostingClassifier.
    max_iter : int, default=200
        Maximum number of boosting iterations.
    class_weight : str, dict or None, default="balanced"
        Class weights passed to the backend. ``"balanced"`` reweights
        inversely proportional to stage frequency.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Moves-stage labels seen during ``fit``.
    label_encoder_ : LabelEncoder
        Encoder mapping stage labels to the integer classes the backend
        estimator was fit on.
    estimator_ : HistGradientBoostingClassifier
        The fitted backend estimator.
    n_features_in_ : int
        Number of features seen during ``fit``.
    n_iter_ : int
        Number of boosting iterations performed by ``estimator_``.

    Examples
    --------
    >>> import numpy as np
    >>> from philanthropy.models import MovesManagementClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((12, 3))
    >>> y = ["IDENTIFY", "QUALIFY", "CULTIVATE"] * 4
    >>> clf = MovesManagementClassifier(max_iter=10, random_state=0).fit(X, y)
    >>> sorted(clf.classes_.tolist())
    ['CULTIVATE', 'IDENTIFY', 'QUALIFY']

    Notes
    -----
    ``action_priority``'s ``"confidence"`` is the raw max class probability
    from the backend estimator, not a calibrated probability: on held-out
    data, rows with a reported confidence of 0.73-0.99 were observed correct
    only 56-70% of the time. Treat it as a ranking signal for prioritizing
    donors, not as a calibrated likelihood.
    """

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

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
            If ``y`` is not a classification target, or if it contains fewer
            than 2 classes.
        """
        X, y = validate_data(self, X, y, ensure_all_finite="allow-nan", reset=True)
        # Reject continuous targets: this is a classifier, so a regression
        # target must not be silently label-encoded into pseudo-classes.
        check_classification_targets(y)
        self.n_features_in_ = X.shape[1]

        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        if len(self.classes_) < 2:
            raise ValueError(
                "MovesManagementClassifier requires at least 2 classes in "
                f"y, got {len(self.classes_)} class: {list(self.classes_)}."
            )

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
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
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
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
        return self.estimator_.predict_proba(X)

    def action_priority(self, X: Any) -> dict:
        """Predict the next-best stage per donor plus a portfolio rollup.

        Unlike ``predict``/``predict_proba`` (which return ndarrays), this
        returns a dict with keys ``"stage"`` (ndarray of predicted stage
        labels), ``"confidence"`` (ndarray of max class probabilities), and
        ``"portfolio_summary"`` (dict mapping each stage to its donor count).
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=False)

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
