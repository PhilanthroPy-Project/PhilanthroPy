import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

MOVES_STAGES = ["IDENTIFY", "QUALIFY", "CULTIVATE", "SOLICIT", "STEWARD"]

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
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y):
        X, y = validate_data(self, X, y, reset=True)
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
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        y_pred = self.estimator_.predict(X)
        return self.label_encoder_.inverse_transform(y_pred)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self.estimator_.predict_proba(X)

    def predict_action_priority(self, X) -> dict:
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
