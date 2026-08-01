"""
philanthropy.models._propensity_baseline
================================
"""

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data


class PropensityScorer(ClassifierMixin, BaseEstimator):
    """Constant-probability baseline that predicts P=0.5 for every donor.

    A deliberately trivial, sklearn-compliant reference point: it fits nothing
    and returns 0.5 for all rows. Use it as a floor to beat when benchmarking.
    For real
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

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def fit(self, X, y):
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

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        if len(self.classes_) == 1:
            return np.full(X.shape[0], self.classes_[0])
        proba = self.predict_proba(X)[:, 1]
        idx = (proba > self.threshold).astype(int)
        return self.classes_[idx]

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        n = X.shape[0]
        if len(self.classes_) == 1:
            return np.ones((n, 1))
        prob_pos = np.full(n, 0.5)
        return np.column_stack([1 - prob_pos, prob_pos])

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        tags.classifier_tags.multi_class = False
        return tags
