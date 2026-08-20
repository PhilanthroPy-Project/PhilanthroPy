"""
philanthropy.experimental._uplift
=================================
Two-model (T-learner) uplift estimator for fundraising appeals.

Experimental: not yet check_estimator compliant. ``fit`` takes an extra
``treatment`` argument, which breaks the standard ``fit(X, y)`` signature —
this is why the estimator lives in the experimental package.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted, validate_data


class UpliftTLearner(ClassifierMixin, BaseEstimator):
    """Estimate how much a solicitation lifts a donor's probability of giving.

    Implements the classic **T-learner** (two-model) approach to treatment-
    effect estimation.  Two :class:`~sklearn.ensemble.RandomForestClassifier`
    models are fit independently: one on the *treated* arm (donors who
    received the appeal) and one on the *control* arm (donors who did not).
    For a new donor, the uplift is the difference in predicted giving
    probability between the two arms:

    .. math::

        \\text{uplift}(x) = \\hat{P}(\\text{give} \\mid x, \\text{treated})
                          - \\hat{P}(\\text{give} \\mid x, \\text{control})

    A positive uplift means the appeal is expected to *increase* the donor's
    probability of giving, so the donor is worth soliciting; a negative uplift
    flags "sleeping dogs" whom the appeal may annoy.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in each arm's :class:`RandomForestClassifier`.
    max_depth : int or None, default=None
        Maximum depth of each tree.  ``None`` grows trees until leaves are
        pure; set to 5–10 to regularise on small appeal panels.
    random_state : int or None, default=None
        Seed shared by both arms for reproducible uplift scores.

    Attributes
    ----------
    model_treated_ : RandomForestClassifier
        Model fitted on rows where ``treatment == 1``.
    model_control_ : RandomForestClassifier
        Model fitted on rows where ``treatment == 0``.
    classes_ : ndarray of shape (n_classes,)
        Unique giving labels seen during :meth:`fit`, typically
        ``array([0, 1])``.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> X = rng.normal(size=(n, 1))
    >>> treatment = rng.integers(0, 2, size=n)
    >>> # Treated donors with a positive feature give far more often.
    >>> p = np.where((treatment == 1) & (X[:, 0] > 0), 0.8, 0.2)
    >>> y = (rng.random(n) < p).astype(int)
    >>> model = UpliftTLearner(random_state=0).fit(X, y, treatment)
    >>> uplift = model.predict_uplift_score(X)
    >>> uplift.shape
    (400,)
    >>> bool(uplift[X[:, 0] > 0].mean() > 0)
    True
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y, treatment) -> "UpliftTLearner":
        """Fit the two arm-specific models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Donor feature matrix.
        y : array-like of shape (n_samples,)
            Binary giving outcome: ``1`` = gave, ``0`` = did not give.
        treatment : array-like of shape (n_samples,)
            Binary treatment indicator: ``1`` = received the appeal,
            ``0`` = control.

        Returns
        -------
        self : UpliftTLearner
            Fitted estimator.

        Raises
        ------
        ValueError
            If ``treatment`` is not binary ``{0, 1}``, if its length does not
            match ``n_samples``, or if either arm (treated / control) is empty.
        """
        X, y = validate_data(self, X, y, ensure_all_finite="allow-nan", reset=True)
        treatment = np.asarray(treatment)

        if treatment.shape[0] != X.shape[0]:
            raise ValueError(
                f"treatment has length {treatment.shape[0]}, "
                f"expected {X.shape[0]} to match the number of samples."
            )
        if not set(np.unique(treatment)).issubset({0, 1}):
            raise ValueError("treatment must be binary with values in {0, 1}.")

        is_treated = treatment == 1
        is_control = treatment == 0
        if not is_treated.any() or not is_control.any():
            raise ValueError(
                "Both arms must be present: need at least one treated "
                "(treatment==1) and one control (treatment==0) sample."
            )

        self.classes_ = np.unique(y)

        self.model_treated_ = self._fit_arm(X[is_treated], y[is_treated])
        self.model_control_ = self._fit_arm(X[is_control], y[is_control])
        return self

    def _fit_arm(self, X_arm, y_arm) -> RandomForestClassifier:
        """Fit one arm's RandomForestClassifier."""
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        model.fit(X_arm, y_arm)
        return model

    @staticmethod
    def _prob_give(model: RandomForestClassifier, X) -> np.ndarray:
        """Return P(class == 1) from a fitted arm, guarding single-class arms.

        If the arm saw only one class during fit, ``predict_proba`` returns a
        single column; P(give) is then 1.0 iff that sole class is the positive
        class, else 0.0.  Mirrors the guard in
        ``philanthropy.models._lapse.predict_lapse_score``.
        """
        proba = model.predict_proba(X)
        if proba.shape[1] < 2:
            return np.full(proba.shape[0], 1.0 if 1 in model.classes_ else 0.0)
        return proba[:, 1]

    def predict_uplift_score(self, X) -> np.ndarray:
        """Return the estimated uplift for each donor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Donor feature matrix.

        Returns
        -------
        uplift : ndarray of shape (n_samples,)
            ``P(give | treated) - P(give | control)`` per donor, in the closed
            interval ``[-1.0, 1.0]``.  Positive values indicate the appeal is
            expected to lift giving probability.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
        p_treated = self._prob_give(self.model_treated_, X)
        p_control = self._prob_give(self.model_control_, X)
        return p_treated - p_control

    def predict(self, X) -> np.ndarray:
        """Return 1 where soliciting is expected to help, else 0.

        Convenience wrapper: ``(predict_uplift_score(X) > 0).astype(int)``.
        A ``1`` marks a donor worth soliciting (positive expected uplift).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Donor feature matrix.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Binary predictions (1 or 0).
        """
        return (self.predict_uplift_score(X) > 0).astype(int)
