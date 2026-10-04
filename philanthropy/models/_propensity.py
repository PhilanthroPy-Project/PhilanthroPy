"""
philanthropy.models._propensity
================================
Production-grade DonorPropensityModel for hospital major-gift fundraising.

This module provides a fully scikit-learn–compatible estimator that wraps
a tunable RandomForestClassifier and surfaces a human-readable affinity
score (0–100) for use by prospect-management officers and gift officers.
"""

from __future__ import annotations

from typing import Any, Optional, TypeVar, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import Tags
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

_SelfD = TypeVar("_SelfD", bound="DonorPropensityModel")
_SelfM = TypeVar("_SelfM", bound="MajorGiftClassifier")


class DonorPropensityModel(ClassifierMixin, BaseEstimator):
    """Predict whether a hospital prospect is a major-gift donor.

    ``DonorPropensityModel`` wraps a :class:`sklearn.ensemble.RandomForestClassifier`
    and is designed specifically for hospital advancement and major-gift
    fundraising teams.  Given a feature matrix describing donors (e.g. recency,
    frequency, monetary value, event attendance, giving capacity estimates), the
    model outputs:

    * **Binary predictions** (``predict``): 0 for standard donors, 1 for
      major-gift prospects above the team's threshold.
    * **Probability estimates** (``predict_proba``): class probabilities in
      the standard sklearn two-column format. These come straight from the
      underlying ``RandomForestClassifier`` and are not calibrated; see
      :meth:`predict_affinity_score` for details.
    * **Affinity scores** (``predict_affinity_score``): the positive-class
      probability mapped to a 0–100 integer scale, enabling gift officers to
      quickly rank prospects in wealth-screening reports or CRM dashboards
      (e.g. Salesforce NPSP, Raiser's Edge NXT, Veeva CRM).

    The model is pipeline-safe and passes ``sklearn.utils.estimator_checks.
    check_estimator``.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the underlying :class:`RandomForestClassifier`.
        Increase for more stable probability estimates at the cost of
        inference speed.
    max_depth : int or None, default=None
        Maximum depth of each decision tree.  ``None`` allows trees to grow
        until leaves are pure, which may overfit on small prospect pools;
        set to 5–10 for regularisation.
    min_samples_split : int or float, default=2
        Minimum number of samples (or fraction) required to split an internal
        node.  Larger values act as a regulariser, improving generalisation
        on sparse hospital datasets.
    min_samples_leaf : int or float, default=0.008
        Minimum number of samples required to be at a leaf node. An int is
        an absolute count; a float is a fraction of the training rows
        (``ceil(min_samples_leaf * n_samples)``), so the default scales with
        file size: about 2 rows per leaf on 250 donors, 400 on 50,000.
        Changed from 1: with ``max_depth=None``, single-sample leaves let
        the forest memorise the training set, collapsing ``predict_proba``
        to near-0/1 votes and turning the affinity-score ranking into noise.
        0.008 was chosen on a KDD98 validation fold (not the test split)
        among 0.001/0.002/0.004/0.008 and confirmed on a second independent
        split; see ``scripts/benchmark_models_vs_baselines.py``.
    min_weight_fraction_leaf : float, default=0.0
        Minimum weighted fraction of the sum of weights required to be at a
        leaf node.  When ``class_weight`` is set, this interacts strongly with
        weight values and can be used to prevent minority-class leaves.
    class_weight : dict, "balanced", "balanced_subsample" or None, default=None
        Weight scheme for the two classes.  Pass ``"balanced"`` to let the
        model automatically compensate for class imbalance (recommended when
        major-donor examples are <5 % of your prospect pool), or supply an
        explicit dict such as ``{0: 1, 1: 10}`` for finer control.
    random_state : int or None, default=None
        Seed for the internal random-number generator.  Pass an integer to
        make model training fully reproducible, important for audit trails
        in gift-officer accountability dashboards.

    Attributes
    ----------
    estimator_ : RandomForestClassifier
        The fitted backend estimator.  Inspect via
        ``model.estimator_.feature_importances_`` to surface the top
        propensity drivers for stewardship reporting.
    classes_ : ndarray of shape (n_classes,)
        The unique class labels seen during :meth:`fit`.  Typically
        ``array([0, 1])``.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Examples
    --------
    **Basic usage with synthetic data:**

    >>> from philanthropy.datasets import generate_synthetic_donor_data
    >>> from philanthropy.models import DonorPropensityModel
    >>> df = generate_synthetic_donor_data(n_samples=500, random_state=0)
    >>> feature_cols = [
    ...     "total_gift_amount", "years_active", "event_attendance_count"
    ... ]
    >>> X = df[feature_cols].to_numpy()
    >>> y = df["is_major_donor"].to_numpy()
    >>> model = DonorPropensityModel(random_state=42)
    >>> model.fit(X, y)
    DonorPropensityModel(random_state=42)
    >>> scores = model.predict_affinity_score(X)
    >>> bool(scores.min() >= 0 and scores.max() <= 100)
    True

    **Pipeline integration:**

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> pipe = Pipeline([
    ...     ("scaler", StandardScaler()),
    ...     ("model", DonorPropensityModel(n_estimators=200, random_state=0)),
    ... ])
    >>> pipe.fit(X, y)
    Pipeline(...)

    Notes
    -----
    **Why RandomForest?**
    Random forests are a natural fit for philanthropic data science because:

    1. They handle the diverse mix of numerical and ordinal features common
       in CRM exports (recency in days, monetary amounts spanning four orders
       of magnitude, event counts) without feature scaling.
    2. Their ensemble nature provides stable, rank-ordered probability
       estimates suitable for affinity scoring (though not calibrated
       probabilities; see the note under :meth:`predict_affinity_score`).
    3. Feature importances are easily explained to non-technical gift
       officers and development committees.

    **Affinity Score Interpretation (0–100 scale):**

    ====== =================================
    Range  Recommended action
    ====== =================================
    80–100 Premium prospect: assign major gift officer immediately.
    60–79  Strong prospect: include in next biannual solicitation cycle.
    40–59  Moderate prospect: steward via annual fund or planned giving.
    0–39   Low propensity: retain in broad annual-appeal pool.
    ====== =================================

    **Isotonic calibration was checked and is not recommended.** Wrapping
    the model in ``CalibratedClassifierCV(method="isotonic", cv=5)`` on the
    KDD Cup 1998 validation fold (five seeds) lowered the decile
    calibration gap on only three seeds (it is already 0.003 to 0.005
    there) and lowered the top-10% hit rate on all five (e.g. 9.43 to 9.15
    of 100). The wrapper is still a valid sklearn pipeline step if you need
    probabilities on your own file; check it on held-out data first.

    See Also
    --------
    philanthropy.datasets.generate_synthetic_donor_data :
        Generate a synthetic prospect pool to prototype this model.
    philanthropy.metrics.donor_retention_rate :
        Measure year-over-year donor retention alongside propensity scoring.
    """

    def __sklearn_tags__(self) -> Tags:
        """Declare sklearn-compatible metadata tags for this estimator.

        Overrides the default :class:`ClassifierMixin` tags to indicate that
        ``DonorPropensityModel`` supports multi-class targets (inherited from
        the backend :class:`RandomForestClassifier`).

        Returns
        -------
        tags : Tags
            Populated sklearn Tags object.
        """
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = True
        tags.input_tags.allow_nan = True
        return tags

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: Union[int, float] = 0.008,
        min_weight_fraction_leaf: float = 0.0,
        class_weight: Any = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.class_weight = class_weight
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self: _SelfD, X: Any, y: Any) -> _SelfD:
        """Fit the DonorPropensityModel to labelled donor data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.  Accepts NumPy arrays or Pandas DataFrames.
            Common features include RFM metrics, event attendance counts,
            and wealth-screening capacity estimates.
        y : array-like of shape (n_samples,)
            Binary target vector.  ``1`` indicates a major-gift prospect;
            ``0`` indicates a standard annual-fund donor.

        Returns
        -------
        self : DonorPropensityModel
            Fitted estimator (enables method chaining).

        Raises
        ------
        ValueError
            If ``X`` and ``y`` have incompatible shapes, or if ``y``
            contains values outside ``{0, 1}``.
        """
        X, y = validate_data(self, X, y, ensure_all_finite="allow-nan", reset=True)

        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]

        self.estimator_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.estimator_.fit(X, y)

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict binary major-donor labels for each prospect.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.  Must have the same number of columns as
            the data passed to :meth:`fit`.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (``0`` or ``1``).

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
        return self.estimator_.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return class-probability estimates for each prospect.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Columns are ``[P(class=0), P(class=1)]``.  Each row sums to
            1.0.  The second column is the major-donor positive probability
            used internally by :meth:`predict_affinity_score`.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
        return self.estimator_.predict_proba(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Raw P(major_donor) scores. Used by sklearn scoring and calibration.

        Returns
        -------
        np.ndarray of shape (n_samples,), dtype float64
            Scores for each sample. Centered at 0 for binary case to match
            predict threshold.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
        proba = self.estimator_.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1] - 0.5
        elif proba.shape[1] == 1:
            # Single class case: if classes_ is [1], prob of class 1 is all 1.0.
            # If classes_ is [0], prob of class 1 is all 0.0.
            if self.classes_[0] == 1:
                return np.ones(proba.shape[0]) - 0.5
            return np.zeros(proba.shape[0]) - 0.5
        return proba  # Multiclass

    def predict_affinity_score(self, X: np.ndarray) -> np.ndarray:
        """Map major-donor probability to a 0–100 affinity score.

        This method is the primary interface for gift officers and CRM
        integrations.  The raw ``predict_proba`` positive-class probability is
        linearly rescaled from [0.0, 1.0] to [0, 100] and rounded to two
        decimal places, making scores directly comparable across fiscal years
        and prospect cohorts.

        Affinity scores are monotonically equivalent to model probabilities,
        so any rank-ordering derived from probabilities is preserved.  Scores
        do **not** represent calibrated probabilities and should not be
        interpreted as the literal odds of a major gift.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.  Accepts NumPy arrays or Pandas DataFrames.

        Returns
        -------
        affinity_scores : ndarray of shape (n_samples,)
            Float values in the closed interval [0.0, 100.0].  Higher
            scores indicate stronger major-gift propensity.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.

        Examples
        --------
        >>> import numpy as np
        >>> from philanthropy.datasets import generate_synthetic_donor_data
        >>> from philanthropy.models import DonorPropensityModel
        >>> df = generate_synthetic_donor_data(500, random_state=7)
        >>> X = df[["total_gift_amount", "years_active",
        ...          "event_attendance_count"]].to_numpy()
        >>> y = df["is_major_donor"].to_numpy()
        >>> model = DonorPropensityModel(random_state=0).fit(X, y)
        >>> scores = model.predict_affinity_score(X)
        >>> scores.shape
        (500,)
        >>> bool((scores >= 0).all() and (scores <= 100).all())
        True
        """
        df = self.decision_function(X)
        if df.ndim == 1:
            return np.round((df + 0.5) * 100, 2)
        # Multiclass case: affinity score for "major gift" (usually class 1)
        # We assume class 1 is at index 1 if it exists
        if df.shape[1] > 1:
            return np.round(df[:, 1] * 100, 2)
        return np.round(df.ravel() * 100, 2)


class MajorGiftClassifier(ClassifierMixin, BaseEstimator):
    """
    Classifies whether a donor is likely to make a major gift using calibrated probabilities.
    
    This uses HistGradientBoostingClassifier to handle missing data natively, and wraps
    it in a CalibratedClassifierCV so the output probabilities are true calibrated probabilities.
    Calibration uses sklearn's default sigmoid (Platt) method over 5 folds, the
    stable choice on small files; isotonic calibration needs far more rows per
    fold and is not used. Boosting stops early on its own once a file has more
    than 10,000 rows (sklearn's ``early_stopping="auto"``), so raising
    ``max_iter`` or lowering ``learning_rate`` rarely changes the ranking: on a
    KDD Cup 1998 validation fold (55/15/30 split, five seeds, response target),
    ``learning_rate=0.05, max_iter=300`` scored the same as the defaults.

    Parameters
    ----------
    max_iter : int, default=100
        Maximum number of boosting iterations for the underlying
        :class:`HistGradientBoostingClassifier`.
    learning_rate : float, default=0.1
        Shrinkage applied to each boosting iteration.
    min_samples_leaf : int, default=20
        Minimum number of samples per leaf, passed straight to the underlying
        :class:`HistGradientBoostingClassifier`.
    max_leaf_nodes : int or None, default=31
        Maximum number of leaves per tree, passed straight to the underlying
        :class:`HistGradientBoostingClassifier`.
    monotonic_cst : array-like of int or dict, default=None
        Monotonic constraint on each feature, passed straight to the
        underlying :class:`HistGradientBoostingClassifier`. ``None`` applies
        no constraint (current behaviour); see sklearn's docs for the
        ``{-1, 0, 1}`` per-feature encoding. Do not expect the textbook RFM
        constraint (+1 frequency and monetary, -1 recency) to help: on a
        KDD Cup 1998 validation fold (55/15/30 split, five seeds, response
        target) it lowered mean ROC-AUC from 0.601 to 0.595 and the top-10%
        hit rate from 9.5% to 9.0%.
    class_weight : dict, "balanced" or None, default=None
        Weight scheme for the two classes. Useful when the major-gift class
        is rare (e.g. a 2-3% base rate), where ``predict`` can otherwise
        favour the majority class almost exclusively. Applied as
        ``sample_weight`` during fitting rather than passed to the
        underlying :class:`HistGradientBoostingClassifier`'s own
        ``class_weight``, because :class:`CalibratedClassifierCV` recalibrates
        probabilities from cross-validated folds and would otherwise wash out
        (or even invert) a class-weighted base estimator's decision boundary.
        When ``class_weight`` is set, the calibrated probabilities reflect
        the reweighted class balance rather than the true one, so treat them
        as a ranking signal rather than literal chances of a major gift.
    random_state : int or None, default=None
        Seed for reproducibility.
    """
    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.classifier_tags.multi_class = True
        return tags
    
    def __init__(
        self,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 20,
        max_leaf_nodes: Optional[int] = 31,
        monotonic_cst: Any = None,
        class_weight: Any = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.monotonic_cst = monotonic_cst
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self: _SelfM, X: Any, y: Any) -> _SelfM:
        """Fit the classifier to labelled donor data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix. Missing values are accepted.
        y : array-like of shape (n_samples,)
            Binary target vector. ``1`` indicates a major-gift prospect.

        Returns
        -------
        self : MajorGiftClassifier
            Fitted estimator. Sets ``classes_``, ``n_features_in_``,
            ``estimator_``, and ``n_iter_`` (the mean boosting iterations
            across the calibration folds).
        """
        X, y = validate_data(self, X, y, ensure_all_finite="allow-nan", reset=True)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]

        base_estimator = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            monotonic_cst=self.monotonic_cst,
            random_state=self.random_state
        )
        self.estimator_ = CalibratedClassifierCV(base_estimator)
        # class_weight is applied as sample_weight rather than passed to the
        # base estimator's own class_weight param: CalibratedClassifierCV
        # recalibrates probabilities from cross-validated folds, which
        # otherwise washes out (and can even invert) a class-weighted base
        # estimator's decision boundary. Feeding the same weights as
        # sample_weight into fit keeps the calibration step consistent with
        # the requested class balance.
        sample_weight = (
            compute_sample_weight(self.class_weight, y)
            if self.class_weight is not None
            else None
        )
        self.estimator_.fit(X, y, sample_weight=sample_weight)
        # Mean boosting iterations across the calibration folds. Reporting a
        # hardcoded 1 here just to satisfy check_estimator would be masking.
        self.n_iter_ = int(
            np.mean([c.estimator.n_iter_ for c in self.estimator_.calibrated_classifiers_])
        )
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict binary major-donor labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix with the same columns used in :meth:`fit`.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (``0`` or ``1``).

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
        proba : ndarray of shape (n_samples, 2)
            Columns are ``[P(class=0), P(class=1)]``. The second column is
            the calibrated major-donor probability.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
        return self.estimator_.predict_proba(X)

    def predict_affinity_score(self, X: Any) -> np.ndarray:
        """Map the calibrated major-donor probability to a 0-100 score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Positive-class probability multiplied by 100 and rounded to two
            decimal places. Class ``1`` is treated as the positive class.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        """
        proba = self.predict_proba(X)
        if proba.shape[1] == 2:
            proba_positive = proba[:, 1]
        elif proba.shape[1] == 1:
            # Single class case (e.g. fit on all-0 or all-1 labels): classes_
            # has one entry, and predict_proba mirrors it with one column.
            if self.classes_[0] == 1:
                proba_positive = np.ones(proba.shape[0])
            else:
                proba_positive = np.zeros(proba.shape[0])
        else:
            proba_positive = proba[:, 1]  # Multiclass: class 1 by convention
        return np.round(proba_positive * 100.0, 2)
