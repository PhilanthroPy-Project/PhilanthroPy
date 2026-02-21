"""
philanthropy.models._wallet
============================
Share-of-Wallet capacity estimation for major-gift fundraising.

Academic medical centre (AMC) advancement teams need not only a *binary*
propensity signal ("will this donor upgrade?") but also a *continuous*
estimate of their total philanthropic capacity — what fundraising professionals
call the donor's **share of wallet (SoW)**.  This capacity prediction drives:

* Minimum and maximum ask amounts in gift-range tables
* Portfolio assignment decisions (which gift officer manages this prospect?)
* ROI modelling for grateful-patient programmes

``ShareOfWalletRegressor`` predicts the continuous dollar-denominated total
philanthropic capacity of each prospect from CRM and wealth-screening features.
It also exposes :meth:`predict_capacity_ratio`, which compares the predicted
capacity against historical cumulative giving and surfaces a **untapped-capacity
ratio** — the primary metric used by major-gift officers to prioritise outreach.

Under the hood the model deliberately uses
:class:`~sklearn.ensemble.HistGradientBoostingRegressor`, which handles
``NaN`` values natively, removing the need for an explicit imputation step when
wealth-screening data is partially missing.

Examples
--------
>>> import numpy as np
>>> from philanthropy.models import ShareOfWalletRegressor
>>> rng = np.random.default_rng(0)
>>> X = rng.uniform(0, 1_000_000, (100, 5))
>>> y = rng.uniform(10_000, 5_000_000, 100)      # total capacity labels
>>> model = ShareOfWalletRegressor(random_state=0)
>>> model.fit(X, y)
ShareOfWalletRegressor(random_state=0)
>>> caps = model.predict(X)
>>> caps.shape
(100,)
>>> ratios = model.predict_capacity_ratio(X, historical_giving=rng.uniform(0, 500_000, 100))
>>> (ratios > 0).all()
True
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y, validate_data


class ShareOfWalletRegressor(RegressorMixin, BaseEstimator):
    """Predict a donor's total philanthropic capacity (share-of-wallet).

    ``ShareOfWalletRegressor`` is a scikit-learn–compatible regressor that
    wraps :class:`~sklearn.ensemble.HistGradientBoostingRegressor` to estimate
    a prospect's **total philanthropic capacity** — i.e., the maximum lifetime
    gift they *could* make given their wealth profile, giving history, and
    engagement signals.

    By using ``HistGradientBoostingRegressor`` internally, the model handles
    missing CRM and wealth-screening values *natively* without requiring an
    upstream imputation step, reducing pipeline complexity and eliminating one
    source of potential leakage.

    The companion method :meth:`predict_capacity_ratio` exposes the
    **untapped-capacity ratio** (predicted capacity ÷ historical cumulative
    giving), the primary metric gift officers use to prioritise discovery
    calls and major-gift portfolios.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Step size shrinkage applied to each tree.  Smaller values require
        more ``max_iter`` trees to converge but typically generalise better.
    max_iter : int, default=100
        Number of boosting iterations (trees).  Increase to 300–500 for
        production models trained on large prospect pools.
    max_depth : int or None, default=None
        Maximum depth of each individual decision tree.
    l2_regularization : float, default=0.0
        L2 regularisation term on leaf weights.  Increase (e.g., to 1.0)
        to combat overfitting when the feature-to-sample ratio is high —
        a common scenario in small-shop advancement analytics.
    min_samples_leaf : int, default=20
        Minimum number of samples per leaf.  Larger values prevent
        overfitting on sparse major-donor training sets.
    random_state : int or None, default=None
        Seed for the internal random-number generator.  Set to an integer
        for reproducible model artefacts suitable for audit trails.
    capacity_floor : float, default=1.0
        Minimum predicted capacity (in dollars).  Predictions are clipped
        to this floor via ``np.maximum`` to prevent negative capacity
        estimates that are semantically meaningless.

    Attributes
    ----------
    estimator_ : HistGradientBoostingRegressor
        The fitted backend estimator.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Examples
    --------
    **Predict raw capacity and untapped-capacity ratio:**

    >>> import numpy as np
    >>> from philanthropy.models import ShareOfWalletRegressor
    >>> rng = np.random.default_rng(42)
    >>> X = rng.uniform(0, 1e6, (200, 6))
    >>> y = rng.uniform(5e4, 5e6, 200)
    >>> historical = rng.uniform(1e3, 5e5, 200)
    >>> model = ShareOfWalletRegressor(random_state=42).fit(X, y)
    >>> model.predict(X[:3]).shape
    (3,)
    >>> ratios = model.predict_capacity_ratio(X[:3], historical_giving=historical[:3])
    >>> (ratios >= 0).all()
    True

    **Pipeline usage:**

    >>> from sklearn.pipeline import Pipeline
    >>> from philanthropy.preprocessing import WealthScreeningImputer
    >>> # WealthScreeningImputer only used here for non-NaN-native context;
    >>> # ShareOfWalletRegressor can handle NaN inputs natively.
    >>> pipe = Pipeline([("model", ShareOfWalletRegressor(random_state=0))])
    >>> _ = pipe.fit(X, y)

    Notes
    -----
    **Why HistGradientBoosting?**
    Wealth-screening datasets consistently contain 30–70 % missing values.
    ``HistGradientBoostingRegressor`` implements a native missing-value
    splitting strategy that treats ``NaN`` as an informative category rather
    than an erroneous artefact, avoiding the information loss of mean/median
    imputation.

    **Capacity Ratio Interpretation:**

    ====== =====================================================
    Ratio  Recommended action
    ====== =====================================================
    ≥ 10×  Dramatically under-asked; schedule discovery call.
    5–9×   Significant untapped potential; major-gift candidate.
    2–4×   Moderate upside; consider upgrade ask.
    < 2×   Near capacity; focus on retention and stewardship.
    ====== =====================================================

    See Also
    --------
    philanthropy.models.DonorPropensityModel :
        Binary propensity model — use alongside this regressor for a
        two-stage (propensity × capacity) portfolio ranking.
    philanthropy.preprocessing.WealthScreeningImputer :
        Optional upstream imputer for non-NaN-native downstream models.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_depth: Optional[int] = None,
        l2_regularization: float = 0.0,
        min_samples_leaf: int = 20,
        random_state: Optional[int] = None,
        capacity_floor: float = 1.0,
    ) -> None:
        # scikit-learn rule: __init__ stores parameters and does NO logic.
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.l2_regularization = l2_regularization
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.capacity_floor = capacity_floor

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y) -> "ShareOfWalletRegressor":
        """Fit the share-of-wallet capacity model to labelled prospect data."""
        X, y = validate_data(self, X, y, force_all_finite="allow-nan", reset=True)
        self.n_features_in_ = X.shape[1]

        self.estimator_ = HistGradientBoostingRegressor(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            l2_regularization=self.l2_regularization,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.estimator_.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        """Predict philanthropic capacity for each prospect."""
        check_is_fitted(self, ["estimator_"])
        X = validate_data(self, X, force_all_finite="allow-nan", reset=False)
        raw = self.estimator_.predict(X)
        return np.maximum(raw, self.capacity_floor)

    def predict_capacity_ratio(
        self,
        X,
        historical_giving: np.ndarray,
    ) -> np.ndarray:
        """Return the predicted capacity-to-historical-giving ratio.

        This ratio is the primary metric for gift officers prioritising
        discovery calls.  A ratio of 5.0 means the model estimates the donor
        could give five times more than they have historically — a strong
        signal of untapped major-gift potential.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix passed to :meth:`predict`.  May contain ``NaN``.
        historical_giving : array-like of shape (n_samples,)
            Each donor's **cumulative historical giving** in dollars.  Values
            of zero or negative are replaced with ``1.0`` (the
            ``capacity_floor`` fallback) to avoid division-by-zero errors
            and to ensure semantically valid ratios for new donors with no
            prior giving history.

        Returns
        -------
        capacity_ratio : ndarray of shape (n_samples,)
            Element-wise ratio ``predicted_capacity / max(historical_giving, 1.0)``.
            Values ≥ 1.0 indicate untapped capacity; values < 1.0 indicate
            that the predicted capacity is below current cumulative giving
            (which may signal an over-generous prior record or model noise).

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        ValueError
            If ``historical_giving`` length does not match the number of
            rows in ``X``.

        Examples
        --------
        >>> import numpy as np
        >>> from philanthropy.models import ShareOfWalletRegressor
        >>> rng = np.random.default_rng(7)
        >>> X = rng.uniform(0, 1e6, (50, 4))
        >>> y = rng.uniform(1e4, 1e6, 50)
        >>> hist = rng.uniform(500, 1e5, 50)
        >>> model = ShareOfWalletRegressor(random_state=7).fit(X, y)
        >>> ratios = model.predict_capacity_ratio(X, historical_giving=hist)
        >>> ratios.shape
        (50,)
        >>> (ratios > 0).all()
        True
        """
        check_is_fitted(self, ["estimator_"])
        predicted_capacity = self.predict(X)

        historical_giving = np.asarray(historical_giving, dtype=float)
        if predicted_capacity.shape[0] != historical_giving.shape[0]:
            raise ValueError(
                f"`historical_giving` must have the same length as the number "
                f"of rows in ``X`` ({predicted_capacity.shape[0]}), "
                f"got {historical_giving.shape[0]}."
            )

        # Clip denominator to prevent division by zero for new/zero-giving donors
        safe_giving = np.maximum(historical_giving, 1.0)
        return predicted_capacity / safe_giving
