"""
philanthropy.models._ask
=========================
Ask-amount recommendation and gift-array ("ask ladder") generation.

Deciding *how much to ask for* is one of the highest-leverage judgement calls
in major-gift fundraising.  Ask too low and the organisation leaves money on
the table; ask too high and the prospect disengages.  Gift officers therefore
work from a **gift array** (also called an *ask ladder*) — a short, discrete
menu of ascending amounts anchored on a single recommended **base ask**.

``AskAmountRecommender`` predicts that base ask amount from CRM, wealth-screening
and engagement features, and exposes :meth:`ask_ladder`, which expands
the base ask into the low / target / stretch rungs a gift officer presents in
a solicitation.

Under the hood the model deliberately uses
:class:`~sklearn.ensemble.HistGradientBoostingRegressor`, which handles
``NaN`` values natively, removing the need for an explicit imputation step when
wealth-screening data is partially missing.

Examples
--------
>>> import numpy as np
>>> from philanthropy.models import AskAmountRecommender
>>> rng = np.random.default_rng(0)
>>> X = rng.uniform(0, 1_000_000, (100, 5))
>>> y = rng.uniform(1_000, 250_000, 100)          # historical/target ask labels
>>> model = AskAmountRecommender(random_state=0)
>>> model.fit(X, y)
AskAmountRecommender(random_state=0)
>>> asks = model.predict(X)
>>> asks.shape
(100,)
>>> ladder = model.ask_ladder(X[:3])
>>> ladder.shape
(3, 3)
>>> bool((ladder[:, 2] >= ladder[:, 0]).all())
True
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.utils.validation import check_is_fitted, validate_data


class AskAmountRecommender(RegressorMixin, BaseEstimator):
    """Recommend a donor's base ask amount and derive a gift array.

    ``AskAmountRecommender`` is a scikit-learn–compatible regressor that wraps
    :class:`~sklearn.ensemble.HistGradientBoostingRegressor` to predict the
    **base ask amount** — the single dollar figure a gift officer anchors a
    solicitation on for a given prospect.

    By using ``HistGradientBoostingRegressor`` internally, the model handles
    missing CRM and wealth-screening values *natively* without requiring an
    upstream imputation step, reducing pipeline complexity and eliminating one
    source of potential leakage.

    The companion method :meth:`ask_ladder` expands the base ask into a
    discrete **gift array** (or *ask ladder*) — the low / target / stretch
    rungs presented in a real solicitation.

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
    ask_floor : float, default=1.0
        Minimum recommended ask (in dollars).  Predictions are clipped to
        this floor via ``np.maximum`` to prevent negative ask amounts that
        are semantically meaningless.

    Attributes
    ----------
    estimator_ : HistGradientBoostingRegressor
        The fitted backend estimator.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Examples
    --------
    **Predict a base ask and expand it into a gift array:**

    >>> import numpy as np
    >>> from philanthropy.models import AskAmountRecommender
    >>> rng = np.random.default_rng(42)
    >>> X = rng.uniform(0, 1e6, (200, 6))
    >>> y = rng.uniform(1e3, 250_000, 200)
    >>> model = AskAmountRecommender(random_state=42).fit(X, y)
    >>> model.predict(X[:3]).shape
    (3,)
    >>> ladder = model.ask_ladder(X[:3])
    >>> ladder.shape
    (3, 3)
    >>> bool((ladder[:, 2] >= ladder[:, 1]).all())
    True

    **Pipeline usage:**

    >>> from sklearn.pipeline import Pipeline
    >>> pipe = Pipeline([("model", AskAmountRecommender(random_state=0))])
    >>> _ = pipe.fit(X, y)

    Notes
    -----
    **Why HistGradientBoosting?**
    Wealth-screening datasets consistently contain 30–70 % missing values.
    ``HistGradientBoostingRegressor`` implements a native missing-value
    splitting strategy that treats ``NaN`` as an informative category rather
    than an erroneous artefact, avoiding the information loss of mean/median
    imputation.

    **Gift Array Interpretation:**

    The default ``multipliers=(1.0, 1.5, 2.5)`` map the base ask onto three
    rungs a gift officer works from:

    ======= ==================================================
    Rung    Meaning
    ======= ==================================================
    Low     The base ask — a comfortable, likely-accepted gift.
    Target  1.5× the base — the amount the ask is anchored on.
    Stretch 2.5× the base — the aspirational upgrade ask.
    ======= ==================================================

    See Also
    --------
    philanthropy.models.ShareOfWalletRegressor :
        Continuous capacity model — pair with this recommender to bound the
        top of the gift array by estimated philanthropic capacity.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_depth: Optional[int] = None,
        l2_regularization: float = 0.0,
        min_samples_leaf: int = 20,
        random_state: Optional[int] = None,
        ask_floor: float = 1.0,
    ) -> None:
        # scikit-learn rule: __init__ stores parameters and does NO logic.
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.l2_regularization = l2_regularization
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.ask_floor = ask_floor

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.regressor_tags.poor_score = True
        return tags

    @property
    def n_iter_(self):
        """Number of iterations run by the backend estimator."""
        check_is_fitted(self, ["estimator_"])
        return self.estimator_.n_iter_

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y) -> "AskAmountRecommender":
        """Fit the ask-amount recommender to labelled prospect data."""
        X, y = validate_data(self, X, y, ensure_all_finite="allow-nan", reset=True)
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
        """Predict the base ask amount for each prospect."""
        check_is_fitted(self, ["estimator_"])
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
        raw = self.estimator_.predict(X)
        return np.maximum(raw, self.ask_floor)

    def ask_ladder(
        self,
        X,
        multipliers=(1.0, 1.5, 2.5),
    ) -> np.ndarray:
        """Return a discrete gift array (ask ladder) for each prospect.

        The base ask (from :meth:`predict`) multiplied by each entry of
        ``multipliers`` gives the low / target / stretch rungs a gift officer
        works from when structuring a solicitation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix passed to :meth:`predict`.  May contain ``NaN``.
        multipliers : sequence of float, default=(1.0, 1.5, 2.5)
            Ascending positive factors applied to the base ask to build each
            rung of the gift array.  Must be non-empty and strictly positive.

        Returns
        -------
        ask_array : ndarray of shape (n_samples, len(multipliers))
            Element ``[i, j]`` is ``base_ask[i] * multipliers[j]``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        ValueError
            If ``multipliers`` is empty or contains a non-positive value.

        Examples
        --------
        >>> import numpy as np
        >>> from philanthropy.models import AskAmountRecommender
        >>> rng = np.random.default_rng(7)
        >>> X = rng.uniform(0, 1e6, (50, 4))
        >>> y = rng.uniform(1e3, 1e5, 50)
        >>> model = AskAmountRecommender(random_state=7).fit(X, y)
        >>> ladder = model.ask_ladder(X, multipliers=(1.0, 2.0, 4.0))
        >>> ladder.shape
        (50, 3)
        >>> bool((ladder[:, 2] >= ladder[:, 0]).all())
        True
        """
        multipliers = np.asarray(multipliers, dtype=float)
        if multipliers.size == 0:
            raise ValueError("`multipliers` must be non-empty.")
        if not np.all(multipliers > 0):
            raise ValueError("`multipliers` must all be strictly positive.")

        base_ask = self.predict(X)
        return base_ask[:, None] * multipliers[None, :]
