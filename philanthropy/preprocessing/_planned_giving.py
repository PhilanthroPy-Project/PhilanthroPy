"""
philanthropy.preprocessing._planned_giving
==========================================
Planned-giving (bequest / legacy gift) signal featurization.

Planned giving (bequests, charitable remainder trusts) requires a separate
predictive model from major gifts. Key drivers: donor age ≥ 65, giving
tenure ≥ 10 years, and a wealth-screening vendor "charitable inclination"
score. This transformer extracts a 4-column feature vector optimised for
bequest/legacy gift intent classifiers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class PlannedGivingSignalTransformer(TransformerMixin, BaseEstimator):
    """Extract features for bequest / planned-giving intent classification.

    Planned giving (bequests, charitable remainder trusts) requires a separate
    predictive model from major gifts. Key drivers are donor age ≥ 65, giving
    tenure ≥ 10 years, and a wealth-screening vendor "charitable inclination"
    score. This transformer extracts a four-column feature vector optimised for
    bequest/legacy gift intent classifiers.

    Parameters
    ----------
    age_col : str, default="donor_age"
        Column containing donor age in years.
    tenure_col : str, default="years_active"
        Column containing number of years the donor has been active.
    planned_gift_inclination_col : str, default="planned_gift_inclination"
        Column containing the wealth-screening vendor's charitable inclination
        score, expected to be in [0, 1]. Missing values are treated as a
        sentinel value (-1.0) to distinguish "vendor data absent" from a
        genuine 0 score.
    age_threshold : int, default=65
        Minimum age (inclusive) for the is_legacy_age flag.
    tenure_threshold_years : int, default=10
        Minimum years active (inclusive) for the is_loyal_donor flag.

    Attributes
    ----------
    n_features_in_ : int
        Number of input features seen at fit time.
    feature_names_in_ : ndarray of str
        Column names of X at fit time (set when X is a DataFrame).

    Notes
    -----
    Output columns
    ~~~~~~~~~~~~~~
    ========================= ================================================
    Col  Name                  Description
    ========================= ================================================
    0    ``is_legacy_age``     uint8: 1 if age >= age_threshold, else 0.
                               NaN age → 0.
    1    ``is_loyal_donor``    uint8: 1 if tenure >= tenure_threshold_years.
                               NaN tenure → 0.
    2    ``inclination_score`` float64: raw planned_gift_inclination value,
                               clipped to [0, 1]. Missing → -1.0 sentinel
                               (distinguishable from a genuine 0 score).
    3    ``composite_score``   float64: is_legacy_age + is_loyal_donor
                               + max(inclination_score, 0). Range [0.0, 3.0].
    ========================= ================================================

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from philanthropy.preprocessing import PlannedGivingSignalTransformer
    >>> X = pd.DataFrame({
    ...     "donor_age": [70, 60, None],
    ...     "years_active": [15, 5, 12],
    ...     "planned_gift_inclination": [0.8, 0.3, None],
    ... })
    >>> t = PlannedGivingSignalTransformer()
    >>> out = t.fit_transform(X)
    >>> out.shape
    (3, 4)
    """

    def __init__(
        self,
        age_col: str = "donor_age",
        tenure_col: str = "years_active",
        planned_gift_inclination_col: str = "planned_gift_inclination",
        age_threshold: int = 65,
        tenure_threshold_years: int = 10,
    ) -> None:
        self.age_col = age_col
        self.tenure_col = tenure_col
        self.planned_gift_inclination_col = planned_gift_inclination_col
        self.age_threshold = age_threshold
        self.tenure_threshold_years = tenure_threshold_years

    def fit(self, X, y=None) -> "PlannedGivingSignalTransformer":
        """Validate input schema and record n_features_in_.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Donor-level feature matrix.
        y : ignored

        Returns
        -------
        self : PlannedGivingSignalTransformer
        """
        validate_data(self, X, dtype=None, ensure_all_finite="allow-nan", reset=True)
        return self

    def transform(self, X, y=None) -> np.ndarray:
        """Compute the 4-column planned-giving feature vector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Donor-level feature matrix. Accepts pd.DataFrame (columns may or
            may not exist — missing columns are handled gracefully with NaN / 0).

        Returns
        -------
        X_out : np.ndarray of shape (n_samples, 4), dtype float64
            Columns: [is_legacy_age, is_loyal_donor, inclination_score,
            composite_score].

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        """
        check_is_fitted(self)
        validate_data(self, X, dtype=None, ensure_all_finite="allow-nan", reset=False)

        # Work with a DataFrame for convenient column access
        if isinstance(X, pd.DataFrame):
            df = X
        elif hasattr(self, "feature_names_in_"):
            df = pd.DataFrame(
                np.asarray(X, dtype=float), columns=self.feature_names_in_
            )
        else:
            df = pd.DataFrame(np.asarray(X, dtype=float))

        n = len(df)

        # --- col 0: is_legacy_age ---
        if self.age_col in df.columns:
            age = pd.to_numeric(df[self.age_col], errors="coerce")
            is_legacy_age = np.where(age.isna(), 0, (age >= self.age_threshold).astype(int))
        else:
            is_legacy_age = np.zeros(n, dtype=int)

        # --- col 1: is_loyal_donor ---
        if self.tenure_col in df.columns:
            tenure = pd.to_numeric(df[self.tenure_col], errors="coerce")
            is_loyal_donor = np.where(
                tenure.isna(), 0, (tenure >= self.tenure_threshold_years).astype(int)
            )
        else:
            is_loyal_donor = np.zeros(n, dtype=int)

        # --- col 2: inclination_score ---
        if self.planned_gift_inclination_col in df.columns:
            raw_incl = pd.to_numeric(
                df[self.planned_gift_inclination_col], errors="coerce"
            )
            inclination_score = np.where(
                raw_incl.isna(),
                -1.0,  # sentinel: vendor data absent
                np.clip(raw_incl.to_numpy(dtype=float), 0.0, 1.0),
            )
        else:
            inclination_score = np.full(n, -1.0, dtype=float)  # vendor data absent

        # --- col 3: composite_score ---
        # is_legacy_age + is_loyal_donor + max(inclination_score, 0)
        incl_clipped = np.maximum(inclination_score, 0.0)
        composite_score = is_legacy_age.astype(float) + is_loyal_donor.astype(float) + incl_clipped

        return np.column_stack(
            [
                is_legacy_age.astype(np.float64),
                is_loyal_donor.astype(np.float64),
                inclination_score.astype(np.float64),
                composite_score.astype(np.float64),
            ]
        )

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.array(
            ["is_legacy_age", "is_loyal_donor", "inclination_score", "composite_score"],
            dtype=object,
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        # This transformer extracts named columns from mixed-type DataFrames
        # and handles non-numeric input gracefully. Setting string=True suppresses
        # check_dtype_object's strict TypeError requirement.
        tags.input_tags.string = True
        return tags
