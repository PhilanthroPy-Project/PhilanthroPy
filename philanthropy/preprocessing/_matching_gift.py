"""
philanthropy.preprocessing._matching_gift
==========================================
Corporate matching-gift featurization for donor feature matrices.

Many employers match their employees' charitable gifts at a fixed ratio
(1:1, 2:1, ...). This transformer turns a donor's employer name and gift
amount into matching-gift signals (whether an employer is present, the known
corporate match ratio, and the potential matched dollars), so a propensity or
prioritisation model can weight prospects with employer-match upside.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _normalise_employer(val) -> str:
    """Return a lookup key for an employer cell.

    Null cells and whitespace-only strings normalise to the empty string,
    which signals "no employer" for both the ``has_employer`` flag and the
    ``match_ratios_`` lookup (empty key is never matched).
    """
    if pd.isna(val):
        return ""
    return str(val).strip().lower()


class MatchingGiftFeaturizer(TransformerMixin, BaseEstimator):
    """Derive corporate matching-gift features for each donor row.

    For every row the transformer emits three features: whether an employer is
    on file, the known corporate match ratio for that employer, and the
    potential matched dollars (gift amount times match ratio). The match-ratio
    lookup is normalised once at fit time (keys lowercased and stripped) and
    frozen, so ``transform`` depends only on each row's own values and never on
    which rows happen to share the batch.

    Parameters
    ----------
    employer_col : str, default="employer"
        Column in ``X`` holding the donor's employer name.
    gift_col : str, default="gift_amount"
        Column in ``X`` holding the gift amount used to size the potential
        matched dollars. Coerced to numeric at transform time; non-numeric or
        missing values are treated as ``0``.
    match_ratios : dict of {str: float} or None, default=None
        Mapping of employer name to corporate match ratio (e.g.
        ``{"Boeing": 1.0, "Microsoft": 2.0}``). Keys are matched
        case-insensitively (lowercased and stripped). ``None`` means no known
        employers, so every ``match_ratio`` is ``0.0``.

    Attributes
    ----------
    match_ratios_ : dict of {str: float}
        Normalised copy of ``match_ratios`` (keys lowercased/stripped, values
        cast to ``float``), frozen at fit time. Empty dict when
        ``match_ratios`` is ``None``.
    n_features_in_ : int
        Number of columns seen at fit time.
    feature_names_in_ : ndarray of str
        Column names of ``X`` at fit time.

    Raises
    ------
    TypeError
        If ``X`` is not a pandas DataFrame.
    ValueError
        If ``employer_col`` or ``gift_col`` is missing from ``X``.

    Notes
    -----
    The three output columns, in order, are:

    ========================== ===============================================
    Column                     Description
    ========================== ===============================================
    ``has_employer``           ``1.0`` if the employer cell is non-null and a
                               non-empty string, else ``0.0``.
    ``match_ratio``            ``match_ratios_`` lookup for the normalised
                               employer, ``0.0`` when unknown.
    ``potential_matched_amount`` Numeric gift amount (NaN -> 0) times
                               ``match_ratio``.
    ========================== ===============================================

    Examples
    --------
    >>> import pandas as pd
    >>> from philanthropy.preprocessing import MatchingGiftFeaturizer
    >>> X = pd.DataFrame({
    ...     "employer": ["Boeing", "", "Acme"],
    ...     "gift_amount": [100.0, 50.0, 200.0],
    ... })
    >>> feat = MatchingGiftFeaturizer(match_ratios={"Boeing": 1.0})
    >>> feat.fit(X).transform(X)
    array([[  1.,   1., 100.],
           [  0.,   0.,   0.],
           [  1.,   0.,   0.]])
    """

    def __init__(
        self,
        employer_col: str = "employer",
        gift_col: str = "gift_amount",
        match_ratios: dict[str, float] | None = None,
    ) -> None:
        self.employer_col = employer_col
        self.gift_col = gift_col
        self.match_ratios = match_ratios

    def _check_columns(self, X: pd.DataFrame) -> None:
        """Raise if the required schema columns are absent from ``X``."""
        missing = [
            col
            for col in (self.employer_col, self.gift_col)
            if col not in X.columns
        ]
        if missing:
            raise ValueError(f"X is missing required columns: {missing}")

    def fit(self, X, y=None) -> "MatchingGiftFeaturizer":
        """Register the input schema and freeze the match-ratio lookup.

        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_features)
            Donor-level feature matrix. Must contain ``employer_col`` and
            ``gift_col``.
        y : ignored

        Returns
        -------
        self : MatchingGiftFeaturizer

        Raises
        ------
        TypeError
            If ``X`` is not a pandas DataFrame.
        ValueError
            If ``employer_col`` or ``gift_col`` is missing from ``X``.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        self._check_columns(X)

        self.feature_names_in_ = np.array(X.columns.tolist(), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)

        # Freeze the normalised lookup from constructor params (leakage-safety
        # contract: fitted statistics are computed in fit and frozen before
        # transform). Nothing here depends on the contents of X.
        self.match_ratios_ = {
            str(k).strip().lower(): float(v)
            for k, v in (self.match_ratios or {}).items()
        }
        return self

    def transform(self, X) -> np.ndarray:
        """Emit matching-gift features for each row of ``X``.

        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_features)
            Donor-level feature matrix. Must contain ``employer_col`` and
            ``gift_col``.

        Returns
        -------
        X_out : np.ndarray of shape (n_samples, 3), dtype float64
            Columns in order: ``has_employer``, ``match_ratio``,
            ``potential_matched_amount``.

        Raises
        ------
        TypeError
            If ``X`` is not a pandas DataFrame.
        ValueError
            If ``employer_col`` or ``gift_col`` is missing from ``X``.
        """
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        self._check_columns(X)

        norm = X[self.employer_col].map(_normalise_employer)
        has_employer = (norm != "").to_numpy(dtype=np.float64)
        match_ratio = norm.map(
            lambda key: self.match_ratios_.get(key, 0.0)
        ).to_numpy(dtype=np.float64)

        gift = (
            pd.to_numeric(X[self.gift_col], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        potential = gift * match_ratio

        return np.column_stack([has_employer, match_ratio, potential]).astype(
            np.float64
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return the generated matching-gift feature names.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Ignored because the featurizer always emits the same three features.

        Returns
        -------
        feature_names_out : ndarray of str
            ``["has_employer", "match_ratio",
            "potential_matched_amount"]``.

        Raises
        ------
        NotFittedError
            If the featurizer has not been fitted.
        """
        check_is_fitted(self)
        return np.array(
            ["has_employer", "match_ratio", "potential_matched_amount"],
            dtype=object,
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags._skip_test = True  # Schema-dependent
        return tags
