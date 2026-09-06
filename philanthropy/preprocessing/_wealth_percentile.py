from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data


class WealthPercentileTransformer(TransformerMixin, BaseEstimator):
    """
    Computes wealth percentile ranks.
    """

    def __init__(
        self,
        wealth_cols: list[str] | None = None,
        output_suffix: str = "_pct_rank"
    ):
        self.wealth_cols = wealth_cols
        self.output_suffix = output_suffix

    def fit(self, X, y=None):
        """Learn the training wealth distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training-set feature matrix.
        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        self : WealthPercentileTransformer
            Fitted transformer. Freezes ``feature_names_in_``,
            ``imputed_cols_``, and ``percentile_lookup_``.

        Notes
        -----
        ``percentile_lookup_`` stores the sorted training values per wealth
        column. :meth:`transform` ranks held-out data against this frozen
        training distribution, not against the batch being transformed.
        """
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=True)
        # validate_data sets feature_names_in_ when input is a DataFrame
        if not hasattr(self, "feature_names_in_"):
            self.feature_names_in_ = np.array([f"x{i}" for i in range(X.shape[1])], dtype=object)

        # Use feature_names_in_ to resolve columns
        if self.wealth_cols is not None:
            self.imputed_cols_ = [c for c in self.wealth_cols if c in self.feature_names_in_]
        else:
            targets = ("net_worth", "real_estate", "stock", "capacity")
            self.imputed_cols_ = [c for c in self.feature_names_in_ if any(t in str(c) for t in targets)]

        self.percentile_lookup_ = {}
        for col in self.imputed_cols_:
            # Find index of column
            col_idx = list(self.feature_names_in_).index(col)
            # Use X as numpy array
            s = pd.to_numeric(pd.Series(X[:, col_idx]), errors="coerce")
            valid_vals = s.dropna().to_numpy()
            self.percentile_lookup_[col] = np.sort(valid_vals)

        if len(self.imputed_cols_) == 0 and self.wealth_cols is not None:
             # If we expected columns but found none, we should probably warn or raise
             pass

        return self

    def transform(self, X):
        """Rank features against the fitted training distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix (training or held-out).

        Returns
        -------
        X_out : np.ndarray of float64
            Numeric feature matrix with wealth percentile columns appended.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.

        Notes
        -----
        Percentiles are relative to the training cohort captured by
        ``percentile_lookup_``, which is the leakage-safety guarantee.
        """
        check_is_fitted(self, "percentile_lookup_")
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
        X_out = pd.DataFrame(X, columns=self.feature_names_in_)

        for col in self.imputed_cols_:
            if col in X_out.columns:
                ref = self.percentile_lookup_[col]
                s = pd.to_numeric(X_out[col], errors="coerce").to_numpy(dtype=float)
                out_col = f"{col}{self.output_suffix}"
                
                if len(ref) == 0:
                    X_out[out_col] = np.nan
                    continue

                ranks = np.searchsorted(ref, s, side="right") / float(len(ref)) * 100.0
                ranks = np.where(np.isnan(s), np.nan, ranks)
                X_out[out_col] = ranks

        # Rule 5: transform() MUST return np.ndarray (float64)
        X_final = X_out.select_dtypes(include=[np.number])
        return X_final.to_numpy(dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        """Return input names followed by generated wealth-percentile names.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Ignored. Names are derived from the columns recorded by :meth:`fit`.

        Returns
        -------
        feature_names_out : ndarray of str
            Fitted input names followed by ``<column><output_suffix>`` for each
            selected wealth column.

        Raises
        ------
        NotFittedError
            If the transformer has not been fitted.
        """
        check_is_fitted(self)
        out = list(self.feature_names_in_)
        for col in self.imputed_cols_:
            if col in self.feature_names_in_:
                out.append(f"{col}{self.output_suffix}")
        return np.array(out, dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
