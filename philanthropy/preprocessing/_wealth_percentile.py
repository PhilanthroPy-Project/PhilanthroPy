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

    def _resolve_cols(self, X: pd.DataFrame) -> list[str]:
        if self.wealth_cols is not None:
            return [c for c in self.wealth_cols if c in X.columns]
        
        targets = ("net_worth", "real_estate", "stock", "capacity")
        return [c for c in X.columns if any(t in c for t in targets)]

    def fit(self, X, y=None):
        X = validate_data(self, X, reset=True)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns.tolist(), dtype=object)
        self.n_features_in_ = X.shape[1]

        self.imputed_cols_ = self._resolve_cols(X)
        self.percentile_lookup_ = {}

        for col in self.imputed_cols_:
            s = pd.to_numeric(X[col], errors="coerce")
            valid_vals = s.dropna().to_numpy()
            valid_vals.sort()
            self.percentile_lookup_[col] = valid_vals

        return self

    def transform(self, X):
        check_is_fitted(self, "percentile_lookup_")
        X = validate_data(self, X, reset=False)
        X_out = X.copy() if hasattr(X, "columns") else pd.DataFrame(X)

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

        return X_out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        out = list(self.feature_names_in_)
        for col in self.imputed_cols_:
            if col in self.feature_names_in_:
                out.append(f"{col}{self.output_suffix}")
        return np.array(out, dtype=object)

    def _more_tags(self):
        return {"X_types": ["2darray", "dataframe"]}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
