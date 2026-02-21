import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data


class PlannedGivingIndicator(TransformerMixin, BaseEstimator):
    """
    Creates planned giving indicator features.
    """

    def __init__(
        self,
        age_col: str = "age",
        tenure_col: str = "years_active",
        gift_count_col: str = "lifetime_gift_count",
        has_children_col: str | None = None,
        high_propensity_age_threshold: int = 65,
        high_propensity_tenure_threshold: int = 15,
    ):
        self.age_col = age_col
        self.tenure_col = tenure_col
        self.gift_count_col = gift_count_col
        self.has_children_col = has_children_col
        self.high_propensity_age_threshold = high_propensity_age_threshold
        self.high_propensity_tenure_threshold = high_propensity_tenure_threshold

    def fit(self, X, y=None):
        X = validate_data(self, X, reset=True)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns.tolist(), dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X_out = X.copy() if hasattr(X, "columns") else pd.DataFrame(X)

        if self.age_col in X_out.columns:
            age_s = pd.to_numeric(X_out[self.age_col], errors="coerce")
            X_out["pg_age_band"] = pd.cut(
                age_s,
                bins=[-np.inf, 45, 60, 75, np.inf],
                right=False,
                labels=[0, 1, 2, 3]
            ).astype("float64").astype("Int64")

        if self.tenure_col in X_out.columns:
            tenure_s = pd.to_numeric(X_out[self.tenure_col], errors="coerce")
            X_out["pg_tenure_decades"] = tenure_s / 10.0

        if self.gift_count_col in X_out.columns and self.tenure_col in X_out.columns:
            gifts_s = pd.to_numeric(X_out[self.gift_count_col], errors="coerce")
            tenure_s = pd.to_numeric(X_out[self.tenure_col], errors="coerce")
            X_out["pg_loyalty_score"] = gifts_s * np.log1p(tenure_s)

        if self.age_col in X_out.columns and self.tenure_col in X_out.columns:
            age_s = pd.to_numeric(X_out[self.age_col], errors="coerce")
            tenure_s = pd.to_numeric(X_out[self.tenure_col], errors="coerce")

            cond_age = (age_s >= self.high_propensity_age_threshold)
            cond_tenure = (tenure_s >= self.high_propensity_tenure_threshold)
            
            cond_has_children = True
            if self.has_children_col is not None and self.has_children_col in X_out.columns:
                children_s = pd.to_numeric(X_out[self.has_children_col], errors="coerce").fillna(0).astype(bool)
                cond_has_children = ~children_s

            X_out["pg_high_propensity"] = (cond_age & cond_tenure & cond_has_children).astype(np.uint8)

        return X_out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        out = list(self.feature_names_in_)
        if self.age_col in out:
            out.append("pg_age_band")
        if self.tenure_col in out:
            out.append("pg_tenure_decades")
        if self.gift_count_col in out and self.tenure_col in out:
            out.append("pg_loyalty_score")
        if self.age_col in out and self.tenure_col in out:
            out.append("pg_high_propensity")
        return np.array(out, dtype=object)

    def _more_tags(self):
        return {"X_types": ["2darray", "dataframe"]}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
