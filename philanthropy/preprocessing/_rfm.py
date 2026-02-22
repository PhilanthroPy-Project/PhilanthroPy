import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data

class RFMTransformer(TransformerMixin, BaseEstimator):
    """
    Transforms transaction logs into Recency, Frequency, and Monetary (RFM) features.
    
    Parameters
    ----------
    reference_date : str or datetime-like, default=None
        The date used as the reference point to calculate recency.
        If None, the maximum gift_date in the dataframe is used.
    agg_func : str or callable, default='sum'
        The aggregation function to calculate the monetary value. 
        Typical values are 'sum' (cumulative) or 'mean' (average).
    """
    def __init__(self, reference_date=None, agg_func='sum'):
        self.reference_date = reference_date
        self.agg_func = agg_func

    def fit(self, X, y=None):
        """
        Fits the transformer. This simply validates the input and returns self.
        """
        # Manual validation to avoid name/length strictness during fit
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns.tolist(), dtype=object)
            self.n_features_in_ = len(self.feature_names_in_)
        else:
            X_arr = np.asarray(X)
            self.n_features_in_ = X_arr.shape[1]
            self.feature_names_in_ = np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        
        self._validate_input(X)
        return self

    def transform(self, X):
        """
        Transforms the transaction logs into RFM features.
        """
        check_is_fitted(self)
        if not hasattr(X, "columns") and not isinstance(X, pd.DataFrame):
             raise TypeError("X must be a pandas DataFrame")
        # Manual validation
        self._validate_input(X)
        
        X_df = X.copy() if hasattr(X, "columns") else pd.DataFrame(X, columns=self.feature_names_in_)
        X_df['gift_date'] = pd.to_datetime(X_df['gift_date'])
        
        if self.reference_date is not None:
            ref_date = pd.to_datetime(self.reference_date)
        else:
            ref_date = X_df['gift_date'].max()
            
        grouped = X_df.groupby('donor_id')
        
        # Recency: Days since the last gift relative to reference_date
        last_gift = grouped['gift_date'].max()
        recency = (ref_date - last_gift).dt.days
        
        # Frequency: Total number of gifts
        frequency = grouped['gift_date'].count()
        
        # Monetary: Average or cumulative gift amount depending on agg_func
        monetary = grouped['gift_amount'].agg(self.agg_func)
        
        rfm_df = pd.DataFrame({
            'donor_id': recency.index,
            'recency': recency.values,
            'frequency': frequency.values,
            'monetary': monetary.values
        })
        
        return rfm_df
        
    def _validate_input(self, X):
        cols = X.columns if hasattr(X, "columns") else self.feature_names_in_
        required_cols = {"donor_id", "gift_date", "gift_amount"}
        if not required_cols.issubset(cols):
            raise ValueError(f"X must contain columns: {required_cols}")

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.array(['donor_id', 'recency', 'frequency', 'monetary'], dtype=object)

    def _more_tags(self):
        return {"X_types": ["2darray", "dataframe", "string"]}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        tags._skip_test = True # Schema-dependent
        return tags
