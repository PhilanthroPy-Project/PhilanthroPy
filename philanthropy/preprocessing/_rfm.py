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
        self._validate_input(X)
        X = validate_data(self, X, reset=True)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns.tolist(), dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """
        Transforms the transaction logs into RFM features.
        """
        check_is_fitted(self)
        self._validate_input(X)
        X = validate_data(self, X, reset=False)
        
        X_out = X.copy() if hasattr(X, "columns") else pd.DataFrame(X)
        X_out['gift_date'] = pd.to_datetime(X_out['gift_date'])
        
        if self.reference_date is not None:
            ref_date = pd.to_datetime(self.reference_date)
        else:
            ref_date = X_out['gift_date'].max()
            
        grouped = X_out.groupby('donor_id')
        
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
        if not hasattr(X, "columns"):
            raise TypeError("X must be a pandas DataFrame")
        required_cols = {"donor_id", "gift_date", "gift_amount"}
        if not required_cols.issubset(X.columns):
            raise ValueError(f"X must contain columns: {required_cols}")

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.array(['donor_id', 'recency', 'frequency', 'monetary'], dtype=object)

    def _more_tags(self):
        return {"X_types": ["2darray", "dataframe"]}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
