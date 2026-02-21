import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class RFMTransformer(BaseEstimator, TransformerMixin):
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
        return self

    def transform(self, X):
        """
        Transforms the transaction logs into RFM features.
        """
        self._validate_input(X)
        
        X = X.copy()
        X['gift_date'] = pd.to_datetime(X['gift_date'])
        
        if self.reference_date is not None:
            ref_date = pd.to_datetime(self.reference_date)
        else:
            ref_date = X['gift_date'].max()
            
        grouped = X.groupby('donor_id')
        
        # Recency: Days since the last gift relative to reference_date
        last_gift = grouped['gift_date'].max()
        recency = (ref_date - last_gift).dt.days
        
        # Frequency: Total number of gifts
        frequency = grouped['gift_date'].count()
        
        # Monetary: Average or cumulative gift amount depending on agg_func
        monetary = grouped['gift_amount'].agg(self.agg_func)
        
        rfm_df = pd.DataFrame({
            'recency': recency,
            'frequency': frequency,
            'monetary': monetary
        }).reset_index()
        
        return rfm_df
        
    def _validate_input(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        required_cols = {"donor_id", "gift_date", "gift_amount"}
        if not required_cols.issubset(X.columns):
            raise ValueError(f"X must contain columns: {required_cols}")
