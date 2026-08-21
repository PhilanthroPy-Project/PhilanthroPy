import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted

class RFMTransformer(TransformerMixin, BaseEstimator):
    """
    Transforms transaction logs into Recency, Frequency, and Monetary (RFM) features.

    This is a **pre-pipeline aggregation step, not a pipeline member.** It takes
    one row per gift and returns one row per donor, so the sample count changes
    across ``transform``. Putting it inside a
    :class:`~sklearn.pipeline.Pipeline` ahead of an estimator raises
    ``ValueError: Found input variables with inconsistent numbers of samples``,
    because ``y`` is still gift-shaped. Run it first, then build the pipeline on
    its donor-level output. It is exempt from the ``check_estimator`` battery
    for the same reason, with hand-written coverage in
    ``tests/test_sklearn_compliance.py``.

    Parameters
    ----------
    reference_date : str or datetime-like, default=None
        The date used as the reference point to calculate recency.
        If None, the maximum gift_date in the dataframe is used.
    agg_func : str or callable, default='sum'
        The aggregation function to calculate the monetary value. 
        Typical values are 'sum' (cumulative) or 'mean' (average).
    include_tenure : bool, default=False
        Emit a fifth column, ``tenure``: days from the donor's *first* gift to
        the frozen reference date. Recency-frequency-monetary alone cannot feed
        a buy-till-you-die model, which needs the observation window T as well
        [Fader, Hardie and Lee 2005]; ``tenure`` is that T. Defaults to False so
        the output shape does not change under existing callers, and will become
        the default in the next major release.
    """
    def __init__(self, reference_date=None, agg_func='sum', include_tenure=False):
        self.reference_date = reference_date
        self.agg_func = agg_func
        self.include_tenure = include_tenure

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

        # Freeze the recency reference date from TRAINING data (leakage-safety
        # contract: fitted statistics are computed in fit and frozen before
        # transform). Mirrors EncounterRecencyTransformer.reference_date_.
        if self.reference_date is not None:
            self.reference_date_ = pd.to_datetime(self.reference_date)
        else:
            X_df = X if hasattr(X, "columns") else pd.DataFrame(
                X, columns=self.feature_names_in_
            )
            self.reference_date_ = pd.to_datetime(X_df["gift_date"]).max()
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

        # Use the reference date frozen in fit, never the transform batch's
        # max, which would make recency depend on which rows share the batch.
        ref_date = self.reference_date_

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

        if self.include_tenure:
            # T for a buy-till-you-die model: the donor's observation window,
            # measured from the first gift to the same frozen reference date
            # recency uses, so the two are on one clock.
            first_gift = grouped['gift_date'].min()
            rfm_df['tenure'] = (ref_date - first_gift).dt.days.values

        return rfm_df
        
    def _validate_input(self, X):
        cols = X.columns if hasattr(X, "columns") else self.feature_names_in_
        required_cols = {"donor_id", "gift_date", "gift_amount"}
        if not required_cols.issubset(cols):
            raise ValueError(f"X must contain columns: {required_cols}")

    def get_feature_names_out(self, input_features=None):
        """Return the donor identifier and generated RFM feature names.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Ignored because the output columns are fixed by
            ``include_tenure``, not by the input.

        Returns
        -------
        feature_names_out : ndarray of str
            ``["donor_id", "recency", "frequency", "monetary"]``, plus
            ``"tenure"`` when ``include_tenure=True``.

        Raises
        ------
        NotFittedError
            If the transformer has not been fitted.
        """
        check_is_fitted(self)
        names = ['donor_id', 'recency', 'frequency', 'monetary']
        if self.include_tenure:
            names.append('tenure')
        return np.array(names, dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        return tags
