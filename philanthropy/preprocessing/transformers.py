"""
philanthropy.preprocessing.transformers
========================================
"""

import pandas as pd
import numpy as np
from philanthropy.base import BasePhilanthropyTransformer


class CRMCleaner(BasePhilanthropyTransformer):
    """
    Standardises raw CRM exports.
    """

    def __init__(
        self,
        date_col: str = "gift_date",
        amount_col: str = "gift_amount",
        fiscal_year_start: int = 7,
    ):
        super().__init__(fiscal_year_start=fiscal_year_start)
        self.date_col = date_col
        self.amount_col = amount_col

    def fit(self, X: pd.DataFrame, y=None):
        self._validate_fiscal_year_start()
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        return X


class FiscalYearTransformer(BasePhilanthropyTransformer):
    """
    Appends Fiscal Year (FY) and Fiscal Quarter (FQ) columns.
    """

    def __init__(
        self,
        date_col: str = "gift_date",
        fiscal_year_start: int = 7,
    ):
        super().__init__(fiscal_year_start=fiscal_year_start)
        self.date_col = date_col

    def fit(self, X: pd.DataFrame, y=None):
        self._validate_fiscal_year_start()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        dates = pd.to_datetime(X[self.date_col])
        X["fiscal_year"] = dates.apply(
            lambda d: d.year + 1 if d.month >= self.fiscal_year_start else d.year
        )
        X["fiscal_quarter"] = np.nan
        return X
