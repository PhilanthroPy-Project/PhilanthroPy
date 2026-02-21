"""
philanthropy.base
=================
Core abstract base classes for all PhilanthroPy estimators.
"""

from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin


class BasePhilanthropyEstimator(BaseEstimator, metaclass=ABCMeta):
    """
    Base class for all PhilanthroPy estimators.
    """

    def __init__(self, fiscal_year_start: int = 7):
        """
        Parameters
        ----------
        fiscal_year_start : int, default=7
            The starting month (1–12) of the organisation's fiscal year.
        """
        self.fiscal_year_start = fiscal_year_start

    def _validate_fiscal_year_start(self):
        if not (1 <= self.fiscal_year_start <= 12):
            raise ValueError(
                f"`fiscal_year_start` must be between 1 and 12, "
                f"got {self.fiscal_year_start!r}."
            )


class BasePhilanthropyTransformer(BasePhilanthropyEstimator, TransformerMixin):
    """Base class for all PhilanthroPy transformers."""

    @abstractmethod
    def fit(self, X, y=None):
        """Fit the transformer to donor data X."""

    @abstractmethod
    def transform(self, X):
        """Apply the transformation to donor data X."""


class BasePhilanthropyClassifier(BasePhilanthropyEstimator, ClassifierMixin):
    """Base class for all PhilanthroPy classifiers."""

    @abstractmethod
    def fit(self, X, y):
        """Fit the classifier to labelled donor data (X, y)."""

    @abstractmethod
    def predict(self, X):
        """Predict binary donor labels for X."""

    @abstractmethod
    def predict_proba(self, X):
        """Return class-probability estimates for X."""
