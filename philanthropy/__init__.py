"""
PhilanthroPy
============
A scikit-learn compatible toolkit for predictive donor analytics
in the nonprofit sector.
"""

__version__ = "0.4.0"
__author__ = "Shivam Lalakiya"

from . import preprocessing, models, metrics, utils, datasets, ingest
from .ingest import constituent_events_to_features, read_constituent_events

__all__ = [
    "preprocessing",
    "models",
    "metrics",
    "utils",
    "datasets",
    "ingest",
    "constituent_events_to_features",
    "read_constituent_events",
]
