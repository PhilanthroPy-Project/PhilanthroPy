"""
PhilanthroPy
============
A scikit-learn compatible toolkit for predictive donor analytics
in the nonprofit sector.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("philanthropy")
except PackageNotFoundError:  # pragma: no cover - source tree without an install
    __version__ = "0.0.0.dev0"

__author__ = "Shivam Lalakiya"

from . import (
    preprocessing,
    models,
    metrics,
    utils,
    datasets,
    ingest,
    inspection,
    model_selection,
    experimental,
    visualisation,
)
from .ingest import constituent_events_to_features, read_constituent_events

__all__ = [
    "preprocessing",
    "models",
    "metrics",
    "utils",
    "datasets",
    "ingest",
    "inspection",
    "model_selection",
    "experimental",
    "visualisation",
    "constituent_events_to_features",
    "read_constituent_events",
]
