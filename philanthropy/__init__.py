"""
PhilanthroPy
============
A scikit-learn compatible toolkit for predictive donor analytics
in the nonprofit sector.

Generative AI disclosure
------------------------
The design of this package is the author's, who made every core design
decision and reviewed, edited, and validated all AI-assisted output.
Within constraints set in advance, AI assistance (Claude Code) was used
during development. The scope is package-wide rather than specific to any
one module, so the disclosure lives here rather than being stamped on
every module. See the "Generative AI disclosure" section of README.md for
how the tools were used and what human review was performed.
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
from .ingest import (
    civicrm_contributions_to_features,
    constituent_events_to_features,
    read_civicrm_contributions,
    read_constituent_events,
)

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
    "civicrm_contributions_to_features",
    "constituent_events_to_features",
    "read_civicrm_contributions",
    "read_constituent_events",
]
