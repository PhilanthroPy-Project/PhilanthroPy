"""
philanthropy.utils
==================
Generic helpers: model persistence and deprecated aliases.
"""

import warnings
from typing import Any

import pandas as pd

from ._persistence import save_model, load_model


def make_donor_dataset(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Deprecated alias for :func:`philanthropy.datasets.make_donor_dataset`."""
    warnings.warn(
        "philanthropy.utils.make_donor_dataset is deprecated and will be "
        "removed in 0.8.0; import make_donor_dataset from philanthropy.datasets "
        "instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from ..datasets import make_donor_dataset as _make_donor_dataset

    return _make_donor_dataset(*args, **kwargs)


__all__ = ["make_donor_dataset", "save_model", "load_model"]
