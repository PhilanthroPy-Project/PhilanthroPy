"""
philanthropy.utils._persistence
================================
Save / load PhilanthroPy model bundles.

A *bundle* is a self-describing dict: the fitted model plus the feature list,
target name, and the scikit-learn / PhilanthroPy versions it was trained with.
:func:`load_model` warns when those versions differ from the running
environment, which silently un-pickling with plain ``joblib.load`` never did.

.. warning::
    Bundles are pickle-based: :func:`load_model` executes arbitrary code during
    unpickling, exactly like scikit-learn's own persisted estimators. Only load
    bundles from a source you trust. See SECURITY.md.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional, Sequence, Union

import joblib

from .. import __version__

PathLike = Union[str, "os.PathLike[str]"]


def save_model(
    model: Any,
    path: PathLike,
    *,
    features: Optional[Sequence[str]] = None,
    target: Optional[str] = None,
) -> PathLike:
    """Persist ``model`` to ``path`` as a PhilanthroPy bundle.

    Parameters
    ----------
    model : fitted estimator
        Any PhilanthroPy or scikit-learn estimator.
    path : str or pathlib.Path
        Output path (``.joblib`` by convention).
    features : list of str, optional
        Ordered feature-column names the model was trained on.
    target : str, optional
        Name of the target column.

    Returns
    -------
    path
        The ``path`` it was written to (for chaining).
    """
    import sklearn

    bundle = {
        "model": model,
        "features": features,
        "target": target,
        "philanthropy_version": __version__,
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, path)
    return path


def load_model(path: PathLike) -> Dict[str, Any]:
    """Load a bundle written by :func:`save_model`.

    Warns (does not raise) when the stored PhilanthroPy or scikit-learn version
    differs from the running environment, since an estimator un-pickled under a
    different scikit-learn can silently misbehave.

    Parameters
    ----------
    path : str or pathlib.Path
        Bundle path.

    Returns
    -------
    dict
        The bundle: ``{"model", "features", "target", "philanthropy_version",
        "sklearn_version"}``.

    Raises
    ------
    ValueError
        If the file is not a PhilanthroPy bundle.
    """
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"{path} is not a PhilanthroPy model bundle.")
    _warn_on_version_mismatch(bundle)
    return bundle


def _warn_on_version_mismatch(bundle: Dict[str, Any]) -> None:
    import sklearn

    saved_ph = bundle.get("philanthropy_version")
    if saved_ph and saved_ph != __version__:
        warnings.warn(
            f"Model bundle was saved with philanthropy {saved_ph} but you are "
            f"running {__version__}; predictions may differ.",
            stacklevel=2,
        )
    saved_sk = bundle.get("sklearn_version")
    if saved_sk and saved_sk != sklearn.__version__:
        warnings.warn(
            f"Model bundle was saved with scikit-learn {saved_sk} but you are "
            f"running {sklearn.__version__}; unpickled estimators may misbehave.",
            stacklevel=2,
        )
