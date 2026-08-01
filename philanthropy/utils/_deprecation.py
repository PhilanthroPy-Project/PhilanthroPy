"""
philanthropy.utils._deprecation
===============================
The one deprecation mechanism in the package.

Not exported from ``philanthropy.utils``: this is internal plumbing for keeping
a renamed method alive for one published minor, not public API.
"""

from __future__ import annotations

import functools
import warnings


def deprecated_alias(new_name: str, removed_in: str):
    """Make the decorated method a warning wrapper around ``new_name``.

    Parameters
    ----------
    new_name : str
        Name of the replacement method on the same class.
    removed_in : str
        Version in which the alias goes away, e.g. ``"0.7.0"``.

    Examples
    --------
    >>> class Model:
    ...     def ask_ladder(self, X):
    ...         return X
    ...     @deprecated_alias("ask_ladder", removed_in="0.7.0")
    ...     def predict_ask_array(self, X): ...
    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     Model().predict_ask_array(1)
    ...     print(caught[0].category.__name__)
    1
    DeprecationWarning
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            warnings.warn(
                f"{type(self).__name__}.{func.__name__} is deprecated and will "
                f"be removed in {removed_in}; use .{new_name} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(self, new_name)(*args, **kwargs)

        wrapper.__doc__ = (
            f"Deprecated alias of :meth:`{new_name}`, removed in {removed_in}.\n\n"
            f"{func.__doc__ or ''}"
        )
        return wrapper

    return decorator
