"""
philanthropy.utils._validation
==============================
Shared validation logic for PhilanthroPy estimators.
"""

from typing import TypeVar

import pandas as pd

_PathT = TypeVar("_PathT")

def validate_fiscal_year_start(month: int) -> int:
    """
    Validate that the month is between 1 and 12.

    Parameters
    ----------
    month : int
        Starting month of the fiscal year.

    Returns
    -------
    month : int
        The validated month.

    Raises
    ------
    ValueError
        If month is not between 1 and 12.
    """
    if not (1 <= month <= 12):
        raise ValueError(
            f"`fiscal_year_start` must be between 1 and 12, got {month!r}."
        )
    return month


def fiscal_year_for(year: int, month: int, fiscal_year_start: int) -> int:
    """
    Return the fiscal-year label for a single calendar (year, month).

    A fiscal year is named by the calendar year it ends in, except when it
    starts in January: a January-start fiscal year is the same twelve months
    as the calendar year, so it keeps that year's own number rather than
    rolling forward.

    Parameters
    ----------
    year, month : int
        Calendar year and month (1-12) of the date being labelled.
    fiscal_year_start : int
        Month (1-12) the fiscal year starts in.

    Returns
    -------
    int
        The fiscal-year label.
    """
    if fiscal_year_start == 1:
        return year
    return year + 1 if month >= fiscal_year_start else year


def fiscal_year_and_quarter(
    dates: "pd.Series", fiscal_year_start: int
) -> "tuple[pd.Series, pd.Series]":
    """
    Vectorised fiscal year and quarter for a ``datetime64`` Series.

    Same labelling convention as :func:`fiscal_year_for`, applied elementwise
    without a per-row ``.apply``. Rows where ``dates`` is ``NaT`` come back as
    ``NaN`` in both outputs.

    Parameters
    ----------
    dates : pandas.Series of datetime64
        Parsed dates (``pd.to_datetime`` with ``errors="coerce"`` already
        applied by the caller).
    fiscal_year_start : int
        Month (1-12) the fiscal year starts in.

    Returns
    -------
    fiscal_year, fiscal_quarter : pandas.Series of float64
        NaN where ``dates`` is NaT.
    """
    year = dates.dt.year
    month = dates.dt.month
    if fiscal_year_start == 1:
        fiscal_year = year.astype(float)
    else:
        fiscal_year = year + (month >= fiscal_year_start).astype(float)
    missing = dates.isna()
    fiscal_year = fiscal_year.mask(missing).astype(float)
    quarter = (((month - fiscal_year_start) % 12) // 3 + 1).mask(missing).astype(float)
    return fiscal_year, quarter


_LOCAL_SCHEMES = ("", "file")


def ensure_local_path(path: _PathT, param_name: str = "path") -> _PathT:
    """
    Reject network-scheme URIs before a local file read.

    PhilanthroPy never transmits your data and fetches nothing on its own.
    ``pandas`` readers, however, will happily follow ``https://``, ``s3://`` or
    ``gs://`` URIs if handed one, so every user-supplied *data* path passes
    through this check first: the guarantee has to hold for the package's
    documented parameters, not just for its own logic.

    This is deliberately scoped to paths the caller supplies for their own donor
    or encounter data. It is not the package-wide network policy, which lives in
    ``tests/test_no_network.py`` and is enforced there by an import allowlist.

    Parameters
    ----------
    path : str or path-like
        The file path about to be opened.

    param_name : str
        Name of the parameter being validated, used in the error message.

    Returns
    -------
    path
        The unchanged path, with its original type.

    Raises
    ------
    ValueError
        If the path carries a non-local scheme.
    """
    from urllib.parse import urlparse

    scheme = urlparse(str(path)).scheme
    # urlparse treats the drive letter in an absolute Windows path such as
    # C:\\data\\gifts.csv as a one-character URI scheme. It is still a local
    # path, even when this check runs on a non-Windows host.
    if len(scheme) == 1 and scheme.isalpha():
        scheme = ""
    if scheme not in _LOCAL_SCHEMES:
        raise ValueError(
            f"`{param_name}` must be a local file path (no network reads), "
            f"got scheme {scheme!r} in {path!r}."
        )
    return path
