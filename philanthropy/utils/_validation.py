"""
philanthropy.utils._validation
==============================
Shared validation logic for PhilanthroPy estimators.
"""

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


_LOCAL_SCHEMES = ("", "file")


def ensure_local_path(path, param_name: str = "path") -> str:
    """
    Reject network-scheme URIs before a local file read.

    PhilanthroPy makes no network calls of any kind; pandas readers would
    happily follow ``https://``, ``s3://`` or ``gs://`` URIs if handed one,
    so every user-supplied path passes through this check first.

    Parameters
    ----------
    path : str
        The file path about to be opened.

    param_name : str
        Name of the parameter being validated, used in the error message.

    Returns
    -------
    path : str
        The unchanged path.

    Raises
    ------
    ValueError
        If the path carries a non-local scheme.
    """
    from urllib.parse import urlparse

    scheme = urlparse(str(path)).scheme
    if scheme not in _LOCAL_SCHEMES:
        raise ValueError(
            f"`{param_name}` must be a local file path (no network reads), "
            f"got scheme {scheme!r} in {path!r}."
        )
    return path
