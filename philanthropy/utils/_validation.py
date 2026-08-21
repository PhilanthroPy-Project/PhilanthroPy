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
