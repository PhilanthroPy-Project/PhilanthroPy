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
