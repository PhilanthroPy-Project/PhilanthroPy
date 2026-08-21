from __future__ import annotations


def donor_lifetime_value(
    average_donation: float,
    lifespan_years: float,
    discount_rate: float = 0.05,
    retention_rate: float | None = None
) -> float:
    """
    Computes the Net Present Value (NPV) of a donor's future giving.

    Two modes, and they are different calculations rather than the same one with
    a substituted lifespan.

    **Fixed horizon** (``retention_rate=None``). ``lifespan_years`` is taken as
    certain and the result is the NPV of an ordinary annuity::

        discount_rate > 0:   NPV = m * (1 - (1 + d) ** -L) / d
        discount_rate == 0:  NPV = m * L

    **Geometric lifetime** (``retention_rate`` given). The donor gives once at
    the end of year 1, then survives each subsequent year with probability
    ``r``, so the lifetime is geometric on ``{1, 2, ...}`` with
    ``E[L] = 1 / (1 - r)``. The expected NPV is then::

        discount_rate > 0:   E[NPV] = m / (1 + d - r)
        discount_rate == 0:  E[NPV] = m / (1 - r)

    Note that this is **not** the annuity formula evaluated at ``E[L]``. The
    annuity is concave in ``L``, so by Jensen's inequality
    ``NPV(E[L]) >= E[NPV(L)]``, and substituting the expected lifespan into the
    annuity therefore overstates lifetime value in every case where ``d > 0``
    and ``0 < r < 1``. The error is one-signed and not small: at ``r = 0.8``,
    ``d = 0.05`` it is +8.2%, and at ``r = 0.9``, ``d = 0.10`` it is +22.9%.
    This function computed ``NPV(E[L])`` before version 0.7.0.

    The two modes agree where they should: at ``r = 0`` both give
    ``m / (1 + d)``, one gift discounted one year, and the ``d == 0`` branch is
    the same in both because a sum with no discounting is linear in ``L``.

    Parameters
    ----------
    average_donation : float
        The average annual donation amount.
    lifespan_years : float
        The fixed number of years the donor is expected to continue giving.
        Only used if retention_rate is None.
    discount_rate : float, default=0.05
        The discount rate used to compute the net present value of future gifts
        (e.g., 0.05 for 5%).
    retention_rate : float, default=None
        The annual retention rate of the donor (e.g., 0.80 for 80%). If
        provided, the geometric-lifetime expectation above is used and
        ``lifespan_years`` is ignored.

    Returns
    -------
    float
        The calculated Net Present Value of the expected donor lifetime value.
        ``inf`` when ``retention_rate == 1.0`` and ``discount_rate == 0``: a
        donor who never lapses, with no discounting, has unbounded value. With
        ``discount_rate > 0`` the same donor is a perpetuity worth ``m / d``.

    Raises
    ------
    ValueError
        If ``retention_rate``, ``lifespan_years``, or ``discount_rate`` is
        negative, or if ``retention_rate`` exceeds 1.

    Examples
    --------
    >>> round(donor_lifetime_value(1000.0, 5, discount_rate=0.05), 2)
    4329.48

    An 80% retention rate implies the same 5-year expected lifespan, but the
    expected NPV is lower than the 5-year annuity, not equal to it:

    >>> round(donor_lifetime_value(1000.0, 999, discount_rate=0.05,
    ...                            retention_rate=0.8), 2)
    4000.0
    """
    if discount_rate < 0:
        raise ValueError("discount_rate cannot be negative.")

    if retention_rate is not None:
        if retention_rate < 0.0:
            raise ValueError("retention_rate cannot be negative.")
        if retention_rate > 1.0:
            raise ValueError("retention_rate cannot exceed 1.")

        if discount_rate == 0:
            # Undiscounted: m * E[L] = m / (1 - r). Infinite at r == 1.
            if retention_rate == 1.0:
                return float("inf")
            return average_donation / (1.0 - retention_rate)

        # E[NPV] over a geometric lifetime. At r == 1 this is the perpetuity
        # m / d, which the expression already yields.
        return average_donation / (1.0 + discount_rate - retention_rate)

    if lifespan_years < 0:
        raise ValueError("lifespan_years cannot be negative.")

    if discount_rate == 0:
        return average_donation * lifespan_years

    return (
        average_donation
        * (1 - (1 + discount_rate) ** (-lifespan_years))
        / discount_rate
    )
