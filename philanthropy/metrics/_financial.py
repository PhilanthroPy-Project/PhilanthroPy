def donor_lifetime_value(
    average_donation: float, 
    lifespan_years: float, 
    discount_rate: float = 0.05, 
    retention_rate: float = None
) -> float:
    """
    Computes the Net Present Value (NPV) of a donor's future giving.

    If a retention_rate is provided, the expected lifespan is dynamically computed as:
        expected_lifespan = 1 / (1 - retention_rate)
    Otherwise, the provided lifespan_years is used directly.

    The mathematical formula for the NPV of an annuity is used:
    
    If discount_rate > 0:
        NPV = average_donation * (1 - (1 + discount_rate) ** (-lifespan)) / discount_rate
    If discount_rate == 0:
        NPV = average_donation * lifespan

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
        The annual retention rate of the donor (e.g., 0.80 for 80%). If provided, 
        this overrides lifespan_years by dynamically calculating expected lifespan.

    Returns
    -------
    float
        The calculated Net Present Value of the expected donor lifetime value.
    """
    if retention_rate is not None:
        if retention_rate >= 1.0:
            return float('inf')
        elif retention_rate < 0.0:
            raise ValueError("retention_rate cannot be negative.")
        
        lifespan = 1 / (1 - retention_rate)
    else:
        lifespan = lifespan_years

    if lifespan < 0:
        raise ValueError("lifespan_years cannot be negative.")

    if discount_rate == 0:
        return average_donation * lifespan
    elif discount_rate < 0:
        raise ValueError("discount_rate cannot be negative.")
    
    return average_donation * (1 - (1 + discount_rate) ** (-lifespan)) / discount_rate
