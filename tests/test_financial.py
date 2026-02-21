import pytest
import math
from philanthropy.metrics import donor_lifetime_value

def test_donor_lifetime_value_basic():
    # NPV = 100 * (1 - (1.05)^-5) / 0.05
    # 1.05^-5 = 0.783526166...
    # 1 - 0.783526166 = 0.216473833...
    # / 0.05 = 4.329476...
    # * 100 = 432.9476
    ltv = donor_lifetime_value(average_donation=100.0, lifespan_years=5, discount_rate=0.05)
    assert math.isclose(ltv, 432.9476, rel_tol=1e-4)

def test_donor_lifetime_value_zero_discount():
    # If discount rate is 0, it's just average_donation * lifespan
    ltv = donor_lifetime_value(average_donation=100.0, lifespan_years=5, discount_rate=0.0)
    assert ltv == 500.0

def test_donor_lifetime_value_with_retention():
    # 80% retention -> expected lifespan = 1 / (1 - 0.8) = 5 years
    # 0 discount rate -> ltv = 100 * 5 = 500
    ltv = donor_lifetime_value(average_donation=100.0, lifespan_years=10, discount_rate=0.0, retention_rate=0.8)
    assert math.isclose(ltv, 500.0, rel_tol=1e-4)

def test_donor_lifetime_value_100_percent_retention():
    ltv = donor_lifetime_value(average_donation=100.0, lifespan_years=5, discount_rate=0.05, retention_rate=1.0)
    assert math.isinf(ltv)

def test_donor_lifetime_value_negative_validation():
    with pytest.raises(ValueError, match="cannot be negative"):
        donor_lifetime_value(average_donation=100.0, lifespan_years=-1)
    
    with pytest.raises(ValueError, match="cannot be negative"):
        donor_lifetime_value(average_donation=100.0, lifespan_years=5, discount_rate=-0.05)
        
    with pytest.raises(ValueError, match="cannot be negative"):
        donor_lifetime_value(average_donation=100.0, lifespan_years=5, retention_rate=-0.2)
