"""
philanthropy.datasets
=====================
Synthetic data generators and real reference datasets for donor analytics.
"""

from ._ciob import load_ciob_fundraising
from ._generator import generate_synthetic_donor_data

__all__ = ["generate_synthetic_donor_data", "load_ciob_fundraising"]
