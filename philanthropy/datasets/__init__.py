"""
philanthropy.datasets
=====================
Synthetic data generators and real reference datasets for donor analytics.
"""

from ._ciob import load_ciob_fundraising
from ._generator import generate_synthetic_donor_data, make_donor_dataset

__all__ = [
    "generate_synthetic_donor_data",
    "make_donor_dataset",
    "load_ciob_fundraising",
]
