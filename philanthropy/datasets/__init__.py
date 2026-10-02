"""
philanthropy.datasets
=====================
Synthetic data generators and real reference datasets for donor analytics.
"""

from ._ciob import load_ciob_fundraising
from ._donorschoose import load_donorschoose
from ._generator import generate_synthetic_donor_data, make_donor_dataset
from ._kdd98 import fetch_kdd98_donors, fetch_kdd98_val_donors
from ._panel import make_donor_panel
from ._psid import load_psid_philanthropy

__all__ = [
    "fetch_kdd98_donors",
    "fetch_kdd98_val_donors",
    "generate_synthetic_donor_data",
    "make_donor_dataset",
    "make_donor_panel",
    "load_ciob_fundraising",
    "load_donorschoose",
    "load_psid_philanthropy",
]
