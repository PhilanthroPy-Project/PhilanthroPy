"""
philanthropy.utils._testing
===========================
"""

import numpy as np
import pandas as pd
from typing import Optional


def make_donor_dataset(
    n_donors: int = 200,
    fiscal_year_start: int = 7,
    start_year: int = 2018,
    end_year: int = 2024,
    lapse_rate: float = 0.25,
    major_gift_threshold: float = 10_000.0,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """Generate a synthetic, seeded gift-level donor dataset.

    Parameters
    ----------
    n_donors : int, default=200
        Number of synthetic donors to generate.
    fiscal_year_start : int, default=7
        Reserved. Accepted for signature stability but not currently used
        when generating gift dates.
    start_year : int, default=2018
        Earliest calendar year used when sampling gift dates.
    end_year : int, default=2024
        Latest calendar year used when sampling gift dates.
    lapse_rate : float, default=0.25
        Reserved. Accepted for signature stability but not currently used
        when generating gift dates.
    major_gift_threshold : float, default=10_000.0
        Gift amount (inclusive) that marks a row as a major gift.
    random_state : int or None, default=42
        Seed for the NumPy random-number generator. Pass an integer to
        obtain a reproducible dataset; ``None`` draws a fresh seed every call.

    Returns
    -------
    df : pd.DataFrame
        A gift-level DataFrame, not a donor-level DataFrame: each donor
        contributes 1-5 gift rows, so ``len(df) > n_donors``. Rows are sorted
        by ``gift_date`` and the index is reset. Columns:

        ``donor_id`` : str
            Zero-padded donor id, e.g. ``D00001``.
        ``gift_date`` : datetime64[ns]
            Sampled between ``start_year`` and ``end_year``.
        ``gift_amount`` : float
            Log-normal gift amount, rounded to two decimal places.
        ``appeal_code`` : str
            One of ``ANNUAL``, ``MAJOR``, ``PLANNED``, ``ONLINE``, or ``EVENT``.
        ``is_major_gift`` : bool
            True when ``gift_amount >= major_gift_threshold``.

    Examples
    --------
    >>> from philanthropy.utils import make_donor_dataset
    >>> df = make_donor_dataset(n_donors=5, random_state=0)
    >>> df.shape
    (15, 5)
    >>> list(df.columns)
    ['donor_id', 'gift_date', 'gift_amount', 'appeal_code', 'is_major_gift']
    >>> df['donor_id'].nunique()
    5
    """
    rng = np.random.default_rng(random_state)
    donor_ids = [f"D{str(i).zfill(5)}" for i in range(1, n_donors + 1)]
    records = []
    for donor_id in donor_ids:
        n_gifts = rng.integers(1, 6)
        for _ in range(n_gifts):
            year = rng.integers(start_year, end_year + 1)
            month = rng.integers(1, 13)
            day = rng.integers(1, 28)
            gift_date = pd.Timestamp(year=int(year), month=int(month), day=int(day))
            gift_amount = float(rng.lognormal(mean=5.5, sigma=1.2))
            gift_amount = round(gift_amount, 2)
            appeal_code = rng.choice(["ANNUAL", "MAJOR", "PLANNED", "ONLINE", "EVENT"])
            records.append(
                {
                    "donor_id": donor_id,
                    "gift_date": gift_date,
                    "gift_amount": gift_amount,
                    "appeal_code": appeal_code,
                }
            )
    df = pd.DataFrame(records).sort_values("gift_date").reset_index(drop=True)
    df["is_major_gift"] = df["gift_amount"] >= major_gift_threshold
    return df
