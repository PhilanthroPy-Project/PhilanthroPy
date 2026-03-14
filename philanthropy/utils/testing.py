"""
philanthropy.utils.testing
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
