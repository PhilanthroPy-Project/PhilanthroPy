"""
philanthropy.datasets._donorschoose
====================================
Local-file reader for the DonorsChoose donation history (ICPSR 37898).
"""

from __future__ import annotations

import pandas as pd

_COLUMN_MAP = {
    "Donor ID": "donor_id",
    "Donation Amount": "gift_amount",
    "Donation Received Date": "gift_date",
}


def load_donorschoose(path: str) -> pd.DataFrame:
    """Load a user-downloaded DonorsChoose donation export into a gift table.

    DonorsChoose Open Data (ICPSR 37898, "DonorsChoose Open Data, United
    States, 2002-2019") is public-use microdata: one row per donation, with a
    stable per-donor identifier spanning the whole period. Its terms (ICPSR
    study page, https://www.icpsr.umich.edu/web/ICPSR/studies/37898) restrict
    use to statistical reporting and analysis. This reader never downloads or
    redistributes that file: point it at a CSV you have already obtained
    yourself under your own ICPSR account; the file itself is never vendored
    with this package.

    Expects the standard DonorsChoose donations-export columns (``Donor
    ID``, ``Donation Amount``, ``Donation Received Date``). If your extract
    names them differently, rename the columns before calling this function.

    Parameters
    ----------
    path : str
        Path to a donations CSV with the columns above.

    Returns
    -------
    pandas.DataFrame
        Columns ``donor_id`` (string), ``gift_date`` (datetime64),
        ``gift_amount`` (float64): the gift-table shape
        :func:`~philanthropy.ingest.build_upgrade_snapshots` and
        :class:`~philanthropy.preprocessing.RFMTransformer` expect.
    """
    raw = pd.read_csv(
        path, usecols=list(_COLUMN_MAP), dtype={"Donor ID": "string"}
    )
    gifts = raw.rename(columns=_COLUMN_MAP)
    gifts["gift_date"] = pd.to_datetime(gifts["gift_date"])
    gifts["gift_amount"] = gifts["gift_amount"].astype(float)
    return gifts[["donor_id", "gift_date", "gift_amount"]]
