"""
philanthropy.datasets._ciob
===========================
Loader for the real NYC "Official Fundraising by City Agencies" dataset.
"""

from __future__ import annotations

from importlib.resources import files

import pandas as pd


def load_ciob_fundraising() -> pd.DataFrame:
    """Load the NYC CIOB "Official Fundraising by City Agencies" registry.

    A real open-government dataset: every not-for-profit that a New York City
    agency reported soliciting donations for, disclosed under the Conflicts of
    Interest Board's (CIOB) mandate. It is an **affiliation registry** — one row
    per ``(year, agency, nonprofit)`` link — **not** donor-level giving data.

    It carries no gift amounts, donor records, or engagement labels, so it does
    **not** support the RFM / propensity modelling in the rest of this library;
    use :func:`generate_synthetic_donor_data` for that. It is included for
    honest, reproducible analysis of agency ↔ nonprofit fundraising
    relationships (e.g. most-solicited nonprofits, agency fundraising breadth).

    Returns
    -------
    pandas.DataFrame of shape (2336, 3)
        Columns:

        ``year`` : int
            Calendar year of the reported fundraising (2019–2024).
        ``agency`` : string
            NYC agency that solicited the donation.
        ``name_of_not_for_profit`` : string
            Beneficiary not-for-profit organisation.

    Notes
    -----
    Source: NYC Open Data, dataset ``basd-2jwn`` (Conflicts of Interest Board),
    https://data.cityofnewyork.us/City-Government/Official-Fundraising-by-City-Agencies/basd-2jwn.
    Distributed under the NYC Open Data Terms of Use (free public use). The CSV
    is vendored with the package, so this loader needs no network access.
    """
    csv = files("philanthropy.datasets").joinpath(
        "data/ciob_official_fundraising.csv"
    )
    with csv.open("r", encoding="utf-8") as fh:
        return pd.read_csv(
            fh,
            dtype={
                "year": "int64",
                "agency": "string",
                "name_of_not_for_profit": "string",
            },
        )
