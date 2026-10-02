"""
philanthropy.datasets._donorschoose
====================================
Local-file reader for the DonorsChoose donation history (ICPSR 37898).
"""

from __future__ import annotations

import pandas as pd

_YES_NO_COLUMNS = {
    "PAYMENT_INCLUDED_CAMPAIGN_GIFT_1": "included_campaign_gift_card",
    "PAYMENT_INCLUDED_WEB_PURCHASED_1": "included_web_purchased_gift_card",
    "PAYMENT_WAS_MATCHED": "was_matched",
}

_COLUMNS = ["DONOR_ID", "AMOUNT", "CREATED_MONTH", "DONOR_TYPE", *_YES_NO_COLUMNS]


def load_donorschoose(path: str) -> pd.DataFrame:
    """Load a user-downloaded DonorsChoose Donations file (ICPSR 37898, DS0001).

    DonorsChoose Open Data, United States (ICPSR 37898,
    doi:10.3886/ICPSR37898.v1) is distributed under its own terms of use
    (``TermsOfUse.html`` in the study download): research or statistical
    purposes only, no investigation of specific research subjects, and no
    redistribution without ICPSR's written agreement. This reader never
    downloads, caches, or redistributes the file: point it at the delimited
    ``37898-0001-Data.tsv`` (or ``.dta``) you have already obtained yourself
    under your own ICPSR account. Never commit rows read by this function,
    or anything derived at the row level, to this repository or its docs;
    aggregate statistics only.

    ``CREATED_MONTH`` is recorded at **month resolution** (``YYYY-MM``), not
    a full date; this function parses it to the first day of that month, so
    any date-based feature computed from the result (recency, fiscal-year
    boundaries, ...) inherits that one-month granularity rather than the
    daily resolution the synthetic and KDD98 data provide.

    This reader applies no filtering: ``AMOUNT`` rows at or below zero exist
    in the source data and are passed through unchanged, every ``donor_type``
    ("citizen donor", "organization", "teacher") is returned, and a
    donation's own payment-source flags are returned as booleans rather than
    acted on. Decide and document your own filters at the point you use this
    data (e.g. the benchmark script), rather than relying on this function to
    have applied them.

    Parameters
    ----------
    path : str
        Path to the DS0001 Donations file, tab-separated (``.tsv``) or Stata
        (``.dta``, assumed to share the same column names as the ``.tsv``).

    Returns
    -------
    pandas.DataFrame
        Columns ``donor_id`` (string), ``gift_date`` (datetime64, first of
        the donation's month), ``gift_amount`` (float64), ``donor_type``
        (string), ``included_campaign_gift_card``,
        ``included_web_purchased_gift_card``, ``was_matched`` (bool):
        whether the donation's payment included a campaign-supplied gift
        card, a web-purchased gift card, or the payment was matched
        (codebook: "Payment matched"; matching source not specified).

    Notes
    -----
    Source: DonorsChoose Open Data, United States, 2002-2019 (ICPSR 37898),
    https://www.icpsr.umich.edu/web/ICPSR/studies/37898. Cite as
    doi:10.3886/ICPSR37898.v1.
    """
    if path.endswith(".dta"):
        raw = pd.read_stata(path, columns=_COLUMNS)
    else:
        raw = pd.read_csv(
            path, sep="\t", usecols=_COLUMNS, dtype={"DONOR_ID": "string"}
        )

    gifts = pd.DataFrame(
        {
            "donor_id": raw["DONOR_ID"],
            "gift_date": pd.to_datetime(raw["CREATED_MONTH"].str.strip() + "-01"),
            "gift_amount": raw["AMOUNT"].astype(float),
            "donor_type": raw["DONOR_TYPE"].str.strip(),
        }
    )
    for source, name in _YES_NO_COLUMNS.items():
        gifts[name] = raw[source].str.strip() == "Yes"
    return gifts
