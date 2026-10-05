"""
philanthropy.datasets._karlan_list
==================================
Local-file reader for the Karlan and List (2007) matching-grant experiment.
"""

from __future__ import annotations

import pandas as pd

# Source variable -> returned column. Codes follow the authors' AERtables1-5.do.
_COLUMNS = {
    "gave": "gave",
    "amount": "amount",
    "treatment": "matched",
    "ratio": "match_ratio",
    "size": "match_cap",
    "ask": "ask_multiple",
    "freq": "prior_gifts",
    "HPA": "highest_previous_amount",
    "MRM2": "months_since_last_gift",
    "years": "years_since_first_gift",
    "female": "female",
    "couple": "couple",
    "red0": "red_state",
    "redcty": "red_county",
}

_INT_COLUMNS = ["gave", "matched", "match_ratio", "match_cap", "ask_multiple"]


def load_karlan_list(path: str) -> pd.DataFrame:
    """Load the Karlan and List matching-grant experiment from a local file.

    Karlan, D. and List, J. A. (2007), "Does Price Matter in Charitable
    Giving? Evidence from a Large-Scale Natural Field Experiment", *American
    Economic Review* 97(5): 1774-1793. About 50,000 prior donors to one US
    nonprofit were mailed a fundraising letter in 2005; two in three, chosen
    at random, were offered a matching grant, with the match ratio, the cap
    on the matching pool and the example ask amount each randomised.

    The replication package is openICPSR 113224 (doi:10.3886/E113224V1).
    Its ``LICENSE.txt`` puts the data under CC BY 4.0 and the code under
    BSD-3, copyright American Economic Association 2007, so anything built
    on it must credit the authors and the AEA. This reader never downloads
    the file: point it at ``AERtables1-5.dta`` from your own download.

    Rows with every variable missing (47 in the published file) are dropped;
    nothing else is filtered.

    Parameters
    ----------
    path : str
        Path to ``AERtables1-5.dta``.

    Returns
    -------
    pandas.DataFrame
        One row per donor:

        - ``gave`` (int): 1 if the donor gave in response to the letter.
        - ``amount`` (float): dollars given, 0 if not.
        - ``matched`` (int): 1 if the letter offered a matching grant.
        - ``match_ratio`` (int): 1, 2 or 3 for a 1:1, 2:1 or 3:1 match; 0
          without one.
        - ``match_cap`` (int): 1, 2 or 3 for a $25,000, $50,000 or $100,000
          matching pool; 4 if the letter stated no cap; 0 without a match.
        - ``ask_multiple`` (int): 1, 2 or 3 if the example amount was the
          donor's highest previous gift, 1.25 times it or 1.5 times it; 0
          without a match.
        - ``prior_gifts``, ``highest_previous_amount``,
          ``months_since_last_gift``, ``years_since_first_gift`` (float):
          giving history before the letter.
        - ``female``, ``couple`` (float): from the donor record, ``NaN``
          where unknown.
        - ``red_state``, ``red_county`` (float): 1 if the donor's state or
          county voted for Bush in 2004, ``NaN`` where unknown.

    Examples
    --------
    >>> from philanthropy.datasets import load_karlan_list
    >>> df = load_karlan_list("AERtables1-5.dta")  # doctest: +SKIP
    >>> round(df["gave"].mean(), 3)  # doctest: +SKIP
    0.021
    """
    raw = pd.read_stata(path, columns=list(_COLUMNS), convert_categoricals=False)
    df = raw.dropna(how="all").rename(columns=_COLUMNS).reset_index(drop=True)
    df = df.astype({c: "float64" for c in df.columns})
    df[_INT_COLUMNS] = df[_INT_COLUMNS].astype("int64")
    return df
