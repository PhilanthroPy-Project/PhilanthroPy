"""
philanthropy.datasets._panel
============================
A seeded multi-year donor panel: gift-level rows, not one row per donor.

``generate_synthetic_donor_data`` returns one cross-sectional row per donor,
which is enough to fit a classifier and nothing else. It cannot demonstrate
``RFMTransformer`` (needs a gift log), ``FiscalYearGroupedSplitter`` (needs
repeated donor-years), an ``as_of`` cutoff (needs something to cut off), or the
grateful-patient transformers (need encounters). Those are the ideas this
library exists for, and until now the only generator that could show them lived
privately inside ``scripts/leakage_experiment.py``.

The data-generating process is deliberately the same one the published leakage
experiment used, draw for draw: a stable per-donor log-odds ``theta`` plus a
sector-wide drift that makes later years genuinely harder. Persistence is the
point, because persistence is what any leak has to exploit; drift is the point,
because without it a cross-validation scheme could look good purely by ignoring
time.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

# Kept short and boring. These are labels for grouping, not a taxonomy, and a
# longer list would only invite someone to read meaning into synthetic strings.
_APPEALS = ("annual_fund", "spring_gala", "year_end_mail", "direct_ask")
_SERVICE_LINES = ("cardiology", "oncology", "orthopedics", "neurology")
_EMPLOYERS = (
    "Northside Health System",
    "Riverbend University",
    "Latimer & Cole LLP",
    "Kestrel Capital",
    "",  # unemployed / unknown, which every real CRM has plenty of
)


def make_donor_panel(
    n_donors: int = 3000,
    n_years: int = 7,
    start_fiscal_year: int = 2018,
    include_encounters: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Generate a seeded multi-year donor panel with gift-level rows.

    Unlike :func:`generate_synthetic_donor_data`, which returns one
    already-aggregated row per donor, this returns the *raw* tables a
    fundraising shop actually exports: a gift log, a donor table, and
    optionally a clinical-encounter table. Aggregating them is left to the
    caller, on purpose, because *when* you aggregate is the whole subject of
    :doc:`the leakage tutorial </tutorials/avoiding_temporal_data_leakage>`.

    Fiscal years run 1 July to 30 June and are labelled by the calendar year in
    which they end, so fiscal year 2019 spans 2018-07-01 to 2019-06-30. A donor
    gives at most once per fiscal year.

    Parameters
    ----------
    n_donors : int, default=3000
        Number of donors in the panel. Every donor appears in ``donors``;
        donors who never gave contribute no rows to ``gifts``.
    n_years : int, default=7
        Number of consecutive fiscal years. Deriving a "gave in the following
        year" label costs the last one, so the default yields six usable panel
        years.
    start_fiscal_year : int, default=2018
        Label of the first fiscal year.
    include_encounters : bool, default=False
        Also return an ``"encounters"`` table for the grateful-patient
        transformers. Off by default: most callers are not an academic medical
        center, and an unused encounter table invites the mistake of treating
        synthetic clinical rows as if they meant something.
    random_state : int or None, default=None
        Seed for the NumPy generator. Pass an integer for a reproducible
        panel; ``None`` draws a fresh seed on every call.

    Returns
    -------
    panel : dict of str to pandas.DataFrame
        ``"gifts"``
            One row per gift: ``donor_id`` (int), ``gift_date``
            (datetime64[ns]), ``gift_amount`` (float), ``fiscal_year`` (int),
            ``appeal`` (str). Sorted by ``gift_date``. Column names match what
            :class:`~philanthropy.preprocessing.RFMTransformer` requires.
        ``"donors"``
            One row per donor: ``donor_id`` (int), ``first_gift_fy`` (float,
            NaN for donors who never gave), ``wealth_estimate`` (float, ~30%
            NaN by design, for
            :class:`~philanthropy.preprocessing.WealthScreeningImputer`),
            ``employer`` (str, sometimes empty).
        ``"encounters"``, only when ``include_encounters=True``
            Zero or more rows per donor: ``donor_id`` (int), ``admit_date``
            and ``discharge_date`` (datetime64[ns]), ``service_line`` (str).

    Notes
    -----
    **There is no label column, deliberately.** A label is a claim about a
    point in time, and shipping one pre-computed would hand every user the
    exact mistake this package exists to prevent. Derive it, as of the year you
    are scoring:

    >>> from philanthropy.datasets import make_donor_panel
    >>> panel = make_donor_panel(n_donors=200, random_state=0)
    >>> gifts = panel["gifts"]
    >>> gave = set(zip(gifts["donor_id"], gifts["fiscal_year"]))
    >>> label = int((7, 2020) in gave)   # did donor 7 give in FY2020?

    Every number here is invented. The panel establishes mechanisms and shapes,
    never magnitudes you should quote for your own program; for magnitudes
    measured on a real donor file, see
    :doc:`/explanation/real_data_replication`.

    Examples
    --------
    >>> from philanthropy.datasets import make_donor_panel
    >>> panel = make_donor_panel(n_donors=500, random_state=42)
    >>> sorted(panel)
    ['donors', 'gifts']
    >>> list(panel["gifts"].columns)
    ['donor_id', 'gift_date', 'gift_amount', 'fiscal_year', 'appeal']
    >>> int(panel["gifts"]["fiscal_year"].min()), int(panel["gifts"]["fiscal_year"].max())
    (2018, 2024)
    >>> len(panel["donors"])
    500

    Straight into RFM features, which the cross-sectional generator cannot do:

    >>> from philanthropy.preprocessing import RFMTransformer
    >>> rfm = RFMTransformer().fit_transform(panel["gifts"])
    >>> list(rfm.columns)
    ['donor_id', 'recency', 'frequency', 'monetary']

    With encounters, for the grateful-patient path:

    >>> panel = make_donor_panel(
    ...     n_donors=500, include_encounters=True, random_state=42
    ... )
    >>> list(panel["encounters"].columns)
    ['donor_id', 'admit_date', 'discharge_date', 'service_line']
    """
    if n_donors < 1:
        raise ValueError(f"n_donors must be at least 1; got {n_donors}.")
    if n_years < 2:
        raise ValueError(
            "n_years must be at least 2, because a 'gave in the following "
            f"year' label needs a following year; got {n_years}."
        )

    rng = np.random.default_rng(random_state)

    # --- The giving process. ------------------------------------------------
    # These draws, in this order, are the ones scripts/leakage_experiment.py
    # has always made. Anything added below must stay below, or the published
    # benchmark numbers stop reproducing.
    theta = rng.normal(0, 1.2, n_donors)
    drift = np.linspace(0.3, -0.3, n_years)
    gave = np.zeros((n_donors, n_years), dtype=bool)
    amount = np.zeros((n_donors, n_years))
    for j in range(n_years):
        p = 1.0 / (1.0 + np.exp(-(theta + drift[j])))
        gave[:, j] = rng.random(n_donors) < p
        amount[:, j] = np.where(
            gave[:, j], rng.lognormal(6 + 0.35 * theta, 0.8), 0.0
        )

    # --- Everything below is presentation, and draws after the process. -----
    fiscal_years = start_fiscal_year + np.arange(n_years)
    day_offset = rng.integers(0, 365, size=(n_donors, n_years))
    appeal_idx = rng.integers(0, len(_APPEALS), size=(n_donors, n_years))

    donor_ix, year_ix = np.nonzero(gave)
    fy = fiscal_years[year_ix]
    # Fiscal year N opens on 1 July of year N-1.
    fy_start = pd.to_datetime([f"{year - 1}-07-01" for year in fy])
    gift_date = fy_start + pd.to_timedelta(day_offset[donor_ix, year_ix], unit="D")

    gifts = pd.DataFrame(
        {
            "donor_id": donor_ix,
            "gift_date": gift_date,
            # Not rounded to cents, tempting as that is. scripts/leakage_
            # experiment.py aggregates these amounts, and rounding moved the
            # published min-max ranges by 0.001 AUC. A cosmetic decimal is not
            # worth invalidating a number that is already in the docs and the
            # paper. Round at the point of display instead.
            "gift_amount": amount[donor_ix, year_ix],
            "fiscal_year": fy,
            "appeal": [_APPEALS[i] for i in appeal_idx[donor_ix, year_ix]],
        }
    ).sort_values("gift_date", ignore_index=True)

    # NaN at a realistic rate, because a wealth screen that came back for every
    # record is not a wealth screen anyone has ever received.
    wealth = np.exp(11.0 + 0.6 * theta + rng.normal(0, 0.5, n_donors))
    wealth[rng.random(n_donors) < 0.30] = np.nan

    ever_gave = gave.any(axis=1)
    first_gift_fy = np.full(n_donors, np.nan)
    first_gift_fy[ever_gave] = fiscal_years[gave[ever_gave].argmax(axis=1)]

    donors = pd.DataFrame(
        {
            "donor_id": np.arange(n_donors),
            "first_gift_fy": first_gift_fy,
            "wealth_estimate": np.round(wealth, 2),
            "employer": rng.choice(_EMPLOYERS, size=n_donors),
        }
    )

    panel = {"gifts": gifts, "donors": donors}
    if include_encounters:
        panel["encounters"] = _encounters(rng, n_donors, fiscal_years)
    return panel


def _encounters(
    rng: np.random.Generator,
    n_donors: int,
    fiscal_years: np.ndarray,
) -> pd.DataFrame:
    """Admissions for the subset of donors who were ever patients.

    Encounter dates are drawn independently of giving. A generator that made
    grateful-patient features predictive by construction would be a very
    convincing demo of nothing.
    """
    n_encounters = rng.poisson(0.8, n_donors)
    n_encounters[rng.random(n_donors) < 0.55] = 0   # never a patient here

    donor_id = np.repeat(np.arange(n_donors), n_encounters)
    total = len(donor_id)

    window_start = pd.Timestamp(f"{fiscal_years[0] - 1}-07-01")
    window_days = 365 * len(fiscal_years)
    admit = window_start + pd.to_timedelta(
        rng.integers(0, window_days, size=total), unit="D"
    )
    stay = rng.integers(1, 9, size=total)   # 1 to 8 nights

    return pd.DataFrame(
        {
            "donor_id": donor_id,
            "admit_date": admit,
            "discharge_date": admit + pd.to_timedelta(stay, unit="D"),
            "service_line": rng.choice(_SERVICE_LINES, size=total),
        }
    ).sort_values(["donor_id", "admit_date"], ignore_index=True)
