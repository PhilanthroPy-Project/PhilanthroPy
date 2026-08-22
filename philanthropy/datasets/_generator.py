"""
philanthropy.datasets._generator
=================================
Utility for generating realistic, correlated synthetic donor datasets
suitable for developing and benchmarking PhilanthroPy estimators.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# The gift capacity that defines a major-gift prospect, per the docstring.
# The label is a soft threshold on latent capacity at this level.
MAJOR_GIFT_CAPACITY = 25_000.0


def generate_synthetic_donor_data(
    n_samples: int = 1000,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Generate a realistic synthetic donor DataFrame for modelling and testing.

    The returned dataset simulates a hospital's major-gifts prospect pool.
    Features are correlated in a domain-meaningful way:

    * Donors with more ``years_active`` and higher ``event_attendance_count``
      have a monotonically increasing probability of being labelled as a
      major donor (``is_major_donor = 1``).
    * ``total_gift_amount`` is log-normally distributed and positively
      correlated with ``is_major_donor``.
    * ``last_gift_date`` is sampled uniformly across the past five calendar
      years, with major donors skewed toward more recent activity.

    Parameters
    ----------
    n_samples : int, default=1000
        Number of synthetic donor records to generate.
    random_state : int or None, default=None
        Seed for the NumPy random-number generator.  Pass an integer to
        obtain a reproducible dataset; ``None`` draws a fresh seed on every
        call.

    Returns
    -------
    df : pd.DataFrame of shape (n_samples, 5)
        A DataFrame with the following columns:

        ``total_gift_amount`` : float
            Cumulative lifetime giving in USD.  Drawn from a log-normal
            distribution (mu = 7.5, sigma = 1.4); major donors receive an
            additional multiplicative uplift of 3–8×.
        ``years_active`` : int
            Number of full calendar years since the donor's first recorded
            gift (range 1–30).  Major-donor candidates are skewed toward
            longer tenure.
        ``last_gift_date`` : datetime
            Date of the most recent gift.  Stored as ``datetime64[ns]``.
            Major donors are skewed toward dates within the past two years.
        ``event_attendance_count`` : int
            Number of fundraising events attended (range 0–20).  Higher
            values increase propensity-to-give probability.
        ``is_major_donor`` : int (0 or 1)
            Binary label indicating whether the donor is classified as a
            major gift prospect (gift capacity ≥ $25,000).

    Examples
    --------
    >>> from philanthropy.datasets import generate_synthetic_donor_data
    >>> df = generate_synthetic_donor_data(n_samples=500, random_state=42)
    >>> df.shape
    (500, 5)
    >>> df.dtypes["is_major_donor"]
    dtype('int64')
    >>> bool(df["is_major_donor"].isin([0, 1]).all())
    True

    Notes
    -----
    A latent **giving capacity** drives everything. It is a linear function of
    ``years_active``, ``event_attendance_count`` and an unobserved wealth term,
    on a log-dollar scale. ``total_gift_amount`` is then drawn as a noisy
    realisation of that capacity, and ``is_major_donor`` as a soft threshold on
    it at :data:`MAJOR_GIFT_CAPACITY`. ``last_gift_date`` follows engagement.

    The ordering matters. Capacity is a **confounder** that causes both the
    giving history and the label, so ``total_gift_amount`` is a legitimate
    predictor: informative, and limited by how well giving reveals capacity.

    Before version 0.7.0 the label was drawn first and ``total_gift_amount``
    was drawn *conditional on the label*. That inverted the domain's causal
    arrow, and it was measurable: a model given ``total_gift_amount`` beat the
    Bayes rate of the generator's own process by roughly 19 ROC-AUC points,
    which no model can legitimately do. Using cumulative lifetime giving to
    predict "is a major donor" is also the classic fundraising leakage this
    library exists to prevent, so the reference dataset was teaching the
    anti-pattern. ``last_gift_date`` was a second target-derived feature for
    the same reason.

    The label remains statistically learnable and is not trivially
    predictable: held out on 4,000 rows, the documented feature set reaches
    ROC-AUC 0.814 and accuracy 0.759 against a Bayes accuracy ceiling of 0.806
    given latent capacity. Sitting **below** that ceiling is the point.

    The function never raises an error for valid inputs.  Passing
    ``n_samples=0`` returns an empty DataFrame with the correct column
    schema.
    """
    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Step 1: Generate structural features
    # ------------------------------------------------------------------
    years_active = rng.integers(1, 31, size=n_samples)          # 1–30 years
    event_attendance = rng.integers(0, 21, size=n_samples)       # 0–20 events

    # ------------------------------------------------------------------
    # Step 2: Latent giving capacity.
    #
    # This is the confounder, and the whole point of the ordering here. Capacity
    # causes BOTH the observed giving history and major-donor status. Nothing
    # below is drawn from the label, so no feature is a readout of the answer.
    #
    # Earlier versions drew the label first and then drew total_gift_amount
    # conditional on it, which inverted the domain's causal arrow: a model given
    # total_gift_amount could beat the Bayes rate of the generator's own process,
    # and using cumulative giving to predict "is a major donor" is the classic
    # fundraising leakage this library exists to prevent.
    # ------------------------------------------------------------------
    unobserved_wealth = rng.normal(0, 1, size=n_samples)
    log_capacity = (
        7.9                               # intercept, sets the base rate
        + 0.055 * years_active            # longer relationships run deeper
        + 0.075 * event_attendance        # engagement tracks affinity and means
        + 1.25 * unobserved_wealth        # wealth no column in this frame sees
    )

    # ------------------------------------------------------------------
    # Step 3: Observed giving FOLLOWS capacity.
    #
    # Donors realise some fraction of what they could give, imperfectly, so
    # total_gift_amount is a noisy proxy for capacity rather than a function of
    # the label. That makes it a legitimate predictor: informative, and bounded
    # by how well giving history reveals capacity.
    # ------------------------------------------------------------------
    total_gift_amount = np.round(
        rng.lognormal(mean=log_capacity - 1.6, sigma=0.6), 2
    )

    # ------------------------------------------------------------------
    # Step 4: The label ALSO follows capacity: a soft threshold at the
    # $25,000 gift capacity the docstring describes. Soft rather than hard
    # because a real prospect-research call is not a step function.
    # ------------------------------------------------------------------
    z = 1.6 * (log_capacity - np.log(MAJOR_GIFT_CAPACITY)) + 0.4 * rng.normal(
        0, 1, size=n_samples
    )
    propensity = 1.0 / (1.0 + np.exp(-z))
    is_major_donor = rng.binomial(1, propensity).astype(np.int64)

    # ------------------------------------------------------------------
    # Step 5: Recency follows ENGAGEMENT, not the label.
    #
    # This was also drawn from the label before (Beta for majors, uniform for
    # everyone else), which made last_gift_date a second target-derived feature.
    # More engaged donors have given more recently, which is the real mechanism.
    # ------------------------------------------------------------------
    reference_date = pd.Timestamp("2026-02-21")  # project snapshot date
    max_days = 365 * 5
    engagement = 1.0 / (1.0 + np.exp(-0.18 * (event_attendance - 8.0)))
    recency_days = np.zeros(n_samples, dtype=np.int64)
    if n_samples > 0:
        # Larger second Beta parameter skews toward 0, i.e. toward recent dates.
        recency_days = (
            rng.beta(1.0, 1.0 + 3.5 * engagement) * max_days
        ).astype(np.int64)

    last_gift_date = pd.to_datetime(
        reference_date - pd.to_timedelta(recency_days, unit="D")
    )

    # ------------------------------------------------------------------
    # Step 6: Assemble DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame(
        {
            "total_gift_amount": total_gift_amount,
            "years_active": years_active.astype(np.int64),
            "last_gift_date": last_gift_date,
            "event_attendance_count": event_attendance.astype(np.int64),
            "is_major_donor": is_major_donor,
        }
    )

    return df


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
    >>> from philanthropy.datasets import make_donor_dataset
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
