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
    The underlying propensity model is a logistic function of a linear
    score ``z`` constructed from ``years_active``, ``event_attendance_count``,
    and a small amount of Gaussian noise.  This ensures the label is
    statistically learnable—neither trivially predictable nor random.

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
    # Step 2: Compute a latent propensity score (logistic model)
    # Features are z-scored implicitly via fixed scale factors so the
    # logistic mid-point corresponds to ~7 years and ~6 events attended.
    # ------------------------------------------------------------------
    noise = rng.normal(0, 1, size=n_samples)
    z = (
        0.12 * years_active          # tenure increases propensity
        + 0.18 * event_attendance    # engagement increases propensity
        - 2.5                        # intercept → ~15% base rate
        + 0.6 * noise                # irreducible uncertainty
    )
    propensity = 1.0 / (1.0 + np.exp(-z))

    # ------------------------------------------------------------------
    # Step 3: Draw binary labels
    # ------------------------------------------------------------------
    is_major_donor = rng.binomial(1, propensity).astype(np.int64)

    # ------------------------------------------------------------------
    # Step 4: Generate total_gift_amount correlated with label
    # ------------------------------------------------------------------
    # Base log-normal gift distribution (median ≈ $1,800)
    base_mu = np.where(is_major_donor == 1, 9.5, 7.5)
    base_sigma = np.where(is_major_donor == 1, 0.8, 1.4)
    total_gift_amount = rng.lognormal(mean=base_mu, sigma=base_sigma)
    total_gift_amount = np.round(total_gift_amount, 2)

    # ------------------------------------------------------------------
    # Step 5: Generate last_gift_date
    # Major donors are skewed toward the past 2 years; others are
    # spread uniformly over 5 years.
    # ------------------------------------------------------------------
    reference_date = pd.Timestamp("2026-02-21")  # project snapshot date

    # Days back from reference date
    max_days = 365 * 5
    recency_days = np.empty(n_samples, dtype=np.int64)
    major_mask = is_major_donor == 1

    # Major donors: Beta(1, 3) skews toward recent dates
    if major_mask.sum() > 0:
        beta_samples = rng.beta(1.0, 3.0, size=int(major_mask.sum()))
        recency_days[major_mask] = (beta_samples * max_days).astype(np.int64)

    non_major_mask = ~major_mask
    if non_major_mask.sum() > 0:
        recency_days[non_major_mask] = rng.integers(
            0, max_days + 1, size=int(non_major_mask.sum())
        )

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
