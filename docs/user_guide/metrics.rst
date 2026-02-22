Philanthropic Metrics
=====================

Standard statistical metrics (Accuracy, R-Squared) often fail to capture the organizational impact of fundraising. PhilanthroPy provides metrics that speak the language of development officers and CFOs.

.. currentmodule:: philanthropy.metrics

Lifetime Value Analysis
-----------------------

Donor Lifetime Value (DLV)
~~~~~~~~~~~~~~~~~~~~~~~~~~
The "North Star" metric for long-term donor health.

The :func:`donor_lifetime_value` function computes the Net Present Value (NPV) of all future expected gifts from a donor.

**The Formula:**
PhilanthroPy uses a discounted flow model:
.. math::

   DLV = \frac{G \times (1 - (1+d)^{-L})}{d}

Where:
*   **G**: Average Annual Giving.
*   **d**: Discount Rate (Time Value of Money).
*   **L**: Donor Lifespan (often derived as ``1 / (1 - Retention Rate)``).

**Why it matters:**
DLV allows you to justify higher Acquisition Costs for "high-value" segments (like grateful patients) even if their initial gift is small.

Efficiency & ROI
----------------

Donor Acquisition Cost (DAC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The cost of bringing in one new donor.

**Calculation:**
Total Campaign Spend / Number of New Donors Acquired.

**Strategic Benchmarking:**
Ideally, a donor's Year 1 gift should cover their DAC, or their 3-year DLV should be at least 3x their DAC.

Retention Rate
~~~~~~~~~~~~~~
The percentage of donors from Period A who gave again in Period B.

Retention is the single biggest lever for total revenue growth. PhilanthroPy's metrics module allows you to segment retention by acquisition channel, helping you identify which sources (e.g., Direct Mail vs. High-Touch Events) produce the most loyal donors.
