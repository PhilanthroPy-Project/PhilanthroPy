# What is PhilanthroPy?

PhilanthroPy is a production-ready Python library that slots directly into `sklearn.pipeline.Pipeline`. It covers the full predictive workflow for nonprofit and academic medical center (AMC) fundraising — from raw CRM cleaning and wealth imputation to major-gift propensity scoring, lapse prediction, and planned-giving intent.

## Quick Start

Get up and running with PhilanthroPy in seconds:

=== "pip"
    ```bash
    pip install philanthropy
    ```

=== "conda"
    ```bash
    conda env create -f environment.yml && conda activate Philanthropy
    pip install -e ".[dev]"
    ```

---

## Motivation

Predictive fundraising in nonprofits and healthcare foundations is often dominated by proprietary, black-box vendor tools or brittle, ad-hoc Python scripts that suffer from subtle temporal data leakage across fiscal-year boundaries. Machine learning code specifically tailored for the nuances of philanthropic giving was mostly non-existent. 

I built PhilanthroPy to change this. The goal was to create a rigorous, open-source, and **scikit-learn-compatible** foundation for donor analytics. It democratizes advanced fundraising data science, allowing nonprofits to use their own data to safely and effectively identify their best prospects without relying entirely on expensive outside vendors.

---

## Key Features & Capabilities

PhilanthroPy provides a comprehensive suite of tools that are easy to understand and use:

<div class="grid cards" markdown>

- :material-database-refresh: **Messy Data Cleaning**

    ---
    Automatically standardises raw, messy CRM exports (like from Salesforce NPSP or Raiser's Edge), fixing dates and standardising currency amounts without crashing. *Uses `CRMCleaner`.*

- :material-calendar-range: **Fiscal Calendar Awareness**

    ---
    Nonprofits operate on Fiscal Years (e.g., July to June). PhilanthroPy natively understands these boundaries, preventing "future data" from accidentally leaking into historical predictive models. *Uses `FiscalYearTransformer`.*

- :material-currency-usd: **Smart Wealth Imputation**

    ---
    Third-party wealth vendors rarely match 100% of a database. This tool smartly estimates missing wealth capacities (like real estate value) using K-Nearest Neighbors based on similar donors in the database. *Uses `WealthScreeningImputerKNN`.*

- :material-hospital-building: **Grateful Patient Featurization**

    ---
    Built for Academic Medical Centers (AMCs), these are HIPAA-safe tools that translate clinical encounter histories into powerful major-gift signals, entirely dropping sensitive patient identifiers (PHI). *Uses `GratefulPatientFeaturizer`.*

- :material-chart-bell-curve-cumulative: **Propensity & Share of Wallet**

    ---
    Turn-key estimators to calculate a donor's Share of Wallet (how much of their wealth goes to your cause) and dynamically predict the next best engagement step for a gift officer. *Uses `ShareOfWalletScorer`.*

</div>

!!! tip "Getting Started"
    The quickest way to get familiar with PhilanthroPy is by diving into the **[Tutorials](tutorials/index.md)**.

## Explore the Docs

<div class="grid cards" markdown>

- **[Tutorials](tutorials/index.md)**
  Step-by-step, learning-oriented lessons for beginners.
- **[How-To Guides](how-to/index.md)**
  Goal-oriented recipes for specific tasks.
- **[Explanation](explanation/index.md)**
  Understanding-oriented, high-level concepts and architecture.
- **[API Reference](reference/index.md)**
  Information-oriented API docs.

</div>
