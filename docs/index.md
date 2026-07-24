<div class="ap-hero" markdown>
<span class="ap-hero__eyebrow">Open source · scikit-learn native</span>

# Predictive donor analytics, done right. { .ap-hero__title }

<p class="ap-hero__sub">A leakage-safe, pipeline-ready toolkit for nonprofit and academic-medical-center fundraising. Every estimator passes scikit-learn's <code>check_estimator</code>.</p>

<div class="ap-cta" markdown>
[Get started](tutorials/index.md){ .md-button }
[View on GitHub](https://github.com/PhilanthroPy-Project/PhilanthroPy){ .md-button .md-button--secondary }
</div>

</div>

<div class="ap-specs">
  <div class="ap-specs__item"><span class="ap-specs__k ap-specs__k--ok">Leakage-safe</span><span class="ap-specs__v">train-only statistics, frozen before transform</span></div>
  <div class="ap-specs__item"><span class="ap-specs__k ap-specs__k--ok">check_estimator</span><span class="ap-specs__v">passes scikit-learn's compliance suite</span></div>
  <div class="ap-specs__item"><span class="ap-specs__k ap-specs__k--ok">Pipeline-ready</span><span class="ap-specs__v">drops into sklearn.pipeline.Pipeline</span></div>
  <div class="ap-specs__item"><span class="ap-specs__k ap-specs__k--ok">MIT</span><span class="ap-specs__v">open source, no vendor lock-in</span></div>
</div>

## What is PhilanthroPy?

PhilanthroPy is a production-ready Python library that slots directly into `sklearn.pipeline.Pipeline`. It covers the full predictive workflow for nonprofit and academic medical center (AMC) fundraising — from raw CRM cleaning and wealth imputation to major-gift propensity scoring, lapse prediction, and planned-giving intent.

## Quick start

Get up and running in seconds:

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

Predictive fundraising in nonprofits and healthcare foundations is often dominated by proprietary, black-box vendor tools — or brittle, ad-hoc Python scripts that suffer from subtle temporal data leakage across fiscal-year boundaries. Machine-learning code built for the nuances of philanthropic giving was mostly non-existent.

PhilanthroPy exists to change that: a rigorous, open-source, **scikit-learn-compatible** foundation for donor analytics. It puts advanced fundraising data science within reach of any team, so nonprofits can use their own data to safely and effectively identify their best prospects — without relying entirely on expensive outside vendors.

---

## Key features & capabilities

A comprehensive suite of tools, easy to understand and use:

<div class="grid cards" markdown>

- :material-database-refresh: **Messy data cleaning**

    ---
    Standardises raw CRM exports (Salesforce NPSP, Raiser's Edge) — fixing dates and currency amounts without crashing. *Uses `CRMCleaner`.*

- :material-calendar-range: **Fiscal-calendar awareness**

    ---
    Nonprofits run on fiscal years (e.g. July–June). PhilanthroPy understands these boundaries natively, preventing future data from leaking into historical models. *Uses `FiscalYearTransformer`.*

- :material-currency-usd: **Smart wealth imputation**

    ---
    Third-party wealth vendors rarely match every record. This estimates missing wealth capacity (like real-estate value) from similar donors using K-nearest neighbours. *Uses `WealthScreeningImputerKNN`.*

- :material-hospital-building: **Grateful-patient featurization**

    ---
    For academic medical centers, translates clinical-encounter histories into major-gift signals while decoupling them from explicit patient identifiers (PHI). This reduces compliance risk but is **not** formal HIPAA de-identification — see [Compliance Considerations](explanation/compliance_considerations.md). *Uses `GratefulPatientFeaturizer`.*

- :material-chart-bell-curve-cumulative: **Propensity & share of wallet**

    ---
    Turn-key estimators for a donor's share of wallet (how much of their capacity goes to your cause) and the next best engagement step for a gift officer. *Uses `ShareOfWalletScorer`.*

</div>

!!! tip "Getting started"
    The quickest way to get familiar with PhilanthroPy is to dive into the **[Tutorials](tutorials/index.md)**.

## Explore the docs

<div class="grid cards" markdown>

- **[Tutorials](tutorials/index.md)**

    ---
    Step-by-step, learning-oriented lessons for beginners.

- **[How-To Guides](how-to/index.md)**

    ---
    Goal-oriented recipes for specific tasks.

- **[Explanation](explanation/index.md)**

    ---
    Understanding-oriented concepts and architecture.

- **[API Reference](reference/index.md)**

    ---
    Information-oriented API docs.

</div>
