PhilanthroPy
============

.. rst-class:: lead

    A scikit-learn compatible toolkit for predictive donor analytics in the nonprofit sector.

.. container:: button-container

    .. button-ref:: user_guide
        :ref-type: doc
        :color: primary
        :shadow:

        Getting Started

    .. button-ref:: auto_examples/index
        :ref-type: doc
        :color: secondary
        :shadow:

        View Examples

    .. button-link:: https://github.com/PhilanthroPy-Project/PhilanthroPy
        :color: success
        :shadow:

        GitHub


What is PhilanthroPy?
---------------------

PhilanthroPy is a production-ready Python library that slots directly into `sklearn.pipeline.Pipeline`. It covers the full predictive workflow for nonprofit and academic medical center (AMC) fundraising — from raw CRM cleaning and wealth imputation to major-gift propensity scoring, lapse prediction, and planned-giving intent.

Motivation
----------

Predictive fundraising in nonprofits and healthcare foundations is often dominated by proprietary, black-box vendor tools or brittle, ad-hoc Python scripts that suffer from subtle temporal data leakage across fiscal-year boundaries. Machine learning code specifically tailored for the nuances of philanthropic giving was mostly non-existent. 

I built PhilanthroPy to change this. The goal was to create a rigorous, open-source, and scikit-learn-compatible foundation for donor analytics. It democratizes advanced fundraising data science, allowing nonprofits to use their own data to safely and effectively identify their best prospects without relying entirely on expensive outside vendors.

Key Features & Capabilities
---------------------------

PhilanthroPy provides a comprehensive suite of tools that are easy to understand and use:

* **Messy Data Cleaning** (``CRMCleaner``): Automatically standardises raw, messy CRM exports (like from Salesforce NPSP or Raiser's Edge), fixing dates and standardising currency amounts without crashing.
* **Fiscal Calendar Awareness** (``FiscalYearTransformer`` & ``TemporalDonorSplitter``): Nonprofits operate on Fiscal Years (e.g., July to June). PhilanthroPy natively understands these boundaries, preventing "future data" from accidentally leaking into historical predictive models.
* **Smart Wealth Imputation** (``WealthScreeningImputerKNN``): Third-party wealth vendors rarely match 100% of a database. This tool smartly estimates missing wealth capacities (like real estate value) using K-Nearest Neighbors based on similar donors in the database.
* **Grateful Patient Featurization** (``GratefulPatientFeaturizer`` & ``EncounterRecencyTransformer``): Built for Academic Medical Centers (AMCs), these are HIPAA-safe tools that translate clinical encounter histories (like visit dates and hospital departments) into powerful major-gift signals, entirely dropping sensitive patient identifiers (PHI).
* **Propensity & Share of Wallet** (``ShareOfWalletScorer`` & ``MovesManagementClassifier``): Turn-key estimators to calculate a donor's Share of Wallet (how much of their wealth goes to your cause) and dynamically predict the next best engagement step for a gift officer.

Design Principles
-----------------

* **Leakage-safe by design** — fill statistics, encounter summaries, and encounter snapshots are all frozen at ``fit()`` time; ``transform()`` is fully idempotent.
* **sklearn-native** — all estimators pass ``check_estimator``; support ``set_output(transform="pandas")``, ``clone()``, ``get_params()`` / ``set_params()``.
* **NaN-transparent** — wealth and clinical transformers declare ``allow_nan = True``; no silent data loss.
* **PII-aware** — ``EncounterTransformer`` auto-drops PII-like columns before returning features.

****


.. grid:: 1 2 2 3
    :gutter: 4

    .. grid-item-card:: :octicon:`book;2em` User Guide
        :link: user_guide
        :link-type: doc

        Learn the fundamentals of PhilanthroPy, including how to build propensity models and run donor segmentations.

    .. grid-item-card:: :octicon:`code;2em` API Reference
        :link: api
        :link-type: ref

        Detailed documentation of all functions, classes, and methods available in the `philanthropy` package.

    .. grid-item-card:: :octicon:`graph;2em` Examples
        :link: auto_examples/index
        :link-type: doc

        A gallery of real-world use cases, plotting examples, and end-to-end machine learning pipelines.

    .. grid-item-card:: :octicon:`tools;2em` Development
        :link: user_guide/development
        :link-type: doc

        Learn how to run the full test suite, verify scikit-learn compliance, and build the documentation locally.

.. toctree::
    :maxdepth: 2
    :hidden:

    user_guide
    api
    auto_examples/index
