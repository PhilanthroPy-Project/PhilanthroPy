# What is PhilanthroPy?

PhilanthroPy is a production-ready Python library that slots directly into `sklearn.pipeline.Pipeline`. It covers the full predictive workflow for nonprofit and academic medical center (AMC) fundraising — from raw CRM cleaning and wealth imputation to major-gift propensity scoring, lapse prediction, and planned-giving intent.

## Motivation

Predictive fundraising in nonprofits and healthcare foundations is often dominated by proprietary, black-box vendor tools or brittle, ad-hoc Python scripts that suffer from subtle temporal data leakage across fiscal-year boundaries. Machine learning code specifically tailored for the nuances of philanthropic giving was mostly non-existent. 

I built PhilanthroPy to change this. The goal was to create a rigorous, open-source, and scikit-learn-compatible foundation for donor analytics. It democratizes advanced fundraising data science, allowing nonprofits to use their own data to safely and effectively identify their best prospects without relying entirely on expensive outside vendors.

## Key Features & Capabilities

PhilanthroPy provides a comprehensive suite of tools that are easy to understand and use:

* **Messy Data Cleaning** (`CRMCleaner`): Automatically standardises raw, messy CRM exports (like from Salesforce NPSP or Raiser's Edge), fixing dates and standardising currency amounts without crashing.
* **Fiscal Calendar Awareness** (`FiscalYearTransformer` & `TemporalDonorSplitter`): Nonprofits operate on Fiscal Years (e.g., July to June). PhilanthroPy natively understands these boundaries, preventing "future data" from accidentally leaking into historical predictive models.
* **Smart Wealth Imputation** (`WealthScreeningImputerKNN`): Third-party wealth vendors rarely match 100% of a database. This tool smartly estimates missing wealth capacities (like real estate value) using K-Nearest Neighbors based on similar donors in the database.
* **Grateful Patient Featurization** (`GratefulPatientFeaturizer` & `EncounterRecencyTransformer`): Built for Academic Medical Centers (AMCs), these are HIPAA-safe tools that translate clinical encounter histories (like visit dates and hospital departments) into powerful major-gift signals, entirely dropping sensitive patient identifiers (PHI).
* **Propensity & Share of Wallet** (`ShareOfWalletScorer` & `MovesManagementClassifier`): Turn-key estimators to calculate a donor's Share of Wallet (how much of their wealth goes to your cause) and dynamically predict the next best engagement step for a gift officer.

## Explore the Docs

* Go to **[Tutorials](tutorials/index.md)** to learn the basics.
* See **[How-To Guides](how-to/index.md)** for specific tasks.
* Read the **[Explanation](explanation/index.md)** section to understand the concepts behind PhilanthroPy.
* Browse the auto-generated **[API Reference](reference/index.md)**.
