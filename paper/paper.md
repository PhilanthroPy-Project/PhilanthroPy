---
title: 'PhilanthroPy: A scikit-learn-native toolkit for predictive donor analytics'
tags:
  - Python
  - scikit-learn
  - fundraising
  - nonprofit
  - donor analytics
  - healthcare philanthropy
  - machine learning
authors:
  - name: Shivam Lalakiya
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Northeastern University, Boston, MA, USA
    index: 1
date: 24 July 2026
bibliography: paper.bib
---

# Summary

`PhilanthroPy` is a Python library for predictive fundraising analytics in
nonprofit and academic-medical-center (AMC) advancement offices. It provides
transformers and estimators that plug directly into `scikit-learn`
[@scikit-learn; @sklearn_api] pipelines and are built on `pandas`
[@mckinney2010data] and `numpy` [@harris2020array]. The library covers the
full predictive workflow that development teams rely on: cleaning raw CRM
exports, imputing third-party wealth screening, and scoring donors for
major-gift propensity, lapse (attrition) risk, planned-giving intent, and
share of wallet.

Every component follows the standard `fit`/`transform`/`predict` contract and
is validated against scikit-learn's `check_estimator` conformance suite, so
the estimators support cloning, cross-validation, hyperparameter search, and
`set_output(transform="pandas")` without adapter code. Preprocessing lives in
`philanthropy.preprocessing` (e.g., `CRMCleaner`, `FiscalYearTransformer`,
`WealthScreeningImputerKNN`, `GratefulPatientFeaturizer`,
`ShareOfWalletScorer`); domain models live in `philanthropy.models` (e.g.,
`DonorPropensityModel`, `MajorGiftClassifier`, `LapsePredictor`,
`PlannedGivingIntentScorer`, `MovesManagementClassifier`,
`FinancialForecastModel`); and fiscal-year-aware cross-validation is provided
by `FiscalYearGroupedSplitter` in `philanthropy.model_selection`.

# Statement of need

Predictive fundraising is dominated by proprietary, black-box vendor scores or
by ad-hoc scripts that carry two recurring defects. First, they leak temporal
information: gift and engagement histories are aggregated without respecting
the fiscal-year boundary of the prediction, so a model that looks strong in
backtests degrades in production [@kaufman2012leakage; @kapoor2023leakage].
Second, they discard the "missingness"
signal — third-party wealth and demographic appends routinely match only a
fraction of a database, and naive imputation or row-dropping erases both data
and the predictive value of the gaps themselves.

`PhilanthroPy` addresses both problems as first-class design constraints.
Transformers freeze all fit-time statistics (fill values, encounter
summaries, imputation snapshots) at `fit()` and reuse them at `transform()`,
so repeated application to streaming data is idempotent and non-leaking, and
`FiscalYearGroupedSplitter` anchors cross-validation splits to the
organization's fiscal calendar to simulate realistic walk-forward evaluation.
The transformers operate with `allow_nan=True` throughout, extracting signal
from missingness rather than silently dropping records.

The library also targets a domain that general-purpose ML tooling does not
serve: academic medical center philanthropy [@collins2018grateful].
Grateful-patient fundraising depends on clinical encounter histories that are
governed by patient-privacy regulation. `GratefulPatientFeaturizer` translates encounter records into
major-gift signals (service-line intensity, recency, discharge-to-solicitation
windows) while dropping identifier-like columns such as medical record numbers
by name, reducing the risk that protected health information flows into
downstream feature tables. This name-based dropping is a configurable
defense-in-depth heuristic, not formal HIPAA de-identification, which remains
the deploying institution's responsibility.

Finally, `PhilanthroPy` is the modeling half of an ecosystem. Advancement data
arrives from fragmented sources (giving platforms, event systems, and multiple
CRMs), and the companion project `UniSchema` normalizes these webhooks into a
single `ConstituentEvent` stream. The `philanthropy.ingest` module bridges the
two: `read_constituent_events` loads UniSchema's JSON/NDJSON egress and
`constituent_events_to_features` aggregates it into the one-row-per-donor
feature table the estimators expect. This aggregation is leakage-safe in the
same spirit as the transformers and deduplicates repeated events by their
`eventId`, so practitioners move from raw events to model-ready features
without bespoke glue code.

The intended audience is data practitioners at nonprofits and healthcare
foundations, and researchers studying philanthropic giving, who want a
rigorous, open, and reproducible alternative to opaque vendor scoring built on
the tools they already use.

# Acknowledgements

`PhilanthroPy` builds on the scientific Python ecosystem, in particular
`scikit-learn`, `pandas`, and `numpy`.

# References
