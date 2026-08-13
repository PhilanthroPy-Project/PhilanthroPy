---
title: 'PhilanthroPy: A scikit-learn-native toolkit for leakage-safe donor analytics in nonprofit and academic medical center fundraising'
tags:
  - Python
  - scikit-learn
  - philanthropy
  - nonprofit
  - fundraising
  - predictive analytics
  - healthcare
authors:
  - name: Shivam Ashokbhai Lalakiya
    orcid: "0009-0000-5110-6540"
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 11 August 2026
bibliography: paper.bib
---

# Summary

PhilanthroPy is a scikit-learn-native Python library for predictive fundraising
analytics at nonprofits and academic medical center (AMC) foundations. It
covers the full predictive workflow used by advancement and development
offices — CRM cleaning, wealth-screening imputation, RFM segmentation,
donor-propensity and major-gift scoring, lapse prediction, planned-giving
intent, share-of-wallet estimation, and hybrid revenue forecasting — as a set
of `fit`/`transform`/`predict` estimators and transformers that compose
directly inside `sklearn.pipeline.Pipeline`. Every estimator that fits the
scikit-learn `fit(X, y)` contract passes
`sklearn.utils.estimator_checks.check_estimator`; the one documented exception
is `UpliftTLearner` (`philanthropy.experimental`), whose `fit(X, y,
treatment)` signature breaks that contract. A second, AMC-specific
surface (`EncounterTransformer`, `GratefulPatientFeaturizer`,
`DischargeToSolicitationWindowTransformer`) featurizes clinical-encounter data
for grateful-patient programs, where PHI-adjacent inputs raise the compliance
bar relative to general nonprofit use.

Every fitted statistic in the library — imputation fill values, wealth
percentile lookups, fiscal-year boundaries — is computed from training data
inside `fit` and frozen before `transform` or `predict` is ever called, so a
pipeline built on PhilanthroPy cannot leak test-period or future information
into a score without the user deliberately refitting on it. This contract is
enforced by a dedicated regression test suite and holds a required coverage
floor (currently 92% overall, 93% across the risk-tier subtree of
preprocessing, models, and ingest) as a condition of every merge.

# Statement of need

Nonprofit and university advancement teams, and AMC development offices,
routinely build donor-scoring models, but the tooling available to them sits
at two extremes: proprietary, closed-source scoring add-ons bundled with CRM
platforms (e.g. Salesforce NPSP, Blackbaud Raiser's Edge), which are not
inspectable, not extensible, and not portable across systems; or ad hoc,
one-off scripts written independently at each institution, which are rarely
tested for the temporal-leakage failure mode endemic to donor data — fitting
imputers, scalers, or feature encoders on a full historical dataset before
splitting it into train and test windows, so a model's reported performance
silently includes information from the future relative to the point a
real solicitation decision would be made.

PhilanthroPy addresses this gap by being simultaneously (1) fully
`scikit-learn`-compatible, so it drops into existing cross-validation, grid
search, and pipeline tooling that practitioners and researchers already use;
(2) leakage-safe by construction and by test, rather than by convention; and
(3) domain-specific, encoding fundraising and (for AMCs) grateful-patient
program knowledge — such as discharge-to-solicitation windows and matching-gift
detection — directly into named, documented transformers instead of leaving
that domain logic to be re-derived ad hoc at every institution. The
compliance posture is scoped explicitly: PhilanthroPy documents where its PII
handling is a name-based heuristic rather than a formal HIPAA
de-identification, so AMC adopters can make an informed risk decision instead
of assuming a guarantee the library does not make.

The library's design and target use case are grounded in the author's
published work on predictive donor analytics and fundraising intelligence at
scale [@lalakiya2025], and it is built on the standard scientific Python stack
[@scikit-learn; @numpy; @pandas].

# Acknowledgements

We thank the PhilanthroPy contributors credited in `CONTRIBUTORS.md`.

# References
