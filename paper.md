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
date: 21 August 2026
bibliography: paper.bib
---

# Summary

PhilanthroPy is a scikit-learn-native Python library for predictive fundraising
analytics at nonprofits and academic medical center (AMC) foundations. It covers
the workflow used by advancement and development offices: CRM cleaning,
wealth-screening imputation, RFM summarisation, donor-propensity and major-gift
scoring, lapse prediction, planned-giving intent, share-of-wallet estimation,
and revenue forecasting, exposed as `fit`/`transform`/`predict` estimators that
compose inside `sklearn.pipeline.Pipeline`. The one documented exception is
`RFMTransformer`, which aggregates gift rows to one row per donor and is
therefore a pre-pipeline step rather than a pipeline member.

Conformance is enforced as a mechanism rather than asserted as a property. A
`parametrize_with_checks` battery runs `sklearn.utils.estimator_checks` over 20
configured estimator instances, 1016 checks passing on scikit-learn 1.8.0. Four
classes cannot be instantiated bare or are row-reducing
(`RFMTransformer`, `MatchingGiftFeaturizer`, `EncounterTransformer`,
`GratefulPatientFeaturizer`); each has hand-written equivalent coverage and a
recorded written reason, and a test fails the build if any public estimator
appears in neither the battery nor that exemption registry. `UpliftTLearner`
(`philanthropy.experimental`) is outside the contract entirely, since its
`fit(X, y, treatment)` signature is not `fit(X, y)`.

A second, AMC-specific surface (`EncounterTransformer`,
`GratefulPatientFeaturizer`, `DischargeToSolicitationWindowTransformer`)
featurizes clinical-encounter data for grateful-patient programs, where
PHI-adjacent inputs raise the compliance bar relative to general nonprofit use.

# Statement of need

Advancement teams at nonprofits, universities, and AMC foundations routinely
build donor-scoring models, but their tooling sits at two extremes: proprietary
scoring add-ons sold alongside CRM platforms, which are not inspectable,
extensible, or portable; and ad hoc scripts written independently at each
institution, which are rarely tested for the temporal-leakage failure mode
endemic to donor data [@kaufman2012leakage; @kapoor2023leakage].

The part of that failure mode a `Pipeline` already solves is fitting a
transformer on test rows, and PhilanthroPy claims no credit for it. The part
nothing in the ecosystem solves is auxiliary-table timing. Clinical encounters,
event attendance, and wealth appends do not arrive as rows of `X`; they arrive
as a separate table passed to a transformer's constructor. Those rows are
therefore never part of any train/test split, so `Pipeline`, `cross_val_score`,
and `GroupKFold` have no jurisdiction over them at all. Aggregating such a table
without a cutoff scores a 2020 gift against a 2024 discharge, and the resulting
backtest is optimistic in a way no splitter can detect.

On a seeded synthetic donor-year panel of 3000 donors over six panel years,
five seeds, that distinction dominates every other design choice measured
(`scripts/leakage_experiment.py`). Building features over the whole export
rather than as of each decision point inflates walk-forward cross-validated
ROC-AUC from 0.625 to 0.750, an inflation of **+0.126 AUC**. Choosing a random
`StratifiedKFold` over a walk-forward split, the mistake the folklore warns
about, costs an order of magnitude less: both understate a genuinely held-out
final year, by 0.030 and 0.014 respectively. PhilanthroPy's response is an
`as_of` parameter on the transformers that consume auxiliary tables, which
drops rows observed after a stated cutoff before any aggregation runs.

The same two experiments were then replicated on a real donor file,
KDD Cup 1998 [@kddcup1998], 95,412 donors reshaped into a 22-period
mail/response panel (`scripts/real_data_leakage_experiment.py`). The effect
is not smaller on real data; it is larger. Whole-history feature construction
inflates walk-forward ROC-AUC by **+0.376 AUC** (0.482 to 0.858), roughly three
times the synthetic figure, because a donor's lifetime total is a stickier,
more identity-revealing signal in real heavy-tailed giving data than in the
synthetic generator. Splitter choice also matters more than the synthetic run
suggested: a random `StratifiedKFold` overstates a genuinely held-out final
period by +0.107 AUC, an order of magnitude larger than the synthetic 0.014-
0.030. Both numbers were predicted, in the script's own docstring, to be
*smaller* than the synthetic figures before the script was ever run; that
prediction was wrong, and the disagreement is reported rather than revised
after the fact.

The model-benchmark figures elsewhere in the documentation remain synthetic.
The one other real dataset the package ships, `load_ciob_fundraising`, is an
institutional affiliation registry with no donor rows, gift amounts, or
labels, and no estimator is fitted on it anywhere in the project.

# State of the field

Most individual pieces of PhilanthroPy have an incumbent, and the contribution
is the domain-specific assembly rather than any single component.
`feature-engine` [@featureengine] and scikit-lego provide general
imputation-with-indicator and grouped temporal splitters; `mlxtend`
[@mlxtend] and `sktime` [@sktime] both ship expanding-window grouped
splitters that generalise `FiscalYearGroupedSplitter`; `pymc-marketing`
[@pymcmarketing], successor to `lifetimes`, provides RFM summarisation and a
leakage-aware RFM train/test split built on the Fader-Hardie buy-till-you-die
tradition [@faderhardielee2005]; MAPIE [@mapie] and `crepes` [@crepes]
implement conformal prediction far beyond the single split-conformal p-value
exposed here; and `scikit-uplift` covers two-model uplift estimation.
PhilanthroPy's wealth imputers reduce to `SimpleImputer` or `KNNImputer` with a
missing indicator at their defaults, and its model classes are pipeline-safe,
domain-named adapters over scikit-learn ensembles rather than new estimators.

What has no open-source equivalent, to the author's knowledge, is the AMC
surface: encounter aggregation with an as-of cutoff on a constructor-passed
table, discharge-to-solicitation window encoding, service-line intensity, and a
compliance posture written against the HIPAA fundraising carve-out at 45 CFR
164.514(f) rather than against de-identification. Grateful-patient programs are
an established and ethically scrutinised practice [@collins2018grateful]
serviced almost entirely by proprietary vendor scores. That gap, plus the value
of having the sector's canonical steps under one tested and citable API, is the
case for the package.

# Software design

The library is ten subpackages under one namespace, with public classes living
in private modules and re-exported through each subpackage's `__all__`. Three
design constraints shape everything else.

**Freeze at fit.** Every fitted statistic, imputation fill values and wealth
percentile lookups among them, is computed from training data inside `fit` and
frozen before `transform` or `predict` runs, so `transform` is idempotent and a
transformer cannot silently refit on the batch it is scoring. A dedicated
regression suite enforces this. It is a statement about fitted state, not a
guarantee about feature timing: bounding features to the decision point
additionally requires setting `as_of` on the transformers that take auxiliary
tables, and choosing donor-level rather than row-level holdout when the label is
static per donor.

**scikit-learn, pandas, numpy, matplotlib, and seaborn only.** No deep-learning
framework is a dependency, and the cost is stated rather than hidden:
`FinancialForecastModel` is a linear trend plus an `MLPRegressor` residual
correction, a decomposition in the spirit of hybrid linear-plus-nonlinear
forecasting [@zhang2003hybrid], not an LSTM-ARIMA model.

**Gates in CI, not conventions in prose.** Every pull request runs flake8,
mypy, the docstring examples, and the test suite against a 92% coverage floor
declared once in `pyproject.toml`, plus a separate 93% floor over the risk-tier
subtree defined once in the `Makefile` (preprocessing, models, ingest, the CLI,
and model persistence; `philanthropy.metrics` is outside that tier). The
conformance registry described in the Summary is enforced the same way.

`philanthropy.metrics` also exposes `conformal_pvalue`, the non-smoothed
split-conformal p-value against a held-out calibration set, in the
`(1 + |{i : s_i >= s}|) / (n + 1)` form of eq. (3) of @bates2023outliers, whose
`+1` terms retain the test point [@vovk2005conformal]. Under exchangeability of
the calibration and test scores the statistic is super-uniform, so thresholding
it at `alpha` bounds the expected selection rate at `alpha` in finite samples.
Reading that bound as a false-positive rate additionally requires the
calibration set to contain only nulls, which is the construction in
@bates2023outliers and is documented as the user's responsibility.

# Research impact statement

PhilanthroPy is not yet in documented use at a third-party institution, and
this paper does not claim otherwise. What exists today is the software, a
reproducible leakage experiment whose result is reported above and now
replicated on real donor data (KDD Cup 1998 [@kddcup1998]) rather than only on
a synthetic generator, a benchmark page that states its own synthetic-data
limitations and the Bayes-optimal ceiling of its generator, an archived
release on Zenodo, and eleven merged pull requests from four contributors
external to the project. The author's prior conference paper on predictive
donor analytics [@lalakiya2025] is related work by the same author on
different data, not an independent evaluation of this software.

The intended research audience is twofold: advancement-analytics practitioners
and consultants who currently rebuild these steps per engagement, and
researchers of philanthropic giving who need an auditable, citable alternative
to vendor scores whose construction cannot be inspected. The library is built
on the standard scientific Python stack [@scikit-learn; @numpy; @pandas].

# AI usage disclosure

Generative AI assistance (Anthropic Claude, used through Claude Code in an
agentic workflow rather than line completion alone) was used during development
of this package across implementation, tests, documentation, and the drafting of
this paper.

The design constraints are the author's and predate any generated code: the
freeze-at-fit contract, the dependency rule, the estimator conventions, the
stability tiers, and the compliance posture. Generated code that violated them
was rejected rather than merged. No numeric split between authored and
AI-assisted output is offered, because the author has not measured one and
declines to estimate it; the review gate is stated instead. Nothing lands
without the full gate green: flake8, mypy, the docstring examples, the test
suite against the coverage floors above, an OS and Python-version matrix, a
dependency-floor install, a packaging-metadata check, and the conformance
battery. The author reviewed, edited, and validated all AI-assisted output,
including every claim and citation in this paper, and made the core design
decisions.

# Acknowledgements

I thank the PhilanthroPy contributors credited in `CONTRIBUTORS.md`. This work
received no specific grant funding, and the author declares no competing
interests.

# References
