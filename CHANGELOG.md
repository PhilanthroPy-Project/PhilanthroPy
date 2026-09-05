# Changelog

All notable changes to PhilanthroPy are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

## [Unreleased]

### Added
- **`datasets.make_donor_panel`** (Tier 2, Beta): a seeded multi-year donor
  panel returning gift-level rows rather than one aggregated row per donor.
  `generate_synthetic_donor_data` cannot demonstrate `RFMTransformer` (needs a
  gift log), `FiscalYearGroupedSplitter` (needs repeated donor-years), an
  `as_of` cutoff (needs something to cut off), or the grateful-patient
  transformers (need encounters), so the only generator that could show the
  library's central ideas lived privately inside
  `scripts/leakage_experiment.py`. This promotes it.
  - Returns `{"gifts", "donors"}`, plus `"encounters"` when
    `include_encounters=True`. Column names match what the transformers
    already require, so nothing has to be renamed on the way in.
  - Fiscal years run 1 July to 30 June, labelled by the year they end in. At
    most one gift per donor-year, so "recent" is well defined.
  - **No label column, deliberately.** A label is a claim about a point in
    time, and shipping one pre-computed hands every user the exact mistake this
    package exists to prevent. The docstring shows the one-line derivation.
  - `wealth_estimate` is ~30% missing by design, because a wealth screen that
    came back for every record is not a wealth screen anyone has received.
  - `scripts/leakage_experiment.py` now imports it instead of defining a
    private copy, so the published experiment and the tutorials run on the same
    generator. The experiment's numbers are unchanged, and not merely to three
    decimals: the aggregated frames are asserted byte-identical to the ones the
    private generator produced, on all five published seeds. Gift amounts are
    deliberately **not** rounded to cents for that reason; rounding moved the
    reported min-max ranges by 0.001 AUC.

### Deprecated
- `FiscalYearGroupedSplitter`'s default for `drop_repeat_donors` (currently `False`) is deprecated and will change to `True` in 0.8.0. Leaving it at its default now emits a `DeprecationWarning`. Pass `drop_repeat_donors=False` explicitly to silence the warning and retain current behavior. Closes #108, by @shubhrai23.

### Added
- `credit-guard` CI job: pull requests touching `philanthropy/` must also
  update this changelog, and the author must be credited in
  CONTRIBUTORS.md. Implemented as `scripts/check_credit.sh`, wired into
  `ci.yml` on `pull_request` events only; failures surface as inline
  `::error::` annotations on the Files tab. Closes #113.
- Regression coverage ensuring `MovesManagementClassifier.fit` preserves
  DataFrame column names in `feature_names_in_`. Closes #54.
- `.github/workflows/pypi-smoke.yml`: a weekly (Mondays 12:00 UTC) and
  manually dispatchable job that installs the **published wheel** from PyPI on
  Linux, macOS and Windows, then runs `examples/quickstart.py` and
  `philanthropy --help`. Every other job tests the working tree; this one tests
  what `pip install philanthropy` actually serves, which is the only thing a new
  user or a reviewer runs. The repository is checked out into a subdirectory and
  the job asserts `philanthropy.__file__` resolves inside `site-packages`, so a
  `philanthropy/` directory in the working directory cannot silently shadow the
  wheel and turn this into a second working-tree test. Windows is included
  because the main matrix is Linux plus macOS.
### Fixed
- `CRMCleaner.transform` no longer silently corrupts complex amounts into
  wrong finite floats: cells holding actual `complex` values are masked to
  NaN with a `UserWarning` naming them, and a column where nothing parses
  (all-complex included) still raises `could not parse` per the documented
  contract. Closes #129.
### Added
- **`models.GiftIntervalCalibrator`**: distribution-free intervals on a dollar
  amount. Wraps an already-fitted regressor (`AskAmountRecommender`,
  `ShareOfWalletRegressor`, or any `predict`-per-row estimator) and calibrates on
  held-out rows via split conformal prediction. Until now nothing in the package
  returned an interval on a gift amount; every dollar-valued estimator returned
  a point.
  - Refuses below the certification floor. One order statistic needs
    `n >= 1/alpha - 1` calibration rows, 19 at the 95 % level, and `fit` raises
    rather than returning an interval it cannot certify. The floor is computed in
    `fractions.Fraction`, because it is a ceiling and `int(1 / alpha - 1)`
    truncates: at `alpha = 0.07` that reports 13 where the floor is 14. There is
    no parameter that switches the check off.
  - Reports the **attained** level, `r / (n + 1)`, on the returned
    `GiftInterval` and as `attained_level_`. A request for 0.95 resolves to
    0.9677 at 30 calibration rows and 0.9524 at 20; the requested level is kept
    separately as `requested_level_`.
  - Three one-rank conformity scores via `score=`: `"absolute"`,
    `"difficulty"` (residual over a difficulty estimate) and `"log"` (residual
    on `log1p` dollars, inverted). Equal-tailed two-rank intervals are
    deliberately not offered: two order statistics at `alpha / 2` more than double the
    floor to 39 rows and buy nothing the one-rank forms do not.
  - Intersects the interval with `[lower_bound, inf)`, default `0.0`. A gift
    cannot be negative, so coverage is bit-identical and width strictly falls.
    Calibration targets below the bound raise, since they are evidence the bound
    is wrong.
  - Optional `groups=` calibrates within a segment. A pooled calibration set is
    dominated by whichever segment supplies most of the rows and under-covers the
    others however much marginal data is added; a group below the per-group floor
    is refused by name rather than quietly pooled with a segment at another
    capacity level.
- **`metrics.interval_score` and `metrics.interval_report`**: the interval score
  `(u - l) + (2/alpha)(l - y)+ + (2/alpha)(y - u)+`, which is proper for a
  central interval, plus a report carrying coverage, the score as mean/median/
  trimmed mean (it is a heavy-tailed loss on gift amounts, and a ranking that
  flips between the three is a ranking of the tail), median width, and
  `width_ratio` = median width over median target. A valid interval can carry no
  information; the ratio is what separates the two.
- Test coverage for `PlannedGivingIntentScorer.predict_intent_score`: the
  single-class `predict_proba` fallback path that returns an all-zero score.
  (#56)

### Changed
- `ShareOfWalletScorer` output column 0 is renamed `sow_score` →
  `capacity_utilisation_ratio`. The formula was always capacity ÷ clipped
  modelled wealth: utilisation of estimated capacity, with no term for giving to
  *your* institution, so the old name claimed a share-of-wallet quantity the
  score cannot express (the class docstring has warned about exactly this since
  it shipped). Values, column order, and `capacity_tier` are unchanged; code
  reading column 0 positionally needs nothing. Code spelling the name gets one
  published minor of grace via `get_legacy_feature_names_out()`, which returns
  the old `["sow_score", "capacity_tier"]` under a `DeprecationWarning` and is
  removed in 0.9.0; the shim is registered in `tests/test_deprecations.py`.
  Closes #109.
- `predict_<thing>_interval` joins `_score` and `_forecast` as an accepted
  domain-method suffix in the public-API naming contract
  (`tests/test_public_api_contract.py`, `AGENTS.md`).
- `test_predict_methods_are_callable_with_x_alone_and_return_one_value_per_row`
  now skips non-estimator symbols in `models.__all__` instead of raising
  `KeyError` on the first one.

### Deprecated
- `philanthropy.utils.make_donor_dataset` moves to
  [`philanthropy.datasets.make_donor_dataset`](philanthropy/datasets/) and the
  old location emits a `DeprecationWarning`; removed in 0.8.0. The gift-level
  generator now lives next to `generate_synthetic_donor_data`, which is the
  canonical datasets home. Closes #111.

## [1.0.0] - TBD

The API freeze. No code changes: 1.0.0 is a promise, not a feature.

### Changed
- `Development Status :: 5 - Production/Stable`.
- **Tier 1 is now semver-protected.** A breaking change to any Tier 1 symbol
  requires a major release, preceded by one full published minor emitting
  `DeprecationWarning`. Tier 2 may still break in a minor; Tier 3 carries no
  guarantee. The tiers are listed per-symbol in
  [docs/reference/index.md](docs/reference/index.md).

### Added
- `test_stability_tier_table_covers_every_public_symbol`: the tier table is now
  machine-checked against `__all__`, so a new public symbol cannot ship without
  a stated tier. At 1.0 that table is the contract; an out-of-date one is a
  broken promise, not a docs nit.

### Notes
The five 1.0 gates all hold at this commit: 0.7.0 published; the public-API
contract test green with no exemption added since 0.7.0; no `deprecated_alias`
anywhere in `philanthropy/`; `__version__ == importlib.metadata.version(...)`
and `py.typed` in the wheel; every `__all__` symbol carries a tier and no Tier 1
entry is mid-deprecation.

### Documentation
- New page **Real-Data Replication: KDD Cup 1998**
  (`docs/explanation/real_data_replication.md`), promoted out of a section of
  `benchmarks.md` and expanded: synthetic and real numbers side by side, the
  panel construction from the wide promotion history, the pre-registered
  prediction that was wrong by a factor of five, the download caveat, and the
  Zenodo replication DOI. `benchmarks.md` keeps the headline tables and links
  out, so the measured numbers still live in exactly one place. The README
  gains a "Validated on real donor data" section pointing at it.
- "Which estimator do I need?" table at the top of `docs/tutorials/index.md`,
  keyed by the question a fundraising shop actually asks rather than by module.
  Fifteen rows covering every Tier 1 and Tier 2 estimator plus the metrics, with
  the required data shape in the middle column, because that is usually the real
  work. Closes the gap where a reader had to infer the entry point from the
  feature tables.

### Changed
- Issue templates converted from Markdown to **YAML issue forms**
  (`bug_report.yml`, `feature_request.yml`). The Markdown versions asked for a
  version and a reproducer and could be submitted without either, and GitHub's
  community profile reported `issue_template: false` because it counts only
  forms. The bug form requires the version, the environment line and a runnable
  reproducer, plus an explicit tick that the reproducer contains no real donor
  or patient data. The feature form states the two constraints that decide most
  requests (frozen dependency set, no network in the core) before the author
  starts writing. `config.yml` is unchanged.

## [0.7.0] - 2026-08-21

The removal release, plus everything else merged since 0.6.0. Every shim
under Breaking shipped in 0.6.0 emitting a `DeprecationWarning` for one full
published minor; everything under Added, Changed and Deprecated below is new
work that ships for the first time in this release.

### Breaking
- **Four deprecated method aliases removed.** Use the replacement in every case:

  | Removed | Use instead |
  |---|---|
  | `AskAmountRecommender.predict_ask_array` | `ask_ladder` |
  | `ShareOfWalletRegressor.predict_capacity_ratio` | `capacity_ratio` |
  | `MovesManagementClassifier.predict_action_priority` | `action_priority` |
  | `PlannedGivingIntentScorer.predict_bequest_intent_score` | `predict_intent_score` |

- **Three dead constructor parameters removed.** Passing any of them is now a
  `TypeError`: `LapsePredictor(lapse_window_years=...)` (the window is a
  property of how you labelled `y`), `PropensityScorer(estimator=...)` (the
  baseline is a constant 0.5), `FiscalYearGroupedSplitter(fiscal_year_start=...)`
  (`groups` already carries fiscal-year labels).
- **`donor_acquisition_cost`, `cost_per_dollar_raised` and `fundraising_roi` are
  keyword-only.** They do not share an argument order:
  `cost_per_dollar_raised` takes expense first, `fundraising_roi` takes raised
  first, so a positional call was silently accepted and returned a plausible
  wrong number. It is now a `TypeError`.
- **Four accidental second import paths moved behind underscores** so 1.0 does
  not freeze them: `metrics.scoring` → `metrics._scoring`,
  `preprocessing.transformers` → `preprocessing._transformers`,
  `models.propensity` → `models._propensity_baseline`, `utils.testing` →
  `utils._testing`. Every public symbol is unchanged and still exported from its
  subpackage; only a direct `from philanthropy.metrics.scoring import ...`
  breaks. Import from the subpackage instead.
- `philanthropy/utils/_deprecation.py` is gone. `tests/test_deprecations.py`,
  which existed solely to police the shims removed above, went with it, then
  came back later in this same release to police a new one; see Deprecated
  below.

### Changed
- `GratefulPatientFeaturizer`, `EncounterTransformer` and the `philanthropy`
  CLI now reject network-scheme paths (`https://`, `s3://`, `gs://`) with a
  `ValueError` before any file read. Previously the no-network guarantee held
  for the library's own logic but not for its documented public parameters,
  because `pandas` will follow a remote URI if handed one. Local paths,
  including `file://`, are unaffected. Closes #114.
- The no-network promise in `README.md`, `SECURITY.md` and the security review
  Q&A is now stated as two precise guarantees, "never transmits your data" and
  "downloads nothing", instead of the blanket "no network calls of any kind",
  and the second one is machine-checked. `tests/test_no_network.py` now parses
  every module in the package and fails the build if one imports a
  network-capable library without appearing on an explicit allowlist. The
  allowlist is empty, so the effective promise is unchanged and is now enforced
  across modules no test happens to import, rather than only on the paths the
  socket fixture walks.

### Added
- Question 1a in the security review Q&A documents the remote-path rejection,
  which is the behaviour a privacy officer asks about after reading question 1.
- `philanthropy.datasets.fetch_kdd98_donors`, an opt-in fetcher for the KDD Cup
  1998 direct-mail donor dataset, cached locally after first download. It is
  the one entry in the no-network allowlist added above, and it exists so the
  library can be validated against real donor data instead of only synthetic
  data. Part of #124.
- `scripts/real_data_leakage_experiment.py` replicates `leakage_experiment.py`
  on the real KDD Cup 1998 file instead of the synthetic panel. The predicted
  effect (recorded in the script before it was run) was smaller than the
  synthetic numbers; the measured effect is larger: whole-history feature
  construction inflates walk-forward ROC-AUC by +0.376 AUC (versus +0.126
  synthetic), and a random `StratifiedKFold` split overstates the true future
  by +0.107 AUC (versus 0.014-0.030 synthetic). Documented in
  `docs/explanation/benchmarks.md` and in `paper.md`'s Statement of need and
  Research impact statement. The script outputs and environment lock are
  archived on Zenodo (DOI [10.5281/zenodo.22050649](https://doi.org/10.5281/zenodo.22050649)).
  Closes #124.

### Deprecated
- `WealthScreeningImputerKNN(group_col_idx=...)` is **deprecated** and will be
  removed in 0.8.0. It still works and now emits a `DeprecationWarning`. There is
  no replacement because there is nothing to replace: measured across several
  synthetic two-group pools, and on five Python versions in CI, per-group and
  global KNN imputation produce bit-identical output (`50263.48615163204` both
  ways). A donor's nearest neighbours by feature distance almost always share
  their group already, and `KNNImputer` weights distance by column magnitude, so
  a 0/1 group flag barely registers. The parameter costs a per-group imputer,
  three fallback paths and a documented contract, and buys no measurable
  accuracy. Split the frame by group and fit one imputer per part if you need
  that behaviour. `tests/test_deprecations.py` is reintroduced per `RELEASING.md`,
  with the registry meta-test that fails when a shim ships untested; it walks the
  package AST for `warnings.warn(..., DeprecationWarning)` call sites, so a
  docstring merely mentioning the class is not miscounted. Closes #85.

### Added
- `paper.md` now carries the four JOSS sections it was missing: **State of the
  field**, **Software design**, **Research impact statement**, and **AI usage
  disclosure**. JOSS made all six sections required and moved the length window
  to 750-1750 words in January 2026; the paper was 635 words with two of six
  sections, which is a pre-review bounce on its own. It is now 1447 words. The
  AI usage disclosure restates the one already in `README.md` and in
  `philanthropy/__init__.py`, since JOSS requires it as a named section of the
  paper itself.
- `paper.bib` gains the prior art the paper had never cited: feature-engine,
  mlxtend, sktime, pymc-marketing, MAPIE, crepes, Fader-Hardie-Lee (BTYD),
  Zhang (2003) for the linear-plus-nonlinear forecast decomposition, and Bates
  et al. (2023) for the conformal p-value the code already attributes to it.
  JOSS accepts re-implementations "provided that they cite prior similar work",
  and the bibliography previously cited no comparable package.
- `RFMTransformer(include_tenure=True)` emits a fifth column, `tenure`: days
  from the donor's first gift to the frozen reference date. Recency, frequency
  and monetary alone cannot feed a buy-till-you-die model, which needs the
  observation window T as well. Defaults to False so the output shape does not
  move under existing callers.
- `ShareOfWalletScorer(major_tier_threshold=..., principal_tier_threshold=...)`.
  The 0.40 and 0.75 cut points were hardcoded in `transform` with no source and
  no way to match an institution's own tiering.
- `DischargeToSolicitationWindowTransformer(window_shape=...)`, with the legacy
  symmetric triangle available as `"triangle"` for reproducing older runs.
- `EncounterTransformer.fit` warns when `as_of=None` and the encounter table
  contains discharges later than the latest gift date in `X`, naming the row
  count. That is the one leakage path no cross-validation splitter can see: the
  encounter table is a constructor argument, so its rows are never part of any
  split.
- `GratefulPatientFeaturizer` warns when `drg_weight_col` is set. A DRG relative
  weight is diagnosis-derived, and diagnosis is not in the element list the HIPAA
  fundraising carve-out permits (45 CFR 164.514(f)).

### Changed
- **Behaviour change.** `DischargeToSolicitationWindowTransformer` now decays
  `window_position_score` from 1.0 at `min_days_post_discharge` to 0.0 at
  `max_days_post_discharge` instead of peaking at the window midpoint. The old
  symmetric triangle treated the ethical cooling-off floor as a propensity
  minimum: with the default 90-365 window, day 91 and day 364 both scored about
  0.007 while day 227 scored 1.0. Pass `window_shape="triangle"` to reproduce
  the previous numbers.
- **Behaviour change.** A missing days-since-discharge value now yields
  `window_position_score=NaN` rather than `0.0`, so "no discharge on record" is
  distinguishable from "discharged, but outside the window", which still scores a
  hard 0.0. `in_solicitation_window` is unchanged at 0.
- **Behaviour change.** `GratefulPatientFeaturizer(use_capacity_weights=...)`
  now defaults to `False`. The built-in service-line multipliers have no
  published source, and defaulting them on meant the headline
  `clinical_gravity_score` silently carried unsourced 2.7x to 3.2x weighting.
- `EncounterTransformer.dropped_cols_` now includes `gift_date_col`, which
  `transform` drops separately. `compliance_considerations.md` tells operators to
  inspect this attribute as their audit trail, and it was under-reporting what
  actually left.
- `philanthropy.preprocessing.SolicitationWindowTransformer` is deprecated and
  emits a `DeprecationWarning` on access via PEP 562 module `__getattr__`. It
  still resolves to `DischargeToSolicitationWindowTransformer` itself, so
  `isinstance` and `clone` are unaffected, and it is registered in
  `tests/test_deprecations.py` for removal in 1.0.0. Two public names for one
  transformer inflated the API surface without adding capability.
- `CITATION.cff` and `.zenodo.json` now match `paper.md` on title and author
  name, and both carry the ORCID. `CITATION.cff` records `version: 0.6.0`, the
  release the concept DOI actually resolves to, instead of the in-development
  `1.0.0` from `pyproject.toml`. A reviewer following the archive DOI was landing
  on a record that contradicted the paper byline.

### Fixed
- Four claims in `paper.md` that were falsifiable by running the code. The
  conformance claim named `UpliftTLearner` as "the one documented exception"
  against four entries in `_MANUALLY_COVERED`; the leakage claim said a
  PhilanthroPy pipeline "cannot leak test-period or future information" while
  `_encounters.py` documents that exact leak at the default `as_of=None`;
  fiscal-year boundaries were listed as a frozen fitted statistic although
  `FiscalYearTransformer` has no fitted state (its own test is named
  `test_fiscal_year_stateless`); and "compose directly inside
  `sklearn.pipeline.Pipeline`" did not exclude the row-reducing
  `RFMTransformer`. The Summary now describes the conformance registry as the
  mechanism it is: 20 configured instances, 1016 checks on scikit-learn 1.8.0,
  four documented exemptions, and a build-failing guard against a public
  estimator appearing in neither list.
- `conformal_pvalue`: thresholding at `alpha` bounds the expected **selection
  rate**, not the false-positive rate. The FPR reading needs a calibration set
  of nulls only, which is the construction in Bates et al. (2023) and not what
  "donors held out of training" gives you. The wrong statement was in the shipped
  module docstring and therefore in the rendered API docs, not only in the paper.
- The `check_estimator` claim was corrected in `paper.md` but still stood in
  three other places, in its strongest and most falsifiable form: `README.md`
  ("Every public estimator passes `check_estimator`", with the `UpliftTLearner`
  qualification trimmed off at some point), `docs/explanation/design_principles.md`
  ("the one exception"), and `docs/explanation/security_review_answers.md`, which
  is the page written to be forwarded to a privacy or procurement reviewer. All
  three now describe the battery, its four documented exemptions, and the
  build-failing guard, matching the paper.
- `EncounterRecencyTransformer` described itself as producing "HIPAA-safe"
  features in four places. Date-only input is not de-identified: Safe Harbor
  strips every date element more granular than a year, so encounter dates are
  themselves identifiers, permitted for fundraising only under the narrower
  164.514(f) carve-out. This contradicted the project's own compliance page.
- `security_review_answers.md` attributed `PII_PATTERNS` column-dropping to
  `CRMCleaner`, which has neither the attribute nor any dropping logic. It lives
  on `EncounterTransformer`. That page exists to be forwarded to a privacy
  officer, so the error cost more than a docs bug normally would. It now also
  states that `pii_patterns` replaces rather than extends the defaults.
- `ShareOfWalletScorer`'s docstring claimed a share of wallet. The formula has no
  term for giving to your institution anywhere in it, so it cannot express what
  fraction of a donor's philanthropy you receive; it is capacity over modelled
  wealth. `docs/index.md` repeated the wrong definition. The output name is kept
  for compatibility and flagged for renaming in the next major release. The
  fit-time 95th-percentile denominator clip, which inflates exactly the top tier,
  is now documented rather than silent.
- The README figure caption said the affinity scores "cleanly separate major from
  non-major donors", 21 lines below the text explaining that the distributions
  overlap. That was the in-sample overclaim commit c54a6b3 retracted, left behind
  in the caption.
- `AskAmountRecommender` and `ShareOfWalletRegressor` now say in their docstrings
  that they are the same `HistGradientBoostingRegressor` wrapper with different
  targets, and `PropensityScorer` that it is equivalent in effect to
  `DummyClassifier(strategy="uniform")`. Four classes exposed `proba * 100`
  under four names with nothing saying they were the same thing.

### Notes
- `paper.md` cites `scripts/leakage_experiment.py` and its measured result
  (whole-history feature aggregation inflates walk-forward ROC-AUC from 0.625 to
  0.750, +0.126, against 0.014 and 0.030 of splitter-choice error). That script
  arrives with #101, so #101 must land for the reference to resolve. The numbers
  above were reproduced locally against this branch before being written down.
- Still open, and not fixable in a pull request: JOSS requires demonstrated
  research impact, and no estimator here has ever been fitted on real donor
  data. `load_ciob_fundraising` carries no donor rows, amounts or labels. The
  paper's Research impact statement says so plainly rather than implying
  adoption that does not exist.
- `WealthScreeningImputerKNN.group_col_idx` now does what it always claimed.
  It was documented as stratifying KNN imputation per group "improving local
  accuracy", and was stored and never read. When set with `strategy="knn"`, a
  separate `KNNImputer` is now fitted per group, so a donor's missing wealth is
  filled from neighbours inside their own group instead of from the whole
  database. **The measured benefit is small and setup-dependent**, which is worth
  saying plainly given the old docstring promised "improving local accuracy":
  across several synthetic two-group pools the grouped and global fits often
  agree **exactly**, because a donor's nearest neighbours by feature distance
  usually share their group already, and `KNNImputer`'s distance is dominated by
  large-magnitude columns so a 0/1 group flag contributes little either way. CI
  demonstrated this on other numpy/sklearn versions, where the two fills came out
  bit-identical, so no test here asserts that grouping changes a value. The
  honest case for the parameter is explicit control, not a demonstrated accuracy
  gain, and issue #85's option B (deprecate it) remains defensible on that basis. Three fallbacks are frozen at fit time so nothing is learned at transform
  time: a group with fewer than `n_neighbors + 1` training rows gets no imputer of
  its own; a group value unseen at fit, or a row whose group label is missing,
  uses the global imputer; and a column entirely missing *within* a group also
  defers to the global imputer, because
  `KNNImputer(keep_empty_features=True)` fills such a column with a hard `0.0`
  rather than `NaN`, which for a wealth column reads as "no capacity" and would be
  a materially wrong number for every donor in that group. The global imputer is
  always fitted, so output is never `NaN` regardless of grouping. Ignored for the
  columnwise strategies, which have no notion of a neighbourhood. An out-of-range
  index raises, for `strategy="knn"` where the parameter has any effect.
  Closes #85.
- `FiscalYearGroupedSplitter(drop_repeat_donors=True)` for the static-per-donor
  label case. The splitter groups by fiscal year, correctly, but not by donor, so
  a donor with gifts in several fiscal years lands in both folds of a split. That
  is right for a time-varying target and is leakage for a static label such as
  `is_major_donor`, which is the label used throughout the README, the benchmarks
  page and `scripts/benchmark_models.py`. With the flag set, each test fold drops
  donors already present in its training rows; `groups` then takes shape
  `(n_samples, 2)` with the donor identifier in column 1. Training rows are never
  dropped. The cost is made visible rather than silent: `split` warns with the
  number of test rows removed, and notes that the remaining test donors are
  systematically newer to the file. A test fold emptied entirely raises with an
  actionable message rather than being skipped, which would have put `split` and
  `get_n_splits` back out of step. A row with a **missing** donor id is treated as
  already-seen and dropped: `np.isin` never matches `NaN` to `NaN`, so it would
  otherwise have been kept, and an unidentifiable donor cannot be shown to be
  absent from training. A string-typed `groups` (which is what
  `np.column_stack` produces from integer years and string donor ids) has its
  fiscal-year column coerced back to numeric, and a genuinely non-numeric year
  column now raises with an actionable message instead of a bare numpy
  `TypeError`. `__repr__` includes the flag, so two splitters that split
  differently no longer print identically. Defaults to `False`, so nothing
  changes until you opt in. Part of #87; the docs and benchmark follow-up that
  issue also scopes is not done here.
- `scripts/leakage_experiment.py` quantifies the library's central claim, which
  was previously architectural and untested. On a seeded donor-year panel across
  five seeds: walk-forward `FiscalYearGroupedSplitter` estimates the true future
  to within -0.014 ROC-AUC where a random `StratifiedKFold` is off by -0.030, so
  the splitter is worth roughly twice the accuracy in your estimate. Both CV runs
  exclude the year the target scores, because "train on everything earlier, score
  the final year" is what a walk-forward splitter's last fold does and leaving it
  in would hand walk-forward the win by construction. Computing the same aggregate
  features over the whole export instead of as of each panel year inflates the
  score by **+0.126 ROC-AUC**, under an identical model, splitter and label:
  roughly eight times what the splitter choice is worth. Correct feature timing is worth an order of magnitude more than a
  correct splitter, which is the case for freezing fit-time statistics and for the
  new `as_of` cutoff. Reported in `docs/explanation/benchmarks.md`, including the
  negative result: the common claim that a random split *inflates* a backtest did
  not reproduce here in three separate configurations. Closes #84.
- `.gitattributes` sets `CHANGELOG.md merge=union`. `AGENTS.md` requires every PR
  to add an entry under `## [Unreleased]`, so every concurrent PR conflicts with
  every other one, always in the same place and always additively. 10 of the last
  20 commits on `main` touch this file. `union` keeps both sides instead of
  stopping, which is the standard treatment for an append-only file. It is
  line-based rather than section-aware, and it never reports a conflict for this
  file at all: if two branches edit the same entry it keeps both, silently. So the
  `## [Unreleased]` block is worth a skim at release time, for a bullet under the
  wrong heading and for a duplicated one; `RELEASING.md` now says so. GitHub does not read
  `.gitattributes`, so its own merge behaviour is unchanged: the benefit is to the
  local `git merge origin/main` that currently absorbs the cost.
- `philanthropy.metrics.conformal_pvalue`: the non-smoothed split-conformal
- `philanthropy.metrics.conformal_pvalue`: the non-smoothed split-conformal
  p-value of a donor score against a held-out calibration set,
  `(1 + |{i : s_i >= s}|) / (n + 1)`. A calibrated probability threshold fixes no
  error rate; thresholding this p-value at `alpha` bounds the expected selection
  rate at `alpha` in finite samples with no distributional assumption. It is a
  selection-rate bound, not a false-positive rate: the latter reading needs a
  calibration set of nulls only, as in Bates et al. (2023). Both the `1 +`
  and the `+ 1` are load-bearing and tested: the result is never 0 and never
  above 1, and leave-one-out over exchangeable scores lands exactly on the
  uniform lattice.
- `as_of` on `EncounterTransformer` and `GratefulPatientFeaturizer`: an as-of
  cutoff that excludes encounters discharged after a given date from
  `encounter_summary_` at fit time. Without it there was no way to bound the
  encounter table to what was observable at the decision point, so a gift dated
  2020 was featurised from encounters recorded in 2024 and
  `days_since_last_discharge` was measured from the all-time max discharge. The
  failure was systematic rather than random: the more a donor engaged *after* the
  gift, the further the feature was pushed past the gift date and the more often
  it collapsed to `NaN`, destroying it for exactly the donors it should be
  strongest for. Defaults to `None`, which is the previous behaviour, so nothing
  changes until you opt in; set it to the last day of your training window for
  walk-forward evaluation.
- Test coverage for `constituent_events_to_features`: the all-unparseable-timestamps empty-frame path and the `distinct_source_systems` default-to-zero path when `sourceSystem` is absent from the input. (#51)
- `AGENTS.md`: every change, including maintainer- and agent-authored ones, must
  go on a branch and through a PR: no direct commits to `main`, no self-merges.
  go on a branch and through a PR, and never straight to `main`. (The blanket
  "no self-merges" this originally also promised is superseded below: with one
  account holding merge rights it could not hold.)
- `tests/test_no_network.py` enforces in CI what the docs now promise: the package
  makes **no network calls**. Every socket entry point is monkeypatched to raise,
  then a full train/score cycle, an imputation pass and a CiviCRM ingest all run.
  A telemetry hook, HTTP client or lazily downloaded asset added later fails this
  test instead of shipping.
- `docs/explanation/security_review_answers.md`: the ten questions an institutional
  security or privacy review actually asks, on one forwardable page (BAA status,
  dependency provenance, the pickle trust boundary, de-identification scope, bus
  factor, disclosure route).
- `make riskcov`: the risk-tier coverage floor as a single source of truth. `ci.yml`
  and `CONTRIBUTING.md` now both call it.
- `scripts/issue-drafts/_TEMPLATE.md` and `scripts/check_issue_lines.py`: the issue
  shape that converts, and a drift checker for the `path:line` references in issue
  bodies. Deliberately outside `.github/ISSUE_TEMPLATE/`, which is the public chooser.
- Two tests for guards that only fire at transform time and were previously
  uncovered: `MatchingGiftFeaturizer.transform` rejecting a non-DataFrame, and
  `ShareOfWalletScorer` enforcing the `capacity_col_idx` upper bound that `fit`
  deliberately does not check.

### Changed
- Logo: a new mark, an outlined heart crossed by a rising arrow, drawn as SVG so it
  stays crisp at favicon size and follows the colour scheme. `docs/assets/logo.svg`
  is the favicon, `overrides/.icons/philanthropy/heart-rise.svg` is inlined as the
  header logo, and `docs/assets/logo.png` is the regenerated wordmark lockup the
  README uses.
- Homepage figure: the affinity-score distribution is now a chart, not an ASCII dump
  of `describe()`, and it plots the held-out scores the quickstart reports after
  #89, not the in-sample ones. Interquartile bar, median notch, full min-to-max
  range, the overlapping tails left visible, and the 47-point separation between the
  two middle halves called out beside the held-out ROC-AUC of 0.932. The accent marks
  the group being ranked, muted ink the reference group; both fills clear 3:1 on
  their surface in each scheme. Hover gives the five-number summary and a
  collapsible table view carries every number, so nothing is gated behind the
  tooltip.
- Informational admonitions (note, info, tip, abstract, example, quote) now wear the
  palette instead of Material's blue; warning and danger keep their semantic colours.
- Documentation site: a new visual system (Fraunces display serif over Geist,
  a single amber accent, warm near-black canvas with a paper light mode),
  dark scheme first, and a homepage that shows the ten-line quickstart and its
  output above the fold. `mkdocs.yml` also gains section index pages, instant
  navigation, prev/next footer links, footer social links, and a correct
  `edit_uri` (the "edit this page" links previously pointed at a `master`
  branch that does not exist).
- Removed every em dash from the repository's prose, 442 of them across 91 files:
  `paper.md`, `README.md`, all documentation pages, docstrings, inline comments,
  `CHANGELOG.md`, the `Makefile`, `.flake8` and both CI workflows. Each site got
  the punctuation the sentence wanted, picked individually rather than by blanket
  substitution: a colon for a label or definition, commas for an appositive, a
  semicolon for two independent clauses, parentheses for a paired aside. The only
  em dash left is inside a nonprofit's name in
  `philanthropy/datasets/data/ciob_official_fundraising.csv`, which is source
  data rather than prose. No behaviour changes, though a handful of the edited
  sites are user-visible strings rather than prose: the pickle-trust warning in
  `philanthropy/cli.py` and several test comments.
- `AGENTS.md` said "never merge your own PR; open it and leave the merge to
  review" while `.github/CODEOWNERS` is `* @shivamlalakiya` and no second account
  holds merge rights. Taken literally the rule means nothing ever merges, and it
  was visibly not being followed. It now describes what is actually required: a
  PR for every change, green CI before merge, a second reviewer when one is
  available, the maintainer merging their own PR when one is not, and agents never
  merging at all. The section says explicitly that this is a description rather
  than an endorsement, and points at the real fix.
- Four docstrings described behaviour the code does not have, each now corrected
  against a test in `tests/test_documented_contracts.py`. `FiscalYearTransformer`
  said it *appends* `fiscal_year`/`fiscal_quarter`; `transform` in fact returns
  only those two columns and drops the input, which silently discarded a
  pipeline's features. `EncounterTransformer.fit` claimed it "prevents temporal
  data leakage"; it only guarantees that nothing from `X` enters the summary, and
  the summary itself has no as-of cutoff, so a 2020 gift is scored against 2024
  encounters. `WealthScreeningImputerKNN.group_col_idx` documented per-group
  stratified imputation "improving local accuracy"; it is stored and never read.
  The `GratefulPatientFeaturizer` service-line weights were attributed to
  "commonly-cited AMC development benchmarks"; they have no published source.
  No behaviour changed in this entry: the docs moved to meet the code.
- `FiscalYearGroupedSplitter` now documents the leakage it does **not** prevent:
  its grouping unit is the fiscal year, not the donor, so a donor with gifts in
  several fiscal years appears in both folds of a split. That is correct for a
  time-varying target and is leakage for a static per-donor label such as
  `is_major_donor`. The class docstring previously implied it prevented leakage
  generally.
- JOSS paper prep: restored the leakage (`kaufman2012leakage`, `kapoor2023leakage`)
  and grateful-patient-ethics (`collins2018grateful`) citations that the root
  `paper.md` had dropped, so the temporal-leakage claim in the Statement of need
  and the AMC domain claim both have sources again. Settled the author
  affiliation to "Independent Researcher" in `.zenodo.json`, which still said
  "Washington University in St. Louis" and so disagreed with `paper.md`; that
  value is minted into a permanent citable Zenodo record. Deleting the duplicate
  `paper/` draft and repointing `draft-pdf.yml` landed separately in #72.
- Added complete output-column documentation to all eleven preprocessing
  `get_feature_names_out` overrides that previously rendered blank in the API
  reference.
- Documented `fit`/`transform` on `CRMCleaner`, `FiscalYearTransformer` and
  `WealthPercentileTransformer`, including which attributes each `fit` freezes and
  that `WealthPercentileTransformer` ranks held-out rows against the frozen training
  distribution rather than the batch being transformed.
- Documented `fit`, `predict`, `predict_proba` and `predict_affinity_score` on
  `MovesManagementClassifier`, `PlannedGivingIntentScorer`, `MajorGiftClassifier` and
  `PropensityScorer`: the fitted attributes each sets, and that `PropensityScorer`'s
  default threshold returns `classes_[0]` for every row.
- `make_donor_dataset` now documents that it returns a **gift-level** frame, so
  `len(df) > n_donors` (each donor contributes 1–5 rows), and that
  `fiscal_year_start` and `lapse_rate` are accepted but currently unused.
- `CLAUDE.md` is now `AGENTS.md`, the tool-neutral convention, with `CLAUDE.md`
  reduced to an `@AGENTS.md` import. Agents other than Claude Code were reading no
  project instructions at all: not the leakage contract, not the dependency
  constraint, not `make ci`.
- `README.md` quickstart prints a result instead of ending in a bare `assert`, and
  documents the CLI path for readers who do not write Python.
- `SECURITY.md` supported-versions table named `0.5.x`, which has not been the
  installable release since `0.6.0`. Now `0.6.x`, and kept current by the
  `RELEASING.md` checklist. Adds GitHub private vulnerability reporting as the
  preferred disclosure channel.
- `AGENTS.md`'s merging section no longer bars agents from merging outright.
  An agent may now merge a PR under the same bar as the maintainer's own-PR
  merge (all required CI checks green, no second reviewer available), plus
  having actually read the diff and judged it good.
- `.gitignore` now excludes `.claude/CLAUDE.local.md`, for personal working
  notes that shouldn't end up in the repo.

### Fixed
- Version metadata now names the release that actually exists. `pyproject.toml`
  and `CITATION.cff` both declared `1.0.0`, which has no git tag, no PyPI
  artifact and no Zenodo deposit; PyPI's newest is `0.6.0` and so is the newest
  tag. `CITATION.cff` additionally dated that phantom 1.0.0 to `2026-08-01`,
  which is 0.6.0's release date, and its DOI comment cited the v0.6.0
  per-version DOI while the file claimed 1.0.0. Both now say `0.6.0`, and the
  comment states the rule: `version` tracks the newest **published** release,
  not `main`. `main` continues to carry unreleased `0.7.0` and `1.0.0` work, and
  those CHANGELOG headings stay `- TBD` until a release is cut. This also
  unblocks the JOSS archive step, which requires the submitted version to
  correspond to a real tagged, archived release. `RELEASING.md`'s "cutting a
  release `main` has already moved past" section assumed the tip carried the
  newest staged version and walked through tagging an older commit; with the
  tip back at the published version the normal branch-bump-date-tag path
  applies, so that section documents that instead and keeps the older-commit
  case as the caveat it is. Closes #88.
- **`generate_synthetic_donor_data` ran the domain's causal arrow backwards.**
  It drew `is_major_donor` from a logistic model of `years_active` and
  `event_attendance_count`, then drew `total_gift_amount` *conditional on that
  label*, so the strongest feature was generated from the answer. Measurably: a
  model given `total_gift_amount` scored ROC-AUC 0.935 against a causal Bayes
  accuracy ceiling of 0.768, beating the Bayes rate of the generator's own
  process by about 19 AUC points, which no model can legitimately do. Using
  cumulative lifetime giving to predict "is a major donor" is also the classic
  fundraising leakage this library exists to prevent, so the reference dataset
  was teaching the anti-pattern. `last_gift_date` was a second target-derived
  feature, drawn Beta for majors and uniform for everyone else.
  A latent giving capacity now drives everything: a confounder causing both the
  giving history and the label. `total_gift_amount` is a noisy realisation of
  capacity, `is_major_donor` a soft $25,000 threshold on it, and
  `last_gift_date` follows engagement. The model now sits **below** the ceiling
  (accuracy 0.759 against 0.806) rather than above it, which is the correct
  relationship. Held-out ROC-AUC moves from 0.935 to 0.814 and the base rate from
  0.687 to 0.378: worse numbers, trustworthy ones. Benchmark table, README
  quickstart and the benchmarks page are regenerated from the committed script.
  `generate_synthetic_donor_data` is Tier 1, so **this changes returned data for
  a documented-stable function**; release sequencing is the open version
  question. Closes #86.
- The README quickstart fitted and scored **the same rows**, then reported the
  resulting gap ("non-major donors top out at 39; no major donor scores below 65")
  as the headline result. That gap was a random forest reciting its training set:
  RF leaves go pure and `predict_affinity_score` is
  `predict_proba(X)[:, 1] * 100`. On held-out rows from the same 500-row sample
  the two groups overlap almost completely (non-major max 97.5, major min 6.5).
  The quickstart now splits before fitting and reports held-out ROC-AUC 0.932
  with overlapping score distributions, which is a weaker claim and a true one.
- `docs/explanation/benchmarks.md` distrusted its own numbers for the wrong
  reason. It said the synthetic data was "cleanly separable by construction"; the
  label is a Bernoulli draw with a real noise term and the irreducible error over
  the causal features is 23.2%. The actual problem is that the generator draws
  `total_gift_amount` **from** the label, so including that feature lets a model
  score ROC-AUC 0.935 against a causal Bayes ceiling of 0.768 accuracy: it beats
  the Bayes rate of its own data-generating process by about 19 AUC points, which
  is the signature of a target-derived feature. The page now measures and states
  this, and records that there is no validation on real donor data anywhere in the
  repository.
- Two `EncounterTransformer` output columns were documented with the wrong type
  and the wrong semantics. `days_since_last_discharge` was described as an
  "Integer number of days"; it is `float64`, and it has to be, because a donor
  absent from the encounter table gets `NaN` and an integer dtype cannot carry
  that. A caller who trusted the docstring and cast the column would silently
  destroy the missingness, which is signal in this library.
  `encounter_frequency_score` was described as a "Log-scaled count of distinct
  encounter records"; it is `log1p` of the **row** count, so a donor with three
  rows on two dates scores `log1p(3)`, not `log1p(2)`. Both are now stated
  correctly and locked by tests in `tests/test_documented_contracts.py`. Found
  during review of the em-dash branch; docstrings only, no behaviour change.
- `LapsePredictor` and `experimental.UpliftTLearner` validated input with
  `check_array`/`check_X_y` instead of `validate_data`, the convention every
  other estimator follows. Neither set `feature_names_in_`, so a DataFrame with
  reordered columns was silently scored instead of raising. These were the only
  two estimators in the package with that gap, and both are now closed with a
  regression test each.
- `CRMCleaner` NaN'd every value in a currency-formatted amount column, e.g.
  `"$1,000.00"` (the default export format for Raiser's Edge NXT and
  Salesforce NPSP), because `pd.to_numeric` treats the whole string as
  unparseable. It now strips currency symbols, thousands separators and
  parenthesised negatives before parsing, and raises rather than returning an
  all-NaN column when a column truly has nothing parseable in it. The
  string/numeric branch checks `pd.api.types.is_numeric_dtype` rather than
  `dtype == object`, so it also parses correctly under pandas 3.0's non-object
  default string dtype, not just the legacy `object` dtype.
- `MatchingGiftFeaturizer` ran zero `check_estimator` checks: `tags._skip_test =
  True` silently skipped the whole battery instead of excluding it from
  `_STANDARD_ESTIMATORS` with a documented reason, the way `RFMTransformer`
  already was. It has no such reason on its own (it genuinely cannot accept
  the generic numeric ndarrays the battery feeds), so this falsified the
  README/paper claim that every public estimator passes `check_estimator`.
  `FinancialForecastModel` had the same gap for no documented reason at all;
  it in fact passes the battery cleanly and is now in it. A new
  `test_every_public_estimator_is_covered_by_the_battery_or_documented` test
  cross-references `philanthropy.models.__all__` and
  `philanthropy.preprocessing.__all__` against `_STANDARD_ESTIMATORS` plus a
  reasoned exemption registry, so this can't recur silently. README, paper.md,
  and the design-principles/security-review docs now state the one real
  exception (`UpliftTLearner`) instead of claiming "every estimator" flatly.
- Two JOSS paper drafts were tracked at once: `paper.md`/`paper.bib` at the repo
  root (current, last touched 2026-08-11) and a stale copy in `paper/`
  (2026-08-01, different affiliation and bibliography style). `draft-pdf.yml`
  built only the stale one, so the current draft has never produced a PDF.
  Deleted `paper/`; the workflow now points at the root files.
- Saving a fitted `EncounterTransformer` or `GratefulPatientFeaturizer` wrote the
  **raw clinical encounter table into the model bundle**. Both take
  `encounter_df` as a constructor parameter, so `joblib.dump` / `save_model`
  persisted medical record numbers, attending physicians and service lines
  verbatim; a bundle attached to a ticket or handed to a vendor was a PHI
  disclosure. Both now drop the raw table on serialisation and keep only the
  per-donor `encounter_summary_` that `transform` actually reads, so a
  round-tripped transformer still scores identically. `clone` is unaffected
  (it goes through `get_params`, not pickle), and a refit now requires the table
  to be supplied again rather than reusing stale clinical rows. `SECURITY.md`
  previously treated pickles only as an inbound code-execution risk and never
  mentioned that a bundle you produce is itself donor data; it now does.
- `FiscalYearGroupedSplitter` never validated `n_splits`, despite documenting a
  `ValueError` for `n_splits < 1`. A non-positive value reached the
  `unique_fy[-(n_splits):]` slice, where it flips open-ended: `n_splits=0`
  yielded 3 folds on a 4-fiscal-year panel while `get_n_splits()` reported 0, and
  `n_splits=-1` yielded 3 while reporting -1. `cross_val_score` sizes its result
  array from `get_n_splits()`, so the two disagreeing is a real failure. Both
  entry points now validate through one helper, `gap_years < 0` and non-integer
  values are rejected, and a test asserts `get_n_splits() == len(list(split()))`
  across the parameter grid.
- **`donor_lifetime_value` overstated LTV whenever `retention_rate` was given.**
  It converted the retention rate to an expected lifespan, `L = 1 / (1 - r)`, and
  fed that mean into the concave annuity formula. By Jensen's inequality
  `NPV(E[L]) >= E[NPV(L)]`, so the result was biased high in one direction every
  time: +8.2% at `r = 0.8, d = 0.05` and +22.9% at `r = 0.9, d = 0.10`. A
  one-signed error does not average out across a portfolio, and this is a number
  that goes into board decks and acquisition-cost justifications. The retention
  branch now uses the correct closed form for a geometric lifetime,
  `E[NPV] = m / (1 + d - r)`, verified against a term-by-term expectation and a
  two-million-draw Monte Carlo. `retention_rate=1.0` with a positive discount
  rate now returns the perpetuity `m / d` rather than `inf`; it is still `inf`
  when `discount_rate` is 0. `retention_rate > 1` now raises instead of returning
  a negative number. The fixed-horizon path (`retention_rate=None`) is unchanged
  and was always correct, as is the `discount_rate=0` path in both modes, since an
  undiscounted sum is linear in the lifespan. **This changes returned values**:
  see `docs/explanation/fundraising_metrics.md` for both formulas and why they
  differ.
- `mkdocs.yml` had no `site_url`, so the generated `sitemap.xml` was empty and all
  38 documentation pages were uncrawlable, with no `rel=canonical` anywhere.
- `CONTRIBUTING.md` documented a risk-tier coverage command measuring `metrics/` and
  `model_selection/`, while CI measured `ingest/`, `cli.py` and `utils/_persistence.py`.
  A contributor could run the documented command, pass, and still fail CI on files it
  never looked at.

### Removed
- `FiscalYearGroupedSplitter._iter_test_indices` and `_iter_test_masks`. Both were
  unreachable (the class overrides `split`, so `cross_validate` never called
  either) and the comment claiming `BaseCrossValidator` requires them was false.

### Fixed
- `FiscalYearGroupedSplitter`'s module doctest asserted
  `... <= ... + 1 or True`, which passes for every possible input and so proved
  nothing about the split. It now asserts what the class actually promises,
  `fiscal_years[train_idx].max() < fiscal_years[test_idx].min()`, and a new
  `test_default_splitter_no_leakage_gap_years_zero` covers the default
  `gap_years=0` path, which had no leakage test at all. Thanks to
  [@fuleinist](https://github.com/fuleinist) (Chris Chen) for the first external
  contribution ([#30](https://github.com/PhilanthroPy-Project/PhilanthroPy/pull/30),
  closes [#26](https://github.com/PhilanthroPy-Project/PhilanthroPy/issues/26)).

### Added
- **CiviCRM contribution bridge**: `philanthropy.ingest.read_civicrm_contributions`
  and `civicrm_contributions_to_features` (Tier 2). Turns a CiviCRM contribution
  export, or an APIv4 `Contribution.get` result, into the one-row-per-donor
  feature table the estimators consume. Headers normalise to the APIv4 spelling,
  so the human export labels (`Contact ID`, `Total Amount`, `Contribution Date`)
  and the DB columns (`contact_id`, `total_amount`, `receive_date`) both work.

  It exists because two things a bare `pd.read_csv` gets wrong are expensive:
  CiviCRM writes payment-processor **test transactions** into the same table, and
  `contribution_status` separates `Completed` from `Pending`, `Failed`,
  `Refunded` and `Chargeback`. Test rows are always dropped; only `Completed` is
  counted unless `statuses` says otherwise, and asking for a status filter that
  cannot be applied warns instead of silently summing refunds.

  Recency is anchored to `reference_date` or the batch's latest gift, never a
  moving "now", the same leakage contract as the UniSchema bridge. See
  [docs/how-to/ingest_civicrm_contributions.md](docs/how-to/ingest_civicrm_contributions.md).

## [0.6.0] - 2026-08-01

### Breaking
- `pandas>=2.0` is now the declared floor (was `>=1.5`). The ingest bridge pins
  `format="ISO8601"`, which is pandas 2.0+; on a conforming 1.5.x install
  `errors="coerce"` silently produced an all-NaT, zero-row feature frame. The
  build backend floor moves to `setuptools>=77` for the PEP 639 license fields.
- `DischargeToSolicitationWindowTransformer.transform` now raises `ValueError`
  when given a DataFrame without `days_since_discharge_col`, instead of reading
  `X.iloc[:, 0]`. That fallback made a serial `Pipeline` behind
  `FiscalYearTransformer` score every donor `0.0` and exit cleanly. Route the
  transformer with a `ColumnTransformer`; see
  `docs/tutorials/building_your_first_model.md` for the migration.
- `philanthropy.experimental.LapsePredictor` is **removed**. It collided by name
  with `philanthropy.models.LapsePredictor` and took different positional
  arguments. Tier 3, so no deprecation runway. Use `models.LapsePredictor`.
- `PropensityScorer.predict` now uses a strict threshold comparison
  (`proba > threshold`), flipping its default-threshold prediction from class 1
  to class 0. scikit-learn requires `argmax(predict_proba) == predict`, and
  `argmax` of a tied `[0.5, 0.5]` row is index 0. Arbitrary either way for a
  constant scorer; only its ROC-AUC of 0.500 ever carried information.
- `PropensityScorer.fit` now raises `ValueError` on a multiclass `y`.

### Added
- `philanthropy.model_selection`, `.experimental` and `.visualisation` are now
  importable from `import philanthropy` and listed in `__all__`; they raised
  `AttributeError` before while the docs rendered reference pages for them.
- `philanthropy/py.typed`: the package now ships its type information.
- `joblib>=1.2` is declared; it was a direct import carried transitively.
- `tests/test_public_api_contract.py`: an executable spec for the public API:
  subpackage `__all__` completeness, reference-page coverage, the
  `predict_<thing>_(score|forecast)` naming and shape contract, and
  `get_feature_names_out` width. Two named exemptions, each with a reason.
- `tests/test_metrics_oracles.py`: closed-form oracles for the money metrics:
  the textbook Gini definition, a term-by-term discounted annuity, and the
  EEOC four-fifths worked example.
- Seven how-to guides: use the CLI, ingest UniSchema events, recommend ask
  amounts, score matching-gift eligibility, measure campaign efficiency, audit
  score fairness, estimate appeal uplift. Reference pages for `experimental`,
  `utils` and `cli`. A stability-tier and score-scale table in
  `docs/reference/index.md`.
- `.zenodo.json` and a concept-DOI placeholder in `CITATION.cff`.

### Deprecated
All of the following still work and emit `DeprecationWarning`. **Removed in
0.7.0.**

| Deprecated | Use instead |
|---|---|
| `AskAmountRecommender.predict_ask_array` | `ask_ladder` |
| `ShareOfWalletRegressor.predict_capacity_ratio` | `capacity_ratio` |
| `MovesManagementClassifier.predict_action_priority` | `action_priority` |
| `PlannedGivingIntentScorer.predict_bequest_intent_score` | `predict_intent_score` |

The `predict_` prefix is now reserved for methods that take X alone and return
one value per row. The first three returned a `(n, 3)` dollar matrix, required a
second argument, and returned a dict respectively.

Three constructor parameters have no effect and warn when set to a non-default
value: `LapsePredictor(lapse_window_years=...)`,
`PropensityScorer(estimator=...)`,
`FiscalYearGroupedSplitter(fiscal_year_start=...)`. All removed in 0.7.0.

### Fixed
- `philanthropy.__version__` is read from installed metadata. It reported
  `0.4.0` against a `0.5.0` package, and every bundle written by `save_model`
  carried the wrong stamp.
- `MajorGiftClassifier.n_iter_` reports the real mean boosting iterations across
  the calibration folds instead of a hardcoded `1`.
- `GratefulPatientFeaturizer.transform` emits a `UserWarning` before each
  all-zero fallback instead of silently returning `zeros((n, 4))`.
- `philanthropy train --features " , "` now exits with an error instead of
  fitting on a zero-column matrix.
- `read_constituent_events` raises `FileNotFoundError` for a missing path
  instead of surfacing an opaque OSError from the single-file branch.
- `WealthScreeningImputer` no longer emits a `Mean of empty slice`
  `RuntimeWarning` on an all-NaN column.
- `CRMCleaner.fiscal_year_start` is documented as validated-but-unused.
- Doc corrections: `ShareOfWalletScorer` capacity-tier thresholds, PHI-dropping
  attributed to `EncounterTransformer` rather than `CRMCleaner`, and the
  `GratefulPatientFeaturizer` output columns.

### Changed
- The `check_estimator` battery is consolidated into one list in
  `tests/test_sklearn_compliance.py`. `MajorGiftClassifier` runs at
  `max_iter=10`, cutting suite runtime by roughly two thirds;
  `PropensityScorer`, `WealthPercentileTransformer`,
  `WealthScreeningImputerKNN`, `ShareOfWalletScorer`, `CRMCleaner`,
  `EncounterRecencyTransformer` and bare-default variants were added.
  `RFMTransformer` moved to an explicit contract class: its `_skip_test=True`
  tag was running 1 check instead of 46.
- Branch coverage is enabled and gated; the risk-tier subtree has its own floor.
- `docs/explanation/benchmarks.md` reports mean and min–max across five seeds
  instead of three decimals from one split.
- CI: the duplicate full-suite run is gone, lint runs once instead of five
  times, and there are new `floors` (lowest-direct dependency resolution),
  `package`, and `minimal` (no-matplotlib import) jobs plus a macOS leg.
- `publish.yml` gates on tag ↔ `pyproject.toml` ↔ `CHANGELOG.md` agreement, and
  both third-party actions are SHA-pinned.

## [0.5.0] - 2026-07-24
### Added
- Donor-base concentration metrics: `gift_concentration_gini` and
  `top_donor_share` (`philanthropy.metrics`).
- Campaign-efficiency metrics: `cost_per_dollar_raised` and `fundraising_roi`.
- `philanthropy.utils.save_model` / `load_model`: self-describing model
  bundles that warn on a scikit-learn / PhilanthroPy version mismatch at load
  time; the CLI now persists and loads through them.
- `AskAmountRecommender`: capacity regressor exposing a discrete gift-array
  ask ladder (`predict_ask_array`).
- `MatchingGiftFeaturizer`: corporate matching-gift features (`has_employer`,
  `match_ratio`, `potential_matched_amount`), leakage-safe.
- `philanthropy.experimental.UpliftTLearner`: two-model uplift (treatment-
  effect) scorer for appeals (`predict_uplift_score`).
- `MovesManagementClassifier` is now covered by the `check_estimator` battery.
- Dependabot (`github-actions` + `pip`), a CodeQL scanning workflow, and a
  `.pre-commit-config.yaml` running the flake8 / mypy gates locally.
- `philanthropy.ingest` and `philanthropy.visualisation` API reference pages.
- `constituent_events_to_features` carries `first_name` / `last_name` through to
  the donor feature table when the UniSchema feed supplies them (guarded; null
  when absent).
- Community health files: `.github` issue/PR templates,
  `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), and `SECURITY.md`.
- flake8 lint gate: `.flake8` (enforces pyflakes `F` + syntax `E9` defects),
  a `make lint` target folded into `make ci`, and a CI step.
- `philanthropy.inspection.donor_feature_importance`: model-agnostic permutation
  feature importance (dependency-free interpretability; works on calibrated models
  that lack `feature_importances_`).
- `philanthropy.metrics.disparate_impact_ratio` and `selection_rate_by_group`:
  four-fifths-rule fairness diagnostics for scored cohorts.
- `philanthropy` command-line interface (`train` / `score` / `validate`) over CSV.
- `EncounterTransformer(pii_patterns=...)` to override the PII column heuristic
  (defaults broadened); `allow_negative_days=True` now emits a compliance `UserWarning`.
- `GratefulPatientFeaturizer(capacity_weights=...)` to override service-line weights.
- mypy type-check gate wired into `make ci` and CI.
- Docs: Responsible Use & Compliance, Model Validation & Benchmarks, vendor
  comparison, and model-persistence guides; `.github/CODEOWNERS`; a JOSS `paper/`.

### Security
- CLI `score` neutralizes spreadsheet formula-injection (CWE-1236) in
  donor-controlled string cells before writing the output CSV.
- Documented the model-bundle pickle trust boundary in `SECURITY.md` and the
  `score` / `validate` `--help` text.
- `read_constituent_events` skips symlinked files (path-traversal hardening)
  and reports malformed JSON with the offending file and line number.
- Least-privilege `permissions: contents: read` on all GitHub Actions
  workflows.

### Fixed
- `RFMTransformer` freezes the recency reference date in `fit`
  (`reference_date_`) instead of recomputing it from the transform batch, a
  leakage-contract violation that made a donor's recency depend on batchmates.
- `LapsePredictor.predict_lapse_score` no longer raises `IndexError` when fit
  on a single-class training fold.
- `MovesManagementClassifier` rejects continuous targets and exposes `n_iter_`.
- `disparate_impact_ratio` / `selection_rate_by_group` raise on missing
  (`NaN`/`None`) group labels instead of silently returning `NaN`.
- Removed dead code (`_assign_tier` / `_TIER_THRESHOLDS`, `_resolve_cols`,
  redundant `_more_tags`) and cleared 4 pre-existing type-check errors.
- Cleared 31 real-defect lint violations (unused imports/variables) across the
  package and tests, including two dead code blocks.
- Corrected `EncounterTransformer` API drift in the grateful-patient tutorial and
  the README example (invalid `encounter_date_col` / `donor_id_col` kwargs →
  `discharge_col` / `merge_key`; pipeline scored via `predict_proba`).
- README metrics table listed `retention_rate`; the real export is
  `donor_retention_rate`.
- Removed a documented-but-nonexistent `fiscal_year_start` parameter from the
  `EncounterTransformer` and `WealthScreeningImputer` docstrings.

## [0.4.0] - 2026-07-18
### Added
- `philanthropy.ingest`: the UniSchema on-ramp. `constituent_events_to_features()`
  aggregates a UniSchema `ConstituentEvent` stream into a one-row-per-donor
  feature table whose columns (`total_gift_amount`, `years_active`,
  `event_attendance_count`, `last_gift_date`, ...) feed the estimators directly;
  `read_constituent_events()` loads UniSchema's JSON / NDJSON egress files.
  Leakage-safe (recency anchored to an explicit `reference_date` or the batch's
  latest event), at-least-once-safe (deduplicates by `eventId`).
- `constituent_events_to_features` and `read_constituent_events` re-exported at
  the top level (`from philanthropy import constituent_events_to_features`).
- `examples/quickstart.py` and `examples/unischema_to_scores.py`: runnable,
  end-to-end scripts (train + score; UniSchema `ConstituentEvent` stream →
  features → score). Smoke-tested in `tests/test_examples.py`.
- tests/test_ingest.py (aggregation, identity resolution, dedup, file/dir
  readers, mixed-currency warning, estimator integration)

### Fixed
- Pinned `scikit-learn>=1.6`; the code relies on `validate_data` and
  `__sklearn_tags__`, both 1.6+ APIs, so an unpinned install on 1.3–1.5
  imported broken.
- `MovesManagementClassifier` now imports on Python 3.9 (added
  `from __future__ import annotations`; its `str | dict | None` annotation
  was evaluated eagerly and crashed the advertised 3.9).
- Removed the nonexistent `philanthropy==0.2.0` pin from `environment.yml`
  that made `conda env create` fail.
- `constituent_events_to_features` warns on a mixed-currency batch instead of
  silently summing unlike amounts into `total_gift_amount`.
- `EncounterRecencyTransformer` no longer raises `OverflowError` when two
  encounter dates span more than ~292 years (a `datetime64[ns]` timedelta
  overflows int64); it falls back to day-resolution differencing.

### Changed
- README leads installation with `pip install philanthropy`; fixed the Tests
  badge and the UniSchema scoring snippet.
- Sharpened the PyPI `description`, added `machine-learning` /
  `predictive-analytics` / `data-science` / `python` keywords, and added the
  UniSchema project URL (pyproject + CITATION.cff).
- README roadmap corrected (docs site, PyPI, and retention-waterfall plot moved
  to Completed); dropped the stale per-file test table; ingest docs/example now
  point at UniSchema's real `data/egress/` path.
- `PropensityScorer` documented as a constant P=0.5 baseline (points to
  `DonorPropensityModel`); added docstrings for the metrics helpers and
  `predict_action_priority`; `CONTRIBUTING.md` gained a Setup section.

## [0.3.0] - 2026-07-17
### Added
- FinancialForecastModel: hybrid LSTM-ARIMA revenue/giving forecaster
  (linear ARIMA-surrogate + neural residual component) with
  `predict_revenue_forecast(X, horizon)`; leakage-safe: fill values and
  autoregressive coefficients frozen at `fit()`; passes sklearn
  `check_estimator`
- tests/test_forecast_model.py (fit/predict, forecast horizon, leakage,
  NaN handling, check_estimator compliance)
- PyPI packaging: complete project metadata, classifiers, keywords, and
  project URLs (docs / repo / changelog / issues); version bumped to 0.3.0
- MANIFEST.in so the sdist ships source only (no tests/dev artifacts)
- PyPI Trusted Publishing workflow (.github/workflows/publish.yml): OIDC,
  no stored token, fires on published GitHub Releases (v*.*.*)
- CONTRIBUTING.md split out of the README
- CITATION.cff for Zenodo/DOI archival
- README "Research" section mapping the literature to concrete estimators,
  and an affinity-distribution visual

## [0.2.0] - 2026-03-14
### Added
- GitHub Actions CI workflow (Python 3.10 + 3.11 matrix)
- Coverage gate: pytest --cov-fail-under=85
- Makefile with check / test / coverage / ci targets
- Branch protection + PR-based merge workflow
- DischargeToSolicitationWindowTransformer (2-column output: in_window, window_position_score)
- PlannedGivingIntentScorer with predict_intent_score()
- LapsePredictor: production RF, predict_lapse_score(), full param set
- 1052 tests across 23 test files (up from 161 across 7)
- Coverage: 88.29%

### Fixed
- SolicitationWindowTransformer.transform() now returns (n, 2) not (n, 3)
- Removed contradictory test_output_shape_is_n_by_3
- InvalidParameterError accepted alongside ValueError (sklearn 1.6+ compat)
- check_do_not_raise_errors_in_init_or_set_params: validation moved to fit()
- Hypothesis tests stabilised with @settings(suppress_health_check=...)

## [0.1.0] - 2026-01-01
### Added
- Initial release: DonorPropensityModel, ShareOfWalletRegressor,
  MajorGiftClassifier, CRMCleaner, WealthScreeningImputer,
  FiscalYearTransformer, EncounterTransformer, RFMTransformer
- philanthropy.metrics: donor_retention_rate, donor_acquisition_cost,
  donor_lifetime_value
- philanthropy.visualisation: plot_affinity_distribution
- philanthropy.utils: make_donor_dataset
- 161 tests across 7 test files
