# Changelog

All notable changes to PhilanthroPy are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

## [Unreleased]

### Added
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
  `get_n_splits` back out of step. Defaults to `False`, so nothing changes until
  you opt in. Closes #87.
- `philanthropy.metrics.conformal_pvalue` — the non-smoothed split-conformal
  p-value of a donor score against a held-out calibration set,
  `(1 + |{i : s_i >= s}|) / (n + 1)`. A calibrated probability threshold fixes no
  error rate; thresholding this p-value at `alpha` bounds the false-positive rate
  at `alpha` in finite samples with no distributional assumption. Both the `1 +`
  and the `+ 1` are load-bearing and tested: the result is never 0 and never
  above 1, and leave-one-out over exchangeable scores lands exactly on the
  uniform lattice.
- Test coverage for `constituent_events_to_features`: the all-unparseable-timestamps empty-frame path and the `distinct_source_systems` default-to-zero path when `sourceSystem` is absent from the input. (#51)
- `AGENTS.md`: every change, including maintainer- and agent-authored ones, must
  go on a branch and through a PR — no direct commits to `main`, no self-merges.
- `tests/test_no_network.py` enforces in CI what the docs now promise: the package
  makes **no network calls**. Every socket entry point is monkeypatched to raise,
  then a full train/score cycle, an imputation pass and a CiviCRM ingest all run.
  A telemetry hook, HTTP client or lazily downloaded asset added later fails this
  test instead of shipping.
- `docs/explanation/security_review_answers.md` — the ten questions an institutional
  security or privacy review actually asks, on one forwardable page (BAA status,
  dependency provenance, the pickle trust boundary, de-identification scope, bus
  factor, disclosure route).
- `make riskcov` — the risk-tier coverage floor as a single source of truth. `ci.yml`
  and `CONTRIBUTING.md` now both call it.
- `scripts/issue-drafts/_TEMPLATE.md` and `scripts/check_issue_lines.py` — the issue
  shape that converts, and a drift checker for the `path:line` references in issue
  bodies. Deliberately outside `.github/ISSUE_TEMPLATE/`, which is the public chooser.
- Two tests for guards that only fire at transform time and were previously
  uncovered: `MatchingGiftFeaturizer.transform` rejecting a non-DataFrame, and
  `ShareOfWalletScorer` enforcing the `capacity_col_idx` upper bound that `fit`
  deliberately does not check.

### Changed
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
- Added complete output-column documentation to all eleven preprocessing
  `get_feature_names_out` overrides that previously rendered blank in the API
  reference.
- Documented `fit`/`transform` on `CRMCleaner`, `FiscalYearTransformer` and
  `WealthPercentileTransformer`, including which attributes each `fit` freezes and
  that `WealthPercentileTransformer` ranks held-out rows against the frozen training
  distribution rather than the batch being transformed.
- Documented `fit`, `predict`, `predict_proba` and `predict_affinity_score` on
  `MovesManagementClassifier`, `PlannedGivingIntentScorer`, `MajorGiftClassifier` and
  `PropensityScorer` — the fitted attributes each sets, and that `PropensityScorer`'s
  default threshold returns `classes_[0]` for every row.
- `make_donor_dataset` now documents that it returns a **gift-level** frame, so
  `len(df) > n_donors` (each donor contributes 1–5 rows), and that
  `fiscal_year_start` and `lapse_rate` are accepted but currently unused.
- `CLAUDE.md` is now `AGENTS.md`, the tool-neutral convention, with `CLAUDE.md`
  reduced to an `@AGENTS.md` import. Agents other than Claude Code were reading no
  project instructions at all — not the leakage contract, not the dependency
  constraint, not `make ci`.
- `README.md` quickstart prints a result instead of ending in a bare `assert`, and
  documents the CLI path for readers who do not write Python.
- `SECURITY.md` supported-versions table named `0.5.x`, which has not been the
  installable release since `0.6.0`. Now `0.6.x`, and kept current by the
  `RELEASING.md` checklist. Adds GitHub private vulnerability reporting as the
  preferred disclosure channel.

### Fixed
- `LapsePredictor` and `experimental.UpliftTLearner` validated input with
  `check_array`/`check_X_y` instead of `validate_data`, the convention every
  other estimator follows. Neither set `feature_names_in_`, so a DataFrame with
  reordered columns was silently scored instead of raising. These were the only
  two estimators in the package with that gap, and both are now closed with a
  regression test each.
- `CRMCleaner` NaN'd every value in a currency-formatted amount column, e.g.
  `"$1,000.00"` — the default export format for Raiser's Edge NXT and
  Salesforce NPSP — because `pd.to_numeric` treats the whole string as
  unparseable. It now strips currency symbols, thousands separators and
  parenthesised negatives before parsing, and raises rather than returning an
  all-NaN column when a column truly has nothing parseable in it. The
  string/numeric branch checks `pd.api.types.is_numeric_dtype` rather than
  `dtype == object`, so it also parses correctly under pandas 3.0's non-object
  default string dtype, not just the legacy `object` dtype.
- `MatchingGiftFeaturizer` ran zero `check_estimator` checks — `tags._skip_test =
  True` silently skipped the whole battery instead of excluding it from
  `_STANDARD_ESTIMATORS` with a documented reason, the way `RFMTransformer`
  already was. It has no such reason on its own (it genuinely cannot accept
  the generic numeric ndarrays the battery feeds), so this falsified the
  README/paper claim that every public estimator passes `check_estimator`.
  `FinancialForecastModel` had the same gap for no documented reason at all —
  it in fact passes the battery cleanly and is now in it. A new
  `test_every_public_estimator_is_covered_by_the_battery_or_documented` test
  cross-references `philanthropy.models.__all__` and
  `philanthropy.preprocessing.__all__` against `_STANDARD_ESTIMATORS` plus a
  reasoned exemption registry, so this can't recur silently. README, paper.md,
  and the design-principles/security-review docs now state the one real
  exception (`UpliftTLearner`) instead of claiming "every estimator" flatly.
- Two JOSS paper drafts were tracked at once — `paper.md`/`paper.bib` at the repo
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
- `mkdocs.yml` had no `site_url`, so the generated `sitemap.xml` was empty and all
  38 documentation pages were uncrawlable, with no `rel=canonical` anywhere.
- `CONTRIBUTING.md` documented a risk-tier coverage command measuring `metrics/` and
  `model_selection/`, while CI measured `ingest/`, `cli.py` and `utils/_persistence.py`.
  A contributor could run the documented command, pass, and still fail CI on files it
  never looked at.

### Removed
- `FiscalYearGroupedSplitter._iter_test_indices` and `_iter_test_masks`. Both were
  unreachable — the class overrides `split`, so `cross_validate` never called
  either — and the comment claiming `BaseCrossValidator` requires them was false.

### Fixed
- `FiscalYearGroupedSplitter`'s module doctest asserted
  `... <= ... + 1 or True`, which passes for every possible input and so proved
  nothing about the split. It now asserts what the class actually promises —
  `fiscal_years[train_idx].max() < fiscal_years[test_idx].min()` — and a new
  `test_default_splitter_no_leakage_gap_years_zero` covers the default
  `gap_years=0` path, which had no leakage test at all. Thanks to
  [@fuleinist](https://github.com/fuleinist) (Chris Chen) for the first external
  contribution ([#30](https://github.com/PhilanthroPy-Project/PhilanthroPy/pull/30),
  closes [#26](https://github.com/PhilanthroPy-Project/PhilanthroPy/issues/26)).

### Added
- **CiviCRM contribution bridge** — `philanthropy.ingest.read_civicrm_contributions`
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
  moving "now" — the same leakage contract as the UniSchema bridge. See
  [docs/how-to/ingest_civicrm_contributions.md](docs/how-to/ingest_civicrm_contributions.md).

## [1.0.0] - TBD

The API freeze. No code changes — 1.0.0 is a promise, not a feature.

### Changed
- `Development Status :: 5 - Production/Stable`.
- **Tier 1 is now semver-protected.** A breaking change to any Tier 1 symbol
  requires a major release, preceded by one full published minor emitting
  `DeprecationWarning`. Tier 2 may still break in a minor; Tier 3 carries no
  guarantee. The tiers are listed per-symbol in
  [docs/reference/index.md](docs/reference/index.md).

### Added
- `test_stability_tier_table_covers_every_public_symbol` — the tier table is now
  machine-checked against `__all__`, so a new public symbol cannot ship without
  a stated tier. At 1.0 that table is the contract; an out-of-date one is a
  broken promise, not a docs nit.

### Notes
The five 1.0 gates all hold at this commit: 0.7.0 published; the public-API
contract test green with no exemption added since 0.7.0; no `deprecated_alias`
anywhere in `philanthropy/`; `__version__ == importlib.metadata.version(...)`
and `py.typed` in the wheel; every `__all__` symbol carries a tier and no Tier 1
entry is mid-deprecation.

## [0.7.0] - TBD

The removal release. Every shim below shipped in 0.6.0 emitting a
`DeprecationWarning` for one full published minor.

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
  keyword-only.** They do not share an argument order —
  `cost_per_dollar_raised` takes expense first, `fundraising_roi` takes raised
  first — so a positional call was silently accepted and returned a plausible
  wrong number. It is now a `TypeError`.
- **Four accidental second import paths moved behind underscores** so 1.0 does
  not freeze them: `metrics.scoring` → `metrics._scoring`,
  `preprocessing.transformers` → `preprocessing._transformers`,
  `models.propensity` → `models._propensity_baseline`, `utils.testing` →
  `utils._testing`. Every public symbol is unchanged and still exported from its
  subpackage; only a direct `from philanthropy.metrics.scoring import ...`
  breaks. Import from the subpackage instead.
- `philanthropy/utils/_deprecation.py` is gone — nothing is deprecated at 0.7.0.

### Removed
- `tests/test_deprecations.py`, which existed solely to police the shims.

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
- `philanthropy/py.typed` — the package now ships its type information.
- `joblib>=1.2` is declared; it was a direct import carried transitively.
- `tests/test_public_api_contract.py` — an executable spec for the public API:
  subpackage `__all__` completeness, reference-page coverage, the
  `predict_<thing>_(score|forecast)` naming and shape contract, and
  `get_feature_names_out` width. Two named exemptions, each with a reason.
- `tests/test_metrics_oracles.py` — closed-form oracles for the money metrics:
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
  `RFMTransformer` moved to an explicit contract class — its `_skip_test=True`
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
- Donor-base concentration metrics — `gift_concentration_gini` and
  `top_donor_share` (`philanthropy.metrics`).
- Campaign-efficiency metrics — `cost_per_dollar_raised` and `fundraising_roi`.
- `philanthropy.utils.save_model` / `load_model` — self-describing model
  bundles that warn on a scikit-learn / PhilanthroPy version mismatch at load
  time; the CLI now persists and loads through them.
- `AskAmountRecommender` — capacity regressor exposing a discrete gift-array
  ask ladder (`predict_ask_array`).
- `MatchingGiftFeaturizer` — corporate matching-gift features (`has_employer`,
  `match_ratio`, `potential_matched_amount`), leakage-safe.
- `philanthropy.experimental.UpliftTLearner` — two-model uplift (treatment-
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
- flake8 lint gate — `.flake8` (enforces pyflakes `F` + syntax `E9` defects),
  a `make lint` target folded into `make ci`, and a CI step.
- `philanthropy.inspection.donor_feature_importance` — model-agnostic permutation
  feature importance (dependency-free interpretability; works on calibrated models
  that lack `feature_importances_`).
- `philanthropy.metrics.disparate_impact_ratio` and `selection_rate_by_group` —
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
  (`reference_date_`) instead of recomputing it from the transform batch — a
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
- `philanthropy.ingest` — the UniSchema on-ramp. `constituent_events_to_features()`
  aggregates a UniSchema `ConstituentEvent` stream into a one-row-per-donor
  feature table whose columns (`total_gift_amount`, `years_active`,
  `event_attendance_count`, `last_gift_date`, ...) feed the estimators directly;
  `read_constituent_events()` loads UniSchema's JSON / NDJSON egress files.
  Leakage-safe (recency anchored to an explicit `reference_date` or the batch's
  latest event), at-least-once-safe (deduplicates by `eventId`).
- `constituent_events_to_features` and `read_constituent_events` re-exported at
  the top level (`from philanthropy import constituent_events_to_features`).
- `examples/quickstart.py` and `examples/unischema_to_scores.py` — runnable,
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
  `predict_revenue_forecast(X, horizon)`; leakage-safe — fill values and
  autoregressive coefficients frozen at `fit()`; passes sklearn
  `check_estimator`
- tests/test_forecast_model.py (fit/predict, forecast horizon, leakage,
  NaN handling, check_estimator compliance)
- PyPI packaging: complete project metadata, classifiers, keywords, and
  project URLs (docs / repo / changelog / issues); version bumped to 0.3.0
- MANIFEST.in so the sdist ships source only (no tests/dev artifacts)
- PyPI Trusted Publishing workflow (.github/workflows/publish.yml) — OIDC,
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
