# Changelog

All notable changes to PhilanthroPy are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

## [Unreleased]
### Added
- `constituent_events_to_features` carries `first_name` / `last_name` through to
  the donor feature table when the UniSchema feed supplies them (guarded; null
  when absent).
- Community health files: `.github` issue/PR templates,
  `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), and `SECURITY.md`.
- flake8 lint gate — `.flake8` (enforces pyflakes `F` + syntax `E9` defects),
  a `make lint` target folded into `make ci`, and a CI step.

### Fixed
- Cleared 31 real-defect lint violations (unused imports/variables) across the
  package and tests, including two dead code blocks.

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
