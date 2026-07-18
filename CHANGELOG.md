# Changelog

All notable changes to PhilanthroPy are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

## [Unreleased]

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
