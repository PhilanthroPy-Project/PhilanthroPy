# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-03-14

### Added

- DischargeToSolicitationWindowTransformer: post-discharge solicitation window featurization (in_window, window_position_score, discharge_recency_tier)
- PlannedGivingIntentScorer: bequest intent classifier with predict_intent_score() (GradientBoostingClassifier + CalibratedClassifierCV)
- LapsePredictor: production RandomForestClassifier backend with
  predict_lapse_score(), class_weight, max_depth, random_state params
- tests/test_share_of_wallet.py — 25-test suite for ShareOfWalletRegressor
- tests/test_rfm_transformer.py — 20-test suite for RFMTransformer
- tests/test_major_gift_classifier.py — 20-test suite for MajorGiftClassifier
- tests/test_visualisation.py — 12 headless tests for philanthropy.visualisation
- GitHub Actions CI workflow with Python 3.10/3.11 matrix

### Changed

- test_metrics.py expanded from 6 to 18 tests (edge cases for all three metric functions)
- test_propensity.py expanded from 3 to 20+ tests
- InvalidParameterError now accepted alongside ValueError in param-validation tests (sklearn 1.6+ compat)
- test_leakage.py updated to match current WealthScreeningImputer API

### Fixed

- Param validation moved out of __init__ to pass check_do_not_raise_errors_in_init_or_set_params
- Hypothesis tests stabilised with @settings(suppress_health_check=..., max_examples=50)
- Interval(int,...) replaced with Interval(numbers.Integral,...) for sklearn 1.6+ compatibility

## [0.1.0] - 2026-01-01

### Added

- Initial release: DonorPropensityModel, ShareOfWalletRegressor, MajorGiftClassifier
- CRMCleaner, WealthScreeningImputer, FiscalYearTransformer, EncounterTransformer, RFMTransformer
- philanthropy.metrics: donor_retention_rate, donor_acquisition_cost, donor_lifetime_value
- philanthropy.visualisation: plot_affinity_distribution
- philanthropy.utils: make_donor_dataset
- 161 tests across 7 test files
