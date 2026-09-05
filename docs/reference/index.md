# API Reference

The API reference is generated automatically from the Python docstrings. Select a specific module below or from the sidebar navigation.

* [Preprocessing](preprocessing.md)
* [Models](models.md)
* [Metrics](metrics.md)
* [Model Selection](model_selection.md)
* [Inspection](inspection.md)
* [Datasets](datasets.md)
* [Ingest](ingest.md)
* [Visualisation](visualisation.md)
* [Utils](utils.md)
* [Experimental](experimental.md)
* [CLI](cli.md)

## Stability tiers

PhilanthroPy follows [Semantic Versioning](https://semver.org). Which promise applies to a symbol depends on its tier.

| Tier | Promise | Breaking change requires |
|---|---|---|
| **Tier 1: Stable** | The signature and the meaning of the return value are fixed. | A major release, preceded by one full published minor emitting `DeprecationWarning`. |
| **Tier 2: Beta** | Works and is tested, but the shape may still move. | A minor release, called out under **Breaking** in [CHANGELOG.md](https://github.com/PhilanthroPy-Project/PhilanthroPy/blob/main/CHANGELOG.md). |
| **Tier 3: Experimental** | No guarantees at all. May change or disappear without a deprecation cycle. | Nothing. |

Everything reachable from `philanthropy.__all__` is listed below. A symbol not listed here is not public, whatever its name looks like.

### Tier 1: Stable

| Symbol | Module |
|---|---|
| `CRMCleaner`, `FiscalYearTransformer`, `RFMTransformer` | `preprocessing` |
| `WealthScreeningImputer`, `WealthScreeningImputerKNN`, `WealthPercentileTransformer` | `preprocessing` |
| `ShareOfWalletScorer`, `PlannedGivingSignalTransformer` | `preprocessing` |
| `DonorPropensityModel`, `MajorGiftClassifier`, `LapsePredictor` | `models` |
| `PlannedGivingIntentScorer`, `ShareOfWalletRegressor`, `MovesManagementClassifier` | `models` |
| `PropensityScorer` | `models` |
| every function in `philanthropy.metrics` | `metrics` |
| `FiscalYearGroupedSplitter` | `model_selection` |
| `generate_synthetic_donor_data`, `load_ciob_fundraising` | `datasets` |
| `save_model`, `load_model`, `make_donor_dataset` | `utils` |
| `donor_feature_importance` | `inspection` |

### Tier 2: Beta

| Symbol | Module | Why not Tier 1 |
|---|---|---|
| `EncounterTransformer`, `EncounterRecencyTransformer` | `preprocessing` | Clinical schema conventions are still settling across AMCs. |
| `GratefulPatientFeaturizer` | `preprocessing` | Default service-line capacity weights are illustrative, not benchmarked. |
| `DischargeToSolicitationWindowTransformer` | `preprocessing` | Window bounds and the position-score curve may be re-tuned. |
| `SolicitationWindowTransformer` | `preprocessing` | Supported alias of the above; shares its tier. |
| `MatchingGiftFeaturizer` | `preprocessing` | The employer-normalisation rules will grow. |
| `AskAmountRecommender` | `models` | The ask-ladder multipliers are a heuristic. |
| `constituent_events_to_features`, `read_constituent_events` | `ingest` | Tracks the UniSchema `ConstituentEvent` schema, which is versioned upstream. |
| `civicrm_contributions_to_features`, `read_civicrm_contributions` | `ingest` | Tracks CiviCRM's contribution export labels and APIv4 field names, which move with the CRM. |
| `plot_affinity_distribution`, `plot_retention_waterfall` | `visualisation` | Chart composition is presentation, not contract. |
| `fetch_kdd98_donors` | `datasets` | Returns the raw upstream columns untyped; may gain as-of date parsing as the real-data leakage replication in #124 lands. |
| `make_donor_panel` | `datasets` | The returned dict may gain keys (pledges, appeals, soft credits) as more of the library needs panel-shaped fixtures; existing keys and their columns will not change silently. |
| `GiftIntervalCalibrator`, `GiftInterval` | `models` | The conformity-score menu is expected to grow (conformalised quantile regression is not shipped), which adds `score` values rather than changing existing ones. |

### Tier 3: Experimental

| Symbol | Module | Why |
|---|---|---|
| `FinancialForecastModel` | `models` | The hybrid LSTM-ARIMA surrogate is an approximation of a method the dependency policy rules out; its accuracy characteristics are not established. |
| `UpliftTLearner` | `experimental` | `fit(X, y, treatment)` breaks the sklearn signature and the estimator is not `check_estimator` compliant. |

## Score scales

Every domain method returns a number on its own scale. None of them are calibrated probabilities.

| Method | Returns | Scale |
|---|---|---|
| `DonorPropensityModel.predict_affinity_score` | `(n,)` float | 0–100, monotone in `predict_proba` |
| `MajorGiftClassifier.predict_affinity_score` | `(n,)` float | 0–100, from calibrated probabilities |
| `LapsePredictor.predict_lapse_score` | `(n,)` float | 0–100, higher = more likely to lapse |
| `PlannedGivingIntentScorer.predict_intent_score` | `(n,)` float | 0–100 |
| `UpliftTLearner.predict_uplift_score` | `(n,)` float | **−1 to 1**, negative means the appeal suppresses giving |
| `ShareOfWalletScorer.transform` | `(n, 2)` float | `capacity_utilisation_ratio` 0–1; `capacity_tier` in {0, 1, 2} |
| `ShareOfWalletRegressor.capacity_ratio` | `(n,)` float | Unbounded ratio ≥ 0 (capacity ÷ historical giving) |
| `DischargeToSolicitationWindowTransformer.transform` | `(n, 2)` float | `in_solicitation_window` in {0, 1}; `window_position_score` 0–1 |
| `GratefulPatientFeaturizer.transform` | `(n, 4)` float | Unbounded counts and weighted sums, all ≥ 0 |
| `AskAmountRecommender.ask_ladder` | `(n, 3)` float | **Dollars**, not a score: conservative / target / stretch |
| `MovesManagementClassifier.action_priority` | `dict` | Not an array: `stage`, `confidence` (0–1), `portfolio_summary` |
| `FinancialForecastModel.predict_revenue_forecast` | `(horizon,)` float | **Dollars per future period**, length is `horizon`, not `len(X)` |
| `GiftIntervalCalibrator.predict_gift_interval` | `GiftInterval` | Not a score: two `(n,)` dollar bounds, plus the `attained_level` they certify, which is `r / (n + 1)` and not the requested `1 - alpha` |

## Deprecations

!!! info "This page is built from `main`, which is ahead of the release"
    `pip install philanthropy` currently gives you **0.7.0**. The tier tables
    above describe the API as it stands on `main`; the deprecations below are
    live in the version you actually have installed.

### Removed in 0.7.0

If you are upgrading from 0.6.0, these are gone, each after one full published
minor of `DeprecationWarning`:

| Removed | Use instead |
|---|---|
| `AskAmountRecommender.predict_ask_array` | `ask_ladder` |
| `ShareOfWalletRegressor.predict_capacity_ratio` | `capacity_ratio` |
| `MovesManagementClassifier.predict_action_priority` | `action_priority` |
| `PlannedGivingIntentScorer.predict_bequest_intent_score` | `predict_intent_score` |

The `predict_` prefix is reserved for methods that take `X` alone and return one
value per row. The first three returned a `(n, 3)` dollar matrix, required a
second argument, and returned a dict respectively.

Three constructor parameters that had no effect are gone too; passing any of
them is now a `TypeError`: `LapsePredictor(lapse_window_years=...)`,
`PropensityScorer(estimator=...)`, `FiscalYearGroupedSplitter(fiscal_year_start=...)`.

`donor_acquisition_cost`, `cost_per_dollar_raised` and `fundraising_roi` are now
**keyword-only**: they do not share an argument order, so a positional call was
silently accepted and returned a plausible wrong number. Write them with
keywords and the upgrade is a no-op.

Four accidental second import paths moved behind underscores, so 1.0 does not
freeze them: `metrics.scoring` → `metrics._scoring`,
`preprocessing.transformers` → `preprocessing._transformers`,
`models.propensity` → `models._propensity_baseline`, `utils.testing` →
`utils._testing`. Every public symbol they export is unchanged; import from the
subpackage (`from philanthropy.metrics import ...`), as the documented examples
always have, and nothing breaks.

### Live in 0.7.0, removed in 0.8.0

| Deprecated | Use instead |
|---|---|
| `WealthScreeningImputerKNN(group_col_idx=...)` | nothing; see below |

`group_col_idx` has no replacement because there is nothing to replace. It was
documented for a long time as stratifying KNN imputation per group "improving
local accuracy" while being stored and never read, then implemented, and the
implementation is what retired it: measured across several synthetic two-group
pools and on five Python versions in CI, per-group and global KNN imputation
produce **bit-identical** output (`50263.48615163204` both ways). A donor's
nearest neighbours by feature distance almost always share their group already,
and `KNNImputer` weights distance by column magnitude, so a 0/1 group flag
barely registers.

If you need per-group behaviour, split the frame by group and fit one imputer per
part. That is explicit, and it costs nothing that the parameter was buying.

### Live on `main` (0.8.0), removed in 0.9.0

| Deprecated | Use instead |
|---|---|
| `ShareOfWalletScorer` output name `sow_score` | `capacity_utilisation_ratio` |

`ShareOfWalletScorer.transform` column 0 is now named
`capacity_utilisation_ratio` in `get_feature_names_out()`: the formula is
capacity ÷ clipped modelled wealth, which is capacity utilisation. It was never
share of wallet; the docstring has said so since the class shipped, and the old
name claimed a quantity the formula does not compute. The values, the column
order, and the `capacity_tier` encoding are unchanged, so code that reads the
column positionally needs nothing. Code that spells the name can call
`get_legacy_feature_names_out()` for one published minor: it returns the old
`["sow_score", "capacity_tier"]` spelling under a `DeprecationWarning` and is
removed in 0.9.0.
