# Tutorials

Tutorials teach PhilanthroPy one step at a time. Each lesson is learning-oriented and built for beginners. Follow them in order to pick up the core concepts you need to get started.

* [Building Your First Model](building_your_first_model.md)
* [Avoiding Temporal Data Leakage](avoiding_temporal_data_leakage.md)
* [Building a Grateful Patient Pipeline](building_a_grateful_patient_pipeline.md)

## Which estimator do I need?

Find the row that matches the question you were actually asked. The middle column is the shape your
data has to be in first, and it is usually the real work.

| You want to… | Your data | Start with |
|---|---|---|
| Rank prospects for a major-gift ask | one row per donor, with a label you can observe | `DonorPropensityModel`, or `MajorGiftClassifier` if you have missing values (it handles NaN natively) |
| Predict who lapses next year | a donor-year panel: one row per donor per year | `RFMTransformer` → `LapsePredictor`, split with `FiscalYearGroupedSplitter` |
| Decide how much to ask for | a fitted propensity model plus giving history | `AskAmountRecommender.ask_ladder()` (returns dollars, not a score) |
| Put an honest interval on that number | a fitted regressor plus a held-out calibration set | `GiftIntervalCalibrator.predict_gift_interval()` |
| Score grateful-patient prospects | CRM records plus an encounter table | `EncounterTransformer(as_of=…)` → `GratefulPatientFeaturizer` → `MajorGiftClassifier` |
| Time a grateful-patient solicitation | discharge dates | `DischargeToSolicitationWindowTransformer` |
| Fill gaps in a purchased wealth screen | a wealth column with missing values | `WealthScreeningImputer`, or `WealthScreeningImputerKNN` when donors cluster by segment |
| Rank on capacity rather than history | wealth estimates plus giving totals | `ShareOfWalletScorer` (tiers) or `ShareOfWalletRegressor` (ratio) |
| Find planned-giving prospects | age, tenure, and giving pattern | `PlannedGivingSignalTransformer` → `PlannedGivingIntentScorer` |
| Find donors whose employer matches gifts | an employer string column | `MatchingGiftFeaturizer` |
| Decide the next move on a portfolio | current stage plus engagement history | `MovesManagementClassifier.action_priority()` |
| Forecast next year's revenue | annual totals, one row per period | `FinancialForecastModel` (**Tier 3**: no API guarantees) |
| Clean a raw CRM export first | whatever your CRM emitted | `CRMCleaner`, then `FiscalYearTransformer` |
| Report on a campaign you already ran | gifts, costs, donor counts | `philanthropy.metrics` (retention, LTV, ROI, Gini, cost per dollar raised) |
| Check a score for group disparity | scores plus a group column | `selection_rate_by_group`, `disparate_impact_ratio` |

Two things the table cannot say for you:

**If your data has a time dimension, it belongs in the middle column.** Most rows above look like a
modelling choice and are really a data-shaping choice. Building the features before splitting is
worth more error than any estimator here is worth accuracy: measured at **+0.376 ROC-AUC** on a real
donor file, against 0.107 for choosing the wrong splitter. See
[Real-data replication](../explanation/real_data_replication.md).

**Every estimator here is Tier 1 or Tier 2 unless marked.** Tiers, and what each one promises, are in
the [API reference](../reference/index.md#stability-tiers).
