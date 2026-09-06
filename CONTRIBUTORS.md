# Contributors

Everyone who has landed a change in PhilanthroPy, in order of first
contribution. Code, docs, tests, and review all count.

## Maintainer

- **Shivam Lalakiya** ([@shivamlalakiya](https://github.com/shivamlalakiya)):
  author and maintainer.

## Contributors

- **Chris Chen** ([@fuleinist](https://github.com/fuleinist)): replaced a
  tautological doctest in `FiscalYearGroupedSplitter` with one that can actually
  fail, and added the missing no-leakage test for the default `gap_years=0`
  ([#30](https://github.com/PhilanthroPy-Project/PhilanthroPy/pull/30)).
- [@BortnikMaxim](https://github.com/BortnikMaxim): documented the output-column
  contracts of the preprocessing transformers' `get_feature_names_out` methods.
- **Sebastian Legarraga** ([@slegarraga](https://github.com/slegarraga)): documented
  `fit`/`transform` on the preprocessing transformers and `fit`/`predict`/`predict_proba`
  on the model estimators, wrote the `make_donor_dataset` docstring, and covered two
  transform-time input guards that had no tests
  ([#40](https://github.com/PhilanthroPy-Project/PhilanthroPy/pull/40),
  [#42](https://github.com/PhilanthroPy-Project/PhilanthroPy/pull/42),
  [#43](https://github.com/PhilanthroPy-Project/PhilanthroPy/pull/43),
  [#44](https://github.com/PhilanthroPy-Project/PhilanthroPy/pull/44)).
- [@AsavariCharati](https://github.com/AsavariCharati): added test coverage for
  two untested guards in `constituent_events_to_features`: the all-unparseable-
  timestamps empty-frame path and the `distinct_source_systems` default when
  `sourceSystem` is absent.
- [@shubhrai23](https://github.com/shubhrai23): added missing docstrings to
  `UpliftTLearner` and `cli.main`, `Raises` sections to five metrics functions,
  and fallback-path coverage for `predict_affinity_score`
  ([#64](https://github.com/PhilanthroPy-Project/PhilanthroPy/pull/64),
  [#65](https://github.com/PhilanthroPy-Project/PhilanthroPy/pull/65),
  [#66](https://github.com/PhilanthroPy-Project/PhilanthroPy/pull/66)).
- **Harikrishna KP** ([@Mr-Neutr0n](https://github.com/Mr-Neutr0n)): added a
  regression test pinning the `donor_id` error that `RFMTransformer.fit` raises
  for non-DataFrame input
  ([#139](https://github.com/PhilanthroPy-Project/PhilanthroPy/pull/139)).
- [@stoppo22](https://github.com/stoppo22): added DataFrame feature-name
  coverage for `MovesManagementClassifier.fit`
  ([#146](https://github.com/PhilanthroPy-Project/PhilanthroPy/pull/146)).
- [@Mohd-Hamza-Khan](https://github.com/Mohd-Hamza-Khan): added the missing
  single-class fallback coverage for
  `PlannedGivingIntentScorer.predict_intent_score`
  ([#149](https://github.com/PhilanthroPy-Project/PhilanthroPy/pull/149)).
- **Lars** ([@Larslllllll](https://github.com/Larslllllll)): added the missing
  unit-test coverage for `WealthPercentileTransformer`'s all-missing and
  partially-missing column branches in `WealthPercentileTransformer` (closes
  [#169](https://github.com/PhilanthroPy-Project/PhilanthroPy/issues/169)).
- [@HeaTTap](https://github.com/HeaTTap): independently identified and fixed the
  dead `hasattr(X, "columns")` branch in `WealthPercentileTransformer.fit`
  ([#175](https://github.com/PhilanthroPy-Project/PhilanthroPy/pull/175)), later
  subsumed by [#179](https://github.com/PhilanthroPy-Project/PhilanthroPy/pull/179),
  which carried the same fix (closes
  [#168](https://github.com/PhilanthroPy-Project/PhilanthroPy/issues/168)).
- [@be-student](https://github.com/be-student): made `EncounterTransformer`
  accept parsed gift dates and report invalid dates with a column-specific
  error (closes [#163](https://github.com/PhilanthroPy-Project/PhilanthroPy/issues/163)).
- [@be-student](https://github.com/be-student): made explicit wealth-column
  schema mismatches fail with actionable diagnostics and preserved automatic
  and partial-match behavior (closes
  [#156](https://github.com/PhilanthroPy-Project/PhilanthroPy/issues/156)).
- [@be-student](https://github.com/be-student): aligned the missing service-line
  fallback with its `general` category and added regression coverage for
  missing service-line and physician columns (closes
  [#151](https://github.com/PhilanthroPy-Project/PhilanthroPy/issues/151)).

## Getting listed

Add yourself here in the same pull request as your change: one line, your name
or handle and what you did. If you would rather not be listed, that is fine
too; say so in the PR and it stays out.

Not sure where to start? See
[CONTRIBUTING.md](CONTRIBUTING.md) and the
[good first issues](https://github.com/PhilanthroPy-Project/PhilanthroPy/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
- [@haas26p-ctrl](https://github.com/haas26p-ctrl): added EncounterTransformer validation-branch tests ([#152](https://github.com/PhilanthroPy-Project/PhilanthroPy/issues/152)).
