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

## Getting listed

Add yourself here in the same pull request as your change: one line, your name
or handle and what you did. If you would rather not be listed, that is fine
too; say so in the PR and it stays out.

Not sure where to start? See
[CONTRIBUTING.md](CONTRIBUTING.md) and the
[good first issues](https://github.com/PhilanthroPy-Project/PhilanthroPy/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
