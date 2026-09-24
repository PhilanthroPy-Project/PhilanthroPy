# PhilanthroPy: agent instructions

scikit-learn–native toolkit for nonprofit / academic-medical-center (AMC)
fundraising analytics. Every estimator is pipeline-safe, leakage-safe, and
passes `sklearn.utils.estimator_checks.check_estimator`.

## Layout
- `philanthropy/{datasets,ingest,inspection,preprocessing,models,metrics,model_selection,experimental,visualisation,utils}/`,
  plus `philanthropy/cli.py` (the `philanthropy` console script).
- Public classes live in private modules (`_wealth.py`, `_forecast.py`, …) and
  are re-exported from each subpackage's `__init__.py` (and its `__all__`).
- `tests/` holds one file per component. Flat layout (no `src/`); MkDocs in `docs/`.

## Estimator conventions (mirror existing classes, e.g. `_lapse.py`, `_wallet.py`)
- Subclass the sklearn mixin **and** `BaseEstimator`: `ClassifierMixin`,
  `RegressorMixin`, or `TransformerMixin`.
- `__init__` only stores raw params, with no validation or logic, because
  sklearn's `get_params` / `clone` read them back as passed. Include
  `random_state` wherever there is randomness.
- Validate in `fit` via `validate_data(self, X, y, ...)`; set `n_features_in_`
  plus any `trailing_underscore_` fitted attrs; `fit` returns `self`.
- Declare `__sklearn_tags__` when relevant (e.g. `tags.input_tags.allow_nan =
  True`, `tags.regressor_tags.poor_score = True`).
- Name the domain scoring/forecast method `predict_<thing>_score` /
  `predict_<thing>_forecast` / `predict_<thing>_interval` (cf.
  `predict_affinity_score`, `predict_lapse_score`, `predict_revenue_forecast`,
  `predict_gift_interval`). `tests/test_public_api_contract.py` enforces the
  three suffixes; anything else keeping the `predict_` prefix fails.
- Expose `n_iter_` after fit if the class takes a `max_iter` param
  (`check_estimator` requires it).

## Leakage-safety contract (non-negotiable)
All fitted statistics (fill values, summaries, coefficients) are computed from
TRAINING data in `fit` and FROZEN before `transform`/`predict`; `transform` is
idempotent. Reference: `WealthScreeningImputer` and `tests/test_leakage.py`.

## Missing values
`LinearRegression` / `MLPRegressor` reject NaN; impute internally with frozen
per-column medians (see `FinancialForecastModel`). `HistGradientBoosting*`
handles NaN natively (see `ShareOfWalletRegressor`, `MajorGiftClassifier`).

## Dependencies
Runtime dependencies are scikit-learn, pandas, numpy, and joblib; matplotlib and
seaborn are the optional `viz` extra (see `pyproject.toml`). Don't add others,
TensorFlow / Keras / statsmodels / torch included; approximate heavier methods
with this stack (e.g. the hybrid LSTM-ARIMA forecaster uses LinearRegression +
MLPRegressor).

## Adding a new class (order from CONTRIBUTING.md)
Tests import from the subpackage, so a test written before the export exists
fails collection and stops `make ci`. That is why the order is:
1. Implement the class. 2. Export it in the subpackage `__init__.py` and `__all__`.
3. Verify the import: `python -c "from philanthropy.<subpackage> import X"`.
4. Write the tests. 5. Run `make ci`.

## Local dev gotcha
Install editable so the working tree is what's tested:
`python -m pip install -e ".[dev]"`. A non-editable copy in site-packages will
otherwise shadow your edits under pytest and silently run stale code.

## Local gate: exact commands
```bash
python -m pip install -e ".[dev]"   # editable only; see the gotcha above
sh scripts/install_hooks.sh         # pre-push hook runs the FULL suite on every push
make ci                             # flake8 + mypy + doctests + tests + the coverage floor
make riskcov                        # the risk-tier floor CI also enforces
```
`make ci` reads its coverage floor from `pyproject.toml`; `make riskcov` is the
separate, higher floor over the risk-tier subtree. CI runs both
(`.github/workflows/ci.yml`). Do not hardcode either number anywhere else,
because copies drift from the source.

## Branching: no direct commits to main
Every change, including maintainer- and agent-authored ones, goes on a feature
branch and through a pull request. Never commit or push straight to `main`.

### Merging
`.github/CODEOWNERS` is `* @shivamlalakiya` and nobody else holds merge
rights, so a rule requiring someone else's merge would mean nothing merges.
The bar instead:

- Open a PR for every change.
- CI must be green before merge.
- If a second reviewer is available, wait for them.
- If not, the maintainer may merge their own PR once CI is green.
- Agents may merge under the same bar plus a review: all required CI checks
  green (checked directly, not from a stale or partial check list), no second
  reviewer available, and the agent has read the PR's diff and judged it
  good. Green CI alone is not enough; a PR whose content looks wrong,
  incomplete, or out of scope stays open even if every check passes.

This is a single-maintainer workaround, not the preferred rule. Once a second
person has merge rights, go back to leaving every merge to review.

## Every PR must also
- Add an entry under `## [Unreleased]` in CHANGELOG.md.
- Add the PR's human author to CONTRIBUTORS.md if they are not already listed
  and want to be (the PR template makes it optional). Agents don't list
  themselves.
- Never `git push --no-verify`: the pre-push hook is the local copy of the
  test gate.
