# PhilanthroPy — AI session context

scikit-learn–native toolkit for nonprofit / academic-medical-center (AMC)
fundraising analytics. Every estimator is pipeline-safe, leakage-safe, and
passes `sklearn.utils.estimator_checks.check_estimator`.

## Layout
- `philanthropy/{datasets,preprocessing,models,metrics,model_selection,experimental,visualisation,utils}/`
- Public classes live in private modules (`_wealth.py`, `_forecast.py`, …) and
  are re-exported from each subpackage's `__init__.py` (and its `__all__`).
- `tests/` holds one file per component. Flat layout (no `src/`); MkDocs in `docs/`.

## Estimator conventions (mirror existing classes, e.g. `_lapse.py`, `_wallet.py`)
- Subclass the sklearn mixin **and** `BaseEstimator`: `ClassifierMixin`,
  `RegressorMixin`, or `TransformerMixin`.
- `__init__` stores raw params ONLY — no validation, no logic. Include
  `random_state` wherever there is randomness.
- Validate in `fit` via `validate_data(self, X, y, ...)`; set `n_features_in_`
  plus any `trailing_underscore_` fitted attrs; `fit` returns `self`.
- Declare `__sklearn_tags__` when relevant (e.g. `tags.input_tags.allow_nan =
  True`, `tags.regressor_tags.poor_score = True`).
- Name the domain scoring/forecast method `predict_<thing>_score` /
  `predict_<thing>_forecast` (cf. `predict_affinity_score`,
  `predict_lapse_score`, `predict_revenue_forecast`).
- Expose `n_iter_` after fit if the class takes a `max_iter` param
  (`check_estimator` requires it).

## Leakage-safety contract (non-negotiable)
All fitted statistics — fill values, summaries, coefficients — are computed from
TRAINING data in `fit` and FROZEN before `transform`/`predict`; `transform` is
idempotent. Reference: `WealthScreeningImputer` and `tests/test_leakage.py`.

## Missing values
`LinearRegression` / `MLPRegressor` reject NaN — impute internally with frozen
per-column medians (see `FinancialForecastModel`). `HistGradientBoosting*`
handles NaN natively (see `ShareOfWalletRegressor`, `MajorGiftClassifier`).

## Dependencies
scikit-learn, pandas, numpy, matplotlib, seaborn — **only**. Do NOT add
TensorFlow / Keras / statsmodels / torch; approximate heavier methods with the
stack above (e.g. the hybrid LSTM-ARIMA forecaster uses LinearRegression +
MLPRegressor).

## Workflow (from CONTRIBUTING.md — follow exactly)
1. Implement the class. 2. Export it in the subpackage `__init__.py`.
3. Verify the import: `python -c "from philanthropy.models import X"`.
4. Write the tests. 5. Run `make ci` (collection → full suite → coverage ≥ 85%).
Never `git push --no-verify`; the coverage gate is 85% and must stay green.

## Local dev gotcha
Install editable so the working tree is what's tested:
`python -m pip install -e ".[dev]"`. A non-editable copy in site-packages will
otherwise shadow your edits under pytest and silently run stale code.
