# PhilanthroPy: agent instructions

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
- `__init__` stores raw params ONLY: no validation, no logic. Include
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
scikit-learn, pandas, numpy, matplotlib, seaborn, and **nothing else**. Do NOT add
TensorFlow / Keras / statsmodels / torch; approximate heavier methods with the
stack above (e.g. the hybrid LSTM-ARIMA forecaster uses LinearRegression +
MLPRegressor).

## Workflow (from CONTRIBUTING.md, follow exactly)
1. Implement the class. 2. Export it in the subpackage `__init__.py`.
3. Verify the import: `python -c "from philanthropy.models import X"`.
4. Write the tests. 5. Run `make ci` (collection → full suite → coverage ≥ 92%).
Never `git push --no-verify`; the coverage gate is 92% and must stay green.

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
(`.github/workflows/ci.yml`). Do not hardcode either number anywhere else;
that drift is what issues #21 and #22 exist to fix.

## Branching: no direct commits to main
Every change, including maintainer- and agent-authored ones, goes on a feature
branch and through a pull request. Never commit or push straight to `main`.

### Merging
This section used to end "never merge your own PR; open it and leave the merge
to review." With `.github/CODEOWNERS` set to `* @shivamlalakiya` and no second
account holding merge rights, that rule cannot hold: taken literally, nothing
ever merges. It was also visibly not being followed, and a rule the repository
breaks is worth less than a narrower one it keeps.

What is actually required:

- Open a PR for every change. No exceptions.
- CI must be green before merge. Never `git push --no-verify`.
- If a second reviewer is available, wait for them.
- If not, the maintainer may merge their own PR once CI is green.
- **Agents may merge under the same bar as the maintainer above, plus a
  review:** all required CI checks green (verify yourself, not from a stale
  or partial check list), no second reviewer available, AND the agent has
  actually read the PR's diff and judged it good. Green CI alone is not
  sufficient; a PR whose content looks wrong, incomplete, or out of scope
  stays open even if every check passes.

This describes the constraint, it does not endorse it. The fix is a second
person with merge rights, tracked in issue #82; once that exists, the stricter
"leave the merge to review" rule should come back.

## Every PR must also
- Add an entry under `## [Unreleased]` in CHANGELOG.md.
- Add yourself to CONTRIBUTORS.md (same PR).
- Never `git push --no-verify`.
