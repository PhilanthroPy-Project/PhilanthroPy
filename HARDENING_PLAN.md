# PhilanthroPy hardening plan

## State of play

- **Coverage is 93.88% line / 0% branch, and the residual is the dangerous part.** `pytest tests/ -q --cov=philanthropy` → `1375 passed, 4 skipped in 602.44s`, `TOTAL 1928 118 94%`. Of the 118 uncovered lines, 46 are silent-wrong-answer paths: `philanthropy/preprocessing/_grateful_patient.py:307-326` returns `np.zeros((n,4))` for every donor when the merge key is missing (84%, lowest in the tree), `_share_of_wallet.py:279-292` is an unexecuted feature-name contract on money columns, and `_constituent_events.py:270-285` is the untrusted-JSON boundary. `[tool.coverage.run]` has no `branch = true`.
- **89 findings filed across five audit areas; 5 verified already-fixed; ~74 unique still-broken after dedup — 3 blocker, ~20 high.** Blockers: the README/tutorial flagship Pipeline trains `DonorPropensityModel` on an all-zero constant matrix and exits 0; `philanthropy.model_selection` / `.experimental` / `.visualisation` raise `AttributeError` after `import philanthropy` while `mkdocs.yml:96-98` renders reference pages for them; `LapsePredictor` resolves to two different classes with swapped positional args (`models/_lapse.py:41-48` vs `experimental/_lapse.py:26-32`).
- **`pandas>=1.5` at `pyproject.toml:44` is a silent wrong answer.** `format="ISO8601"` at `philanthropy/ingest/_constituent_events.py:312` is pandas 2.0+; with `errors="coerce"` a conforming 1.5.x install yields an all-NaT, zero-row feature frame (`20 failed` on `tests/test_ingest.py`, green after `pandas==2.0.0`). `[build-system] requires setuptools>=61.0` (`pyproject.toml:2`) fails metadata generation against the PEP 639 `license` fields at `:11-12` — true floor is 77.
- **5 of 7 README python fences fail or degenerate, and no test executes any of them.** `tests/test_doc_examples.py:28` is `DOC_DIRS = ("tutorials", "how-to")`; `grep -rn README tests/` → no matches. `:58` also checks `_UNRUNNABLE_MARKERS` against the *whole file*, so one `pd.read_csv` at `docs/tutorials/building_a_grateful_patient_pipeline.md:19` skips the exact regression the file's docstring (`:5`) says it exists to catch.
- **`MajorGiftClassifier` at defaults is 443.1s of a 602.4s suite (73.6%),** from `tests/test_sklearn_compliance.py:52` (bare `MajorGiftClassifier()`, `max_iter=100` inside `CalibratedClassifierCV`) plus a duplicate battery at `tests/test_estimators.py:5`. `parametrize_with_checks([MajorGiftClassifier(max_iter=10, random_state=0)])` → `54 passed in 25.42s`. CI then runs the whole suite twice per leg (`ci.yml:44` and `:46`) across 5 Pythons.
- **Three versions are live at once:** `pyproject.toml:7` = 0.5.0, `philanthropy/__init__.py:8` = 0.4.0, `philanthropy --version` = 0.4.0. Every bundle written by `utils/_persistence.py:51` is stamped 0.4.0. The only git tag is `v0.3.0`; PyPI latest is 0.3.0 and lacks `philanthropy.{ingest,cli,inspection}`, so 2 of 3 `examples/*.py` fail after a plain `pip install philanthropy`.
- **Nothing in the repo checks that the numbers are right.** `philanthropy/metrics/` reads 100% line coverage with no closed-form oracle for `gift_concentration_gini`, `donor_lifetime_value` or `disparate_impact_ratio`. `docs/explanation/benchmarks.md:34-40` publishes three-decimal AUCs from one seed. No `.zenodo.json`, no software DOI in `CITATION.cff` (only the author's IEEE DOI at `:40`), 2 of 7 `paper/paper.bib` entries lack DOIs, and `.github/workflows/draft-pdf.yml:3-7` never runs on a pull request.

## Critical path

1. **W1.1** — `pandas>=2.0`, `setuptools>=77`, declare `joblib`. Two of these are live breakages, one silent. Nothing may be released before this.
2. **W1.2** — `__version__` from `importlib.metadata`. Unblocks every release gate and unstamps 0.4.0 from saved bundles.
3. **W2.1** — `MajorGiftClassifier(max_iter=10, random_state=0)` + collapse the four duplicate batteries. One line buys back 69% of suite runtime; every later test step pays for itself against a 3-minute suite, not a 10-minute one.
4. **W1.3** — make `DischargeToSolicitationWindowTransformer.transform` raise instead of silently reading `X.iloc[:, 0]`. This is the root cause of the blocker; the docs fix alone leaves the trap open for users.
5. **W1.4** — rewrite `tests/test_doc_examples.py` (recursive glob + README, per-file allowlist, no marker heuristic). Single structural gap behind 5 of the 8 highest-severity findings. **W1.5, W4.2, W4.3 and every new how-to depend on it.**
6. **W1.11 → W1.13** — float16 strategy fix, then the `uv --resolution lowest-direct` floor job. That job is what stops steps 1 and 11 regressing.
7. **W2.16 last in W2** — branch coverage + gate, after every value test lands, or it pins a floor against a moving number.
8. **W3.16 → (one published minor) → W3.17 → W3.18** — the release ladder is the only sequence with a real-world wait in it. Everything else in all four workstreams is parallel to it.

---

## W1 — Audit fixes (docs drift, quickstart, CI, deps)

**Goal:** every dependency floor, version string, README claim and CI gate in the repo is true, and the code samples that state them are executed.

**Exit criteria:**
- `python -c "import philanthropy, importlib.metadata as m; assert philanthropy.__version__ == m.version('philanthropy') == '0.5.0'"`
- `! grep -rn '0\.4\.0' README.md docs philanthropy`
- `python -m pytest tests/test_doc_examples.py --collect-only -q --no-cov | grep -q README.md`
- `python -m pytest tests/test_doc_examples.py -q --no-cov -rs 2>&1 | grep SKIPPED | grep -qv 'docs-notest'; test $? -eq 1`
- `python -m pytest philanthropy --doctest-modules -q --no-cov`
- `mkdocs build --strict`
- `python -m build -q && python -m twine check dist/*`
- `! grep -q 'pytest tests/ -v --tb=short' .github/workflows/ci.yml`
- `test "$(wc -l < README.md)" -lt 200 && ! grep -qE 'discharge_recency_tier\|min_samples_leaf\|1189 tests\|Python 3.10 \+ 3.11\|drop PII columns' README.md`

| # | Step | Files | Verify | Effort | Blocked by |
|---|---|---|---|---|---|
| 1 | `pandas>=1.5`→`>=2.0`, `setuptools>=61.0`→`>=77`, add `joblib>=1.2`, `mkdocs>=1.4.0,<2` | `pyproject.toml` | `python -m build -q && python -m twine check dist/*` | xs | — |
| 2 | `__version__` from `importlib.metadata`; drop the 3 prose `0.4.0` copies (`README.md:38,146`) | `philanthropy/__init__.py`, `README.md` | `python -c "import philanthropy,importlib.metadata as m;assert philanthropy.__version__==m.version('philanthropy')"` | xs | — |
| 3 | `DischargeToSolicitationWindowTransformer.transform`: raise `ValueError` instead of `X.iloc[:,0]` fallback | `philanthropy/preprocessing/_discharge_window.py`, `tests/test_audit_regressions.py` | `pytest tests/test_audit_regressions.py tests/test_sklearn_compliance.py -q -k 'discharge or window'` | s | — |
| 4 | Doc executor: recursive `docs/**` + `README.md`, keep only files with a python fence, drop `_UNRUNNABLE_MARKERS` | `tests/test_doc_examples.py` | `pytest tests/test_doc_examples.py -q --no-cov -rs` | s | — |
| 5 | README 563→<200 lines: cut deep-dive tables, tree, Testing, ✅Completed; fix all 6 false claims | `README.md`, `CONTRIBUTING.md` | `test "$(wc -l < README.md)" -lt 200 && pytest tests/test_doc_examples.py -q -k README` | m | 3, 4 |
| 6 | 4 docs prose corrections: capacity tiers, PHI attribution, GPF outputs, `CRMCleaner.fiscal_year_start` | `docs/explanation/{capacity_and_loyalty,design_principles}.md`, `docs/how-to/build_grateful_patient_features.md`, `philanthropy/preprocessing/transformers.py` | `mkdocs build --strict` | xs | — |
| 7 | Fix doctest at `_constituent_events.py:116` (`float(...)`); add `--doctest-modules` to Makefile `ci` | `philanthropy/ingest/_constituent_events.py`, `Makefile` | `pytest philanthropy --doctest-modules -q --no-cov` | xs | — |
| 8 | `_compute_fill`: return 0.0 early when the column is all-NaN — kills the `Mean of empty slice` RuntimeWarning | `philanthropy/preprocessing/_wealth.py` | `pytest tests/test_leakage.py -q -W error::RuntimeWarning` | xs | — |
| 9 | Rewrite persistence how-to on `save_model`/`load_model`; delete the `philanthropy==0.4.0` pin block | `docs/how-to/save_and_load_models.md` | `pytest tests/test_doc_examples.py -q -k save_and_load` | s | 4 |
| 10 | `docs.yml:37` → `mkdocs build --strict`; drop `fetch-depth: 0` (no revision-date plugin) | `.github/workflows/docs.yml` | `mkdocs build --strict` | xs | W4.1 |
| 11 | `floating_dtypes()` → `floating_dtypes(sizes=(32,64))` — float16 breaks on every numpy 1.x | `tests/test_transformers_property.py` | `pytest tests/test_transformers_property.py -q --hypothesis-seed=7` | xs | — |
| 12 | `ci.yml`: delete duplicate suite run, lint job out of matrix, `package` job, macOS on 3.12, minimal-install job, PR-only concurrency | `.github/workflows/ci.yml` | `! grep -q 'pytest tests/ -v --tb=short' .github/workflows/ci.yml && python -c "import yaml;yaml.safe_load(open('.github/workflows/ci.yml'))"` | s | 1 |
| 13 | `floors` job: `uv venv --python 3.9`, `uv pip install --resolution lowest-direct -e ".[dev]"`, `uv run pytest tests/ -q` | `.github/workflows/ci.yml` | see Detail | s | 1, 11 |
| 14 | `publish.yml`: tag↔pyproject↔CHANGELOG gate; SHA-pin `gh-action-pypi-publish` + `openjournals-draft-action` | `.github/workflows/publish.yml`, `.github/workflows/draft-pdf.yml` | `grep -q tag_name .github/workflows/publish.yml && grep -qE 'pypi-publish@[0-9a-f]{40}' .github/workflows/publish.yml` | s | — |
| 15 | Delete dead config: `[tool.hypothesis]`, `MANIFEST.in:11`, `environment.yml` (+`README.md:69`), dependabot `pip` block | `pyproject.toml`, `MANIFEST.in`, `environment.yml`, `README.md`, `.github/dependabot.yml` | `! grep -q 'tool.hypothesis' pyproject.toml && ! grep -rn environment.yml README.md docs .github` | xs | 5 |
| 16 | `read_constituent_events`: `if not p.exists(): raise FileNotFoundError(...)` before the dir/file dispatch | `philanthropy/ingest/_constituent_events.py`, `tests/test_ingest.py` | `pytest tests/test_ingest.py -q -k 'missing or not_found'` | xs | — |

### Detail

**W1.1 — the two floors that are lies.**

```toml
[build-system]
requires = ["setuptools>=77"]          # was >=61.0; PEP 639 license fields at :11-12 need 77

dependencies = [
    "scikit-learn>=1.6",               # unchanged: validate_data + __sklearn_tags__ landed in 1.6
    "pandas>=2.0",                     # was >=1.5; format="ISO8601" is 2.0+ and coerces to all-NaT
    "numpy>=1.23",                     # unchanged: honest for library code
    "joblib>=1.2",                     # direct import at utils/_persistence.py:21, was transitive only
]
```
`requires-python = ">=3.9"` stays — verified honest on a real 3.9.6 venv, and 3.9 is the only leg that pins sklearn 1.6.x coverage. `mkdocs>=1.4.0,<2` goes in the `docs` extra only (upstream already warns MkDocs 2.0 is incompatible with mkdocs-material).

**W1.3 — the root cause, not the sample.** `philanthropy/preprocessing/_discharge_window.py:85-86` currently does `elif isinstance(X, pd.DataFrame): days_raw = X.iloc[:, 0]`, so a `FiscalYearTransformer` upstream feeds `fiscal_year` (2020-2025) in as "days since discharge", every value falls outside `[90, 365]`, and `np.zeros` at `:102-103` is never overwritten. Replace with `raise ValueError(f"{self.days_since_discharge_col!r} not found in X; columns are {list(X.columns)}")`. Keep the bare-ndarray branch at `:88-92` — no names available and `check_estimator` needs it. Verified: `check_estimator(DischargeToSolicitationWindowTransformer())` passes with the guard in place, and line 86 is currently uncovered so no test depends on it.

**W1.4 — one design, not two.** Per-block marker filtering does not work: `:55` concatenates every fence and `:67` `exec`s the result once in one namespace (the docstring at `:7-8` says later blocks depend on earlier ones), so dropping a middle block leaves later blocks referencing undefined names. The correct change is file-level with an explicit allowlist:

```python
DOC_FILES = sorted(DOCS_ROOT.rglob("*.md")) + [Path(__file__).resolve().parent.parent / "README.md"]
_NOTEST = {"README.md"}          # the illustrative UniSchema fence; moves to W4.4, then this empties

def _doc_files():
    return [p for p in DOC_FILES if "```python" in p.read_text()]   # no-fence files aren't collected

def test_notest_allowlist_is_exact():                 # muting a page becomes a visible diff
    marked = {p.name for p in DOC_FILES if "<!-- docs-notest -->" in p.read_text()}
    assert marked == _NOTEST
```
`_UNRUNNABLE_MARKERS` is deleted outright — 20 of 26 `docs/**/*.md` have no python fence at all, so filtering at collection removes the 20 noise skips, and the `read_csv`/`open(` heuristic is what silenced the grateful-patient tutorial.

**W1.5 — README to a landing page, in one step, so nothing re-touches it.** Delete `README.md:217-373` (per-component tables — mkdocstrings already renders every parameter, and this one span carries the phantom `min_samples_leaf`/`max_depth` rows, the nonexistent `discharge_recency_tier` column at `:227`, and the inverted integer-affinity claim at `:303,305`); `:398-436` (package tree); `:375-392` (degenerate Pipeline — corrected version lands in `docs/tutorials/building_your_first_model.md`, W4.3); `:470-489` (Testing — kills "1189 tests across 30 files"; actual is 1379/41); `:494-517` (✅ Completed — kills the false "Python 3.10 + 3.11 matrix" at `:509` and "Branch protection" at `:512`; `ci.yml:21` is 3.9–3.13 and `gh api .../branches/main/protection` → 404). Also: cut "strip whitespace, drop PII columns" from `:152` (`CRMCleaner.transform` at `transformers.py:104-127` only coerces two columns; only `EncounterTransformer` has `PII_PATTERNS`), drop the `⭐ NEW` markers at `:158,159,160,176,177,183,202`, and delete `CONTRIBUTING.md:67` which permits `git push --no-verify` two sections after `:35` forbids it. Keep `### 🔜 Next`. Target shape: badges, what/who, install, one Quick Start, condensed feature tables, docs links, citation, contributing, license.

**W1.12 — CI restructure.**

```yaml
concurrency:                                  # was group: ci-${{ github.ref }} — cancelled main runs mid-suite
  group: ci-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}
```
Delete the `Run test suite` step (`:43-44`); the `Coverage gate` step at `:46` already runs every test and pytest-cov honours `fail_under` from `pyproject.toml`. Move `flake8` (`:32`) and `mypy` (`:34`) into a single-Python `lint` job — identical source, was running 5×. Add `os: [ubuntu-latest, macos-latest]` on the 3.12 leg only: this package reads `pd.Timestamp.today().normalize()` (`_encounter_recency.py:296,305`) and localizes to `America/New_York` (`tests/test_coverage_boost.py:31`), which is exactly the code that differs off Linux, and JOSS reviewers install on macOS. Add two small jobs: `package` (`python -m build && python -m twine check dist/*` — plain, matching `publish.yml`, so the `setuptools>=61` class of bug surfaces on the PR not the release) and `minimal` (`pip install .` then `python -c "import philanthropy, philanthropy.visualisation"` — the only thing that actually tests the lazy-matplotlib guarantee `docs/explanation/comparison_to_vendors.md:17` claims, since all 5 legs install `[dev]` which includes matplotlib).

**W1.13 — the floor job.** `uv` is a CI-only action, never a package dependency. Plain `pip` resolves numpy 2.0.2 / pandas 2.3.3 even on the 3.9 leg, which is exactly why the pandas 1.5 breakage survived; `--resolution lowest-direct` is the only cheap way to install what the metadata declares. Verify locally with the scratchpad, not `/tmp`: `uv venv --python 3.9 "$SCRATCH/floorenv" && VIRTUAL_ENV="$SCRATCH/floorenv" uv pip install --resolution lowest-direct -e ".[dev]" && "$SCRATCH/floorenv/bin/python" -m pytest tests/ -q` — the same `tests/` selection the job runs, so local green means CI green.

**W1.14 — release gate.** One step in the `build` job before `python -m build` (`publish.yml:33`), stdlib `tomllib` on the 3.11 runner:

```yaml
- name: Tag, version and changelog must agree
  run: |
    V=$(python -c "import tomllib;print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")
    [ "v$V" = "${{ github.event.release.tag_name }}" ] || { echo "tag != pyproject $V"; exit 1; }
    grep -q "^## \[$V\]" CHANGELOG.md || { echo "CHANGELOG.md has no ## [$V]"; exit 1; }
```
SHA-pin `pypa/gh-action-pypi-publish` (`publish.yml:60`, runs with `id-token: write` against the `pypi` environment) and `openjournals/openjournals-draft-action@master` (`draft-pdf.yml:21`, a mutable third-party *branch*). `.github/dependabot.yml` already tracks `github-actions` weekly, so the pins stay current — which is also why W1.15 deletes only the `pip` half of that file: no lockfile exists and every constraint is an open `>=`, so it can never open a PR.

**Out of scope**
- Cutting the 0.5.0 PyPI release — needs a maintainer, a tag and the OIDC environment; W3.16 owns it. W1.5 fixes the misleading `pip install philanthropy` instruction meanwhile.
- Enabling branch protection on `main` — W1.12 renames and adds jobs, so required-check names pinned now go stale immediately. Do it after W1.12 merges; W1.5 already deletes the false claim.
- `filterwarnings = ["error"]` / `--strict-markers` in pytest config — W2 owns warning policy; W1.8 fixes the one real RuntimeWarning behind it.
- `philanthropy/__main__.py` for `python -m philanthropy` — nothing documents that invocation; the console script and `python -m philanthropy.cli` both work.
- Guarding `from sklearn.utils._set_output import _get_output_config` (`transformers.py:32`) — real fragility, no observed break on sklearn 1.8, never filed as a finding.
- Adding assertions to `tests/test_examples.py` — the degenerate-output defect lived in the README pipeline, which W1.4+W1.5 put under execution.
- Test-order independence / `pytest-randomly` — speculative; W2.11's RNG seeding and `TZ=UTC` pin address the observed state leakage.

---

## W2 — Test suite to defensible coverage

**Goal:** every public estimator runs one real battery or a machine-enforced exclusion, every stateful transformer has a batch-independence test, every money metric has a closed-form oracle, and branch coverage gates a 92% global floor plus a 90% risk-tier floor.

**Exit criteria:**
- `[ ! -e tests/test_sklearn_compat.py ] && [ ! -e tests/test_coverage_boost.py ]`
- `python -m pytest tests/test_sklearn_compliance.py -q -k 'PropensityScorer or WealthPercentile or WealthScreeningImputerKNN'`
- `python -m pytest tests/test_leakage.py::test_every_stateful_transformer_has_a_leakage_test -q`
- `python -m pytest tests/test_metrics_oracles.py -q`
- `! grep -rn 'np\.random\.' tests/ | grep -v default_rng`
- `python -c "import tomllib;c=tomllib.load(open('pyproject.toml','rb'))['tool']['coverage'];assert c['run']['branch'] and c['report']['fail_under']>=92"`
- `python -m pytest tests/ -q --cov=philanthropy && coverage report --include='philanthropy/preprocessing/*,philanthropy/models/*,philanthropy/ingest/*,philanthropy/cli.py,philanthropy/utils/_persistence.py' --fail-under=90`
- `python -m pytest tests/ -q --durations=5` → total under 200s

| # | Step | Files | Verify | Effort | Blocked by |
|---|---|---|---|---|---|
| 1 | `MajorGiftClassifier(max_iter=10, random_state=0)` at `:52`; fold `test_sklearn_compat.py` in (keep default-param entries); delete 3 dup batteries | `tests/test_sklearn_compliance.py`, `tests/test_sklearn_compat.py`, `tests/test_estimators.py`, `tests/test_propensity.py`, `tests/test_donor_propensity_model.py` | `pytest tests/test_sklearn_compliance.py -q --durations=5` | s | — |
| 2 | Add `WealthPercentileTransformer()`, `WealthScreeningImputerKNN()`, `experimental.LapsePredictor(n_estimators=10,…)` | `tests/test_sklearn_compliance.py` | `pytest tests/test_sklearn_compliance.py -q -k 'WealthPercentile or KNN'` | xs | 1 |
| 3 | `PropensityScorer`: single-class `predict`/`predict_proba`, binary-only check, `>` tie-break, `__sklearn_tags__`; then battery | `philanthropy/models/propensity.py`, `tests/test_sklearn_compliance.py`, `tests/test_audit_regressions.py` | `pytest tests/test_sklearn_compliance.py -q -k PropensityScorer` | s | 1 |
| 4 | `RFMTransformer` out of `_STANDARD_ESTIMATORS` → `TestRFMTransformerCompliance` (its `_skip_test=True` ran 1 check, not 46) | `tests/test_sklearn_compliance.py` | `pytest tests/test_sklearn_compliance.py -q -k RFM` | s | 2, 3 |
| 5 | `GratefulPatientFeaturizer`: `UserWarning` before each `np.zeros((n,4))`; 2 tests on both fallbacks | `philanthropy/preprocessing/_grateful_patient.py`, `tests/test_grateful_patient_featurizer.py` | `pytest tests/test_grateful_patient_featurizer.py -q` | s | — |
| 6 | `WealthScreeningImputerKNN.get_feature_names_out` width/name test + absent-wealth-col `pytest.warns` | `tests/test_share_of_wallet.py` | `pytest tests/test_share_of_wallet.py -q --cov=philanthropy.preprocessing._share_of_wallet` | s | — |
| 7 | CLI: stdout default, all 4 `--model` choices, 3 `SystemExit` paths; delete unreachable `cli.py:37-42` | `tests/test_cli.py`, `philanthropy/cli.py` | `pytest tests/test_cli.py -q --cov=philanthropy.cli` | s | — |
| 8 | Ingest: blank JSONL line, truncated record, empty file, `{` — 2 with `match="Malformed JSON"` | `tests/test_ingest.py` | `pytest tests/test_ingest.py -q --cov=philanthropy.ingest` | s | — |
| 9 | `DonorPropensityModel` single-class (0.0/100.0, ∓0.5) + multiclass affinity forks | `tests/test_audit_regressions.py` | `pytest tests/test_audit_regressions.py -q --cov=philanthropy.models._propensity` | xs | — |
| 10 | `EncounterRecencyTransformer` freeze test + source-scan registry meta-test over `preprocessing.__all__` | `tests/test_transformer_leakage_guards.py`, `tests/test_leakage.py` | `pytest tests/test_leakage.py tests/test_transformer_leakage_guards.py -q` | s | — |
| 11 | Determinism: `default_rng(0)` ×14 sites, explicit `reference_date`, `TZ=UTC` in pytest env + one non-UTC test | 5 test files, `pyproject.toml` | `! grep -rn 'np\.random\.' tests/ \| grep -v default_rng` | s | 1 |
| 12 | Redistribute the 6 real `pytest.raises` blocks out of `test_coverage_boost.py`, then delete the file | `tests/test_coverage_boost.py`, `tests/test_{preprocessing,share_of_wallet,model_selection,moves}.py` | see Detail | s | 5, 11 |
| 13 | Real artist assertions in `test_visualisation.py`; param-acceptance tests assert the inner estimator | `tests/test_visualisation.py`, `tests/test_share_of_wallet.py`, `tests/test_propensity.py` | see Detail | s | 11 |
| 14 | Collapse triplicated property tests to `test_properties.py`; move the overflow regression before deleting its file | `tests/test_{properties,preprocessing_properties,preprocessing,transformers_property,audit_regressions}.py` | `pytest tests/test_properties.py tests/test_audit_regressions.py -q --durations=5` | m | 11, W1.13 |
| 15 | `tests/test_metrics_oracles.py`: closed-form gini, LTV annuity, EEOC four-fifths | `tests/test_metrics_oracles.py` | `pytest tests/test_metrics_oracles.py -q` | s | — |
| 16 | `branch = true`, `fail_under = 92`, `exclude_lines`; risk-tier floor via `coverage report --include=` in `ci.yml` | `pyproject.toml`, `.github/workflows/ci.yml` | see Detail | m | 4–15 |

### Detail

**W2.1 — collapse without losing coverage.** `tests/test_sklearn_compat.py:11-15` runs `WealthScreeningImputer()` and `FiscalYearTransformer()` at **defaults**; `tests/test_sklearn_compliance.py:54-59` runs them **configured** (`date_col="gift_date", fiscal_year_start=7` / `wealth_cols=["x0"], strategy="median"`). Deleting the compat file as pure duplication would drop default-parameter battery coverage for both — so add the two bare-default instances to `_STANDARD_ESTIMATORS` alongside the configured ones, plus `EncounterRecencyTransformer(reference_date="2024-01-01")`, `ShareOfWalletScorer()`, `CRMCleaner()`, `LapsePredictor(n_estimators=10, random_state=0)`, then delete `tests/test_sklearn_compat.py`. Delete `tests/test_estimators.py:5-13`, `tests/test_propensity.py:218`, `tests/test_donor_propensity_model.py:319-337`. One list, one place to read the answer to "is X compliant?".

**W2.3 — verified as a set.** I built these five edits in the scratchpad and ran `parametrize_with_checks([PropensityScorer()])` → **56 passed**. (a) `check_classification_targets(y)` + `ValueError` when `len(self.classes_) > 2`; (b) `predict` returns `np.full(n, self.classes_[0])` when `len(self.classes_) == 1` — the live bug, `self.classes_[idx]` at `propensity.py:36` raises `IndexError` today; (c) `predict_proba` returns `np.ones((n, 1))` in that case; (d) `idx = (proba > self.threshold)` — sklearn asserts `argmax(predict_proba) == predict` at `estimator_checks.py:2833` and the constant 0.5 scorer ties. **(d) flips the default-threshold prediction of the baseline from class 1 to class 0** — arbitrary either way for a constant scorer, but update the `threshold` docstring at `propensity.py:12,22` and note it in CHANGELOG. (e) `__sklearn_tags__` with `poor_score = True`, `multi_class = False`.

**W2.12 — verdict on `test_coverage_boost.py`: delete the file, keep the tests.** ~70% of it is genuinely valuable (six `pytest.raises` blocks whose error paths nothing else covers: `fiscal_year_start=13` `:16`, `strategy="invalid"` `:50`, `n_neighbors=0` `:54`, `epsilon=-1` `:81`, `capacity_col_idx=-1` `:83`, `"Not enough fiscal years"` `:104`), but the filename names the metric as the goal. Move each block to the test file for its class, tightening every `match=` to the full message. Drop `test_grateful_patient_featurizer_extra` (`:126`, asserts only `out.shape[0] == 1` under a comment that misdescribes its own fixture — superseded by W2.5) and replace `:93`'s tier assertion (an `or` over all three labels, so it cannot fail) with an exact `assert_array_equal`. Verify by grepping the moved messages, not just the deletion: `grep -q 'Not enough fiscal years' tests/test_model_selection.py && grep -q 'n_neighbors' tests/test_share_of_wallet.py && [ ! -e tests/test_coverage_boost.py ]`.

**W2.13 — verify the assertion, not the pass.** `pytest <3 files> -q` passes identically whether or not you strengthened anything, so gate on the shape of the file: `! grep -q 'assert True' tests/test_visualisation.py && [ "$(grep -c 'isinstance(ax, plt.Axes)' tests/test_visualisation.py)" -le 2 ] && grep -q 'get_height()' tests/test_visualisation.py`. `_plots.py` reads 100% on 32 statements bought with nine `isinstance(ax, plt.Axes)` assertions and one literal `assert True` (`:105`). Delete `test_plots_close_cleanly`; assert bar heights for `plot_retention_waterfall`, `len(ax.patches)` + `get_xlim()` for the histogram, legend texts when `labels` is passed. Convert `test_l2_regularization_accepted` (`test_share_of_wallet.py:159`) and `test_max_iter_accepted` (`:167`) from `assert preds.shape` to `model.estimator_.l2_regularization == 0.5` / `model.n_iter_ <= max_iter`. Leave the rest — the battery already enforces those contracts.

**W2.15 — the gap four auditors missed: is the arithmetic right?** `philanthropy/metrics/` is 100% line-covered and has no oracle. Fifteen lines:

```python
def test_gini_closed_form():
    assert gift_concentration_gini([1, 1, 1, 1]) == pytest.approx(0.0)      # perfect equality
    assert gift_concentration_gini([0, 0, 0, 4]) == pytest.approx(0.75)     # one donor: (n-1)/n
def test_ltv_matches_discounted_annuity():
    assert donor_lifetime_value(...) == pytest.approx(sum(a / (1 + r) ** t for t in range(n)))
def test_disparate_impact_matches_eeoc_four_fifths():
    assert disparate_impact_ratio(...) == pytest.approx(0.8)                # EEOC worked example
```
A closed-form assertion against an independent source is strictly stronger than the range/monotonicity property tests W2.14 frees up budget for. This is the JOSS reviewer's "have the functional claims been confirmed?" item, and nothing in the repo currently addresses it.

**W2.16 — gate config, no new script.**

```toml
[tool.coverage.run]
source = ["philanthropy"]
branch = true

[tool.coverage.report]
fail_under = 92
show_missing = true
exclude_lines = ["pragma: no cover", "if __name__ == .__main__.:", "if TYPE_CHECKING:",
                 "raise NotImplementedError", "@overload"]
```
Procedure, in this order, or you ship a red `main` and a `# pragma: no cover` spree: (1) land steps 4–15; (2) run once with `branch = true, fail_under = 0` and read the branch TOTAL; (3) pin `fail_under = min(92, floor(measured) - 1)`. 85 today has nine points of slack — it would not notice ~170 statements of tested code being deleted; 95+ forces assert-no-exception tests, which is how the current 84 weak assertions got written. The per-file floor needs **no script and no `coverage.json`** — `coverage` already does it:

```yaml
- run: coverage report --include='philanthropy/preprocessing/*,philanthropy/models/*,philanthropy/ingest/*,philanthropy/cli.py,philanthropy/utils/_persistence.py' --fail-under=90
```
`philanthropy/visualisation/*` is exempt by omission — a chart's correctness is not a line-coverage question. **Measure the tier floor before pinning 90:** even on today's *line* numbers `_encounter_recency.py` is 90%, `transformers.py` 91%, `_rfm.py` 92%, `_encounters.py` 93%, and branch mode drops each 3–5 points. No step here adds a value test to those four, so pin the tier floor at `measured − 1` on the first green run and ratchet to 90 when someone next touches them.

**Out of scope**
- `ci.yml` restructure and the duplicate suite run — W1.12. W2.1 halves per-run cost regardless.
- Dependency floors and the `uv` floor job — W1.1/W1.13. W2.14 must not delete `tests/test_transformers_property.py` until W1.13's job runs `tests/` rather than a file list; sequence W1.11 → W1.13 → W2.14.
- README/doc executor — W1.4/W1.5.
- `predict_*` renames, the deprecation shim, the `predict_*` contract test — W3.5/W3.6/W3.12. Writing the naming test before W3 fixes the names guarantees rewriting it.
- Removing `PropensityScorer.estimator` from `__init__` — W3.7/W3.17 (semver-breaking). Land W2.3 first; it is a correctness fix and does not change the signature.
- Mutation testing, a coverage service, `coverage.xml` artifacts — new surface for a signal the per-file floor plus the two meta-tests already approximate.
- Full compliance for `MatchingGiftFeaturizer`, `EncounterTransformer`, `GratefulPatientFeaturizer`, `UpliftTLearner` — genuinely incompatible signatures; W3.12's contract test records the reason so the exclusion is readable rather than a silent `_skip_test`.
- A Hypothesis profile in `conftest.py` — W1.15 deletes the inert `[tool.hypothesis]` block and W2.14 removes the 500/1000-example duplicates; a profile is machinery for a knob nobody turns.

---

## W3 — API stabilization + versioned releases

**Goal:** freeze a coherent self-enforcing public API on an explicit 0.6.0 → 0.7.0 → 1.0.0 ladder where every removal got one full published minor of `DeprecationWarning` overlap.

**Exit criteria:**
- `python -c "import philanthropy as p;[__import__('philanthropy.'+s) for s in ('model_selection','experimental','visualisation')];assert {'model_selection','experimental','visualisation'}<=set(p.__all__)"`
- `python -m build --wheel -o /tmp/w . >/dev/null && unzip -l /tmp/w/*.whl | grep -q philanthropy/py.typed`
- `python -m pytest tests/test_public_api_contract.py tests/test_deprecations.py -q`
- `python -c "import philanthropy.experimental as e;assert 'LapsePredictor' not in e.__all__"`
- `grep -q tag_name .github/workflows/publish.yml && grep -q CHANGELOG.md .github/workflows/publish.yml`
- At 1.0: `! grep -rn deprecated_alias philanthropy` and no dead param in `get_params()`

| # | Step | Files | Verify | Effort | Blocked by |
|---|---|---|---|---|---|
| 1 | `philanthropy/py.typed` + `"philanthropy" = ["py.typed"]` in `[tool.setuptools.package-data]` | `philanthropy/py.typed`, `pyproject.toml` | `python -m build --wheel -o /tmp/w . && unzip -l /tmp/w/*.whl \| grep -q py.typed` | xs | — |
| 2 | Add `model_selection, experimental, visualisation` to the `from . import` line and `__all__` | `philanthropy/__init__.py` | `python -c "import philanthropy as p;p.model_selection;p.experimental;p.visualisation"` | xs | — |
| 3 | Move `FiscalYearGroupedSplitter` + `_n_samples` to `_temporal_donor_splitter.py`; add `__all__` | `philanthropy/model_selection/{__init__,_temporal_donor_splitter}.py` | `python -c "import philanthropy.model_selection as m;assert m.__all__==['FiscalYearGroupedSplitter']"` | s | — |
| 4 | `deprecated_alias(new_name, removed_in)` in `utils/_deprecation.py` (unexported) + one test | `philanthropy/utils/_deprecation.py`, `tests/test_deprecations.py` | `pytest tests/test_deprecations.py -q` | s | — |
| 5 | Rename `predict_ask_array`→`ask_ladder`, `predict_capacity_ratio`→`capacity_ratio`, `predict_action_priority`→`action_priority`; old names alias | `philanthropy/models/{_ask,_wallet,_moves}.py`, `tests/test_deprecations.py` | `pytest tests/test_deprecations.py -q && pytest --doctest-modules philanthropy/models/_ask.py philanthropy/models/_wallet.py -q` | s | 4 |
| 6 | `predict_bequest_intent_score` → alias of `predict_intent_score` (its body already is one) | `philanthropy/models/_planned_giving.py`, `tests/test_deprecations.py` | `pytest tests/test_deprecations.py -q -k bequest` | xs | 4 |
| 7 | 3 dead params: inline `warnings.warn(DeprecationWarning)` in `fit`/`split` when non-default | `philanthropy/models/{propensity,_lapse}.py`, `philanthropy/model_selection/_temporal_donor_splitter.py`, `tests/test_deprecations.py` | `pytest tests/test_deprecations.py tests/test_propensity.py tests/test_model_selection.py -q` | s | 3, 4 |
| 8 | Delete `experimental/_lapse.py` + its export + `tests/test_experimental_lapse.py` (name collision) | `philanthropy/experimental/{_lapse.py,__init__.py}`, `tests/test_experimental_lapse.py` | `python -c "import philanthropy.experimental as e;assert e.__all__==['UpliftTLearner']"` | xs | W2.2 |
| 9 | `UpliftTLearner(ClassifierMixin, BaseEstimator)` — supplies `score()` / `_estimator_type` | `philanthropy/experimental/_uplift.py` | `python -c "from sklearn.base import is_classifier;from philanthropy.experimental import UpliftTLearner;assert is_classifier(UpliftTLearner())"` | xs | — |
| 10 | Real `n_iter_` from `calibrated_classifiers_` instead of hardcoded `1` (`_propensity.py:385`) | `philanthropy/models/_propensity.py` | see Detail | xs | W2.1 |
| 11 | Delete 2 dead `_more_tags`; `_encounter_recency._validate_fiscal_year_start` delegates its range check | `philanthropy/preprocessing/{_rfm,_encounters,_encounter_recency}.py` | `! grep -rn _more_tags philanthropy && pytest tests/test_encounter_recency.py tests/test_rfm_transformer.py -q` | xs | — |
| 12 | `tests/test_public_api_contract.py` — the executable spec (see Detail) | `tests/test_public_api_contract.py` | `pytest tests/test_public_api_contract.py -q` | m | 2, 3, 5, 8 |
| 13 | Stability tiers + score-scale table in `docs/reference/index.md` | `docs/reference/index.md` | `mkdocs build --strict && grep -q 'Tier 3' docs/reference/index.md` | s | 5, 8, W4.1 |
| 14 | `CONTRIBUTING.md` release checklist; `.zenodo.json`; `CITATION.cff` software-DOI placeholder | `CONTRIBUTING.md`, `.zenodo.json`, `CITATION.cff`, `CHANGELOG.md` | `grep -q 'RELEASE CHECKLIST' CONTRIBUTING.md && python -c "import json;json.load(open('.zenodo.json'))"` | s | 4, 13 |
| 15 | DOIs for `scikit-learn` + `sklearn_api` in `paper.bib`; add `pull_request` trigger to `draft-pdf.yml` | `paper/paper.bib`, `.github/workflows/draft-pdf.yml` | `[ "$(grep -c doi paper/paper.bib)" -eq 7 ] && grep -q pull_request .github/workflows/draft-pdf.yml` | xs | — |
| 16 | **Ship 0.6.0** — all shims live, `experimental.LapsePredictor` gone | `pyproject.toml`, `CHANGELOG.md` | `python -m build && python -m twine check --strict dist/*` | s | 1–15, W1.14 |
| 17 | **0.7.0** — remove shims, dead params, keyword-only metrics, rename 4 public modules | see Detail | `pytest tests/ -q -x && ! grep -rn deprecated_alias philanthropy` | l | 16 published |
| 18 | **1.0.0** — freeze; `Development Status :: 5 - Production/Stable` | `pyproject.toml`, `CHANGELOG.md`, `README.md` | `pytest tests/test_public_api_contract.py -q && ! grep -rn deprecated_alias philanthropy` | s | 17 |

### Detail

**W3.4 — the whole mechanism, one helper.** `CONTRIBUTING.md:71-76` promises a `DeprecationWarning` policy and `grep -rn DeprecationWarning philanthropy` returns nothing, so there is no precedent to copy:

```python
def deprecated_alias(new_name, removed_in):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            warnings.warn(
                f"{type(self).__name__}.{func.__name__} is deprecated and will be removed "
                f"in {removed_in}; use .{new_name} instead.", DeprecationWarning, stacklevel=2)
            return getattr(self, new_name)(*args, **kwargs)
        return wrapper
    return decorator
```
No second `warn_unused_param` helper — W3.7 is three inline `warnings.warn(...)` calls for three params that W3.17 deletes one minor later. Six lines beats a helper plus its own test.

**W3.5 — rename, don't re-signature.** Reserve the `predict_` prefix for methods that satisfy the enforceable contract (callable with X alone, returns 1-D ndarray of `len(X)`) and strip it from the three that cannot: `predict_ask_array` returns `(n, 3)` dollars, `predict_capacity_ratio` requires `historical_giving`, `predict_action_priority` returns a dict. Renaming costs three lines each and gives the same guarantee W3.12 enforces; moving `historical_giving`/`horizon` into `fit` and splitting the dict out would rewrite three estimators and every caller and still leave a dict pretending to be a prediction.

**W3.10 — verified.** On a fitted `MajorGiftClassifier(max_iter=10)`, `[c.estimator.n_iter_ for c in m.estimator_.calibrated_classifiers_]` → `[10, 10, 10, 10, 10]`. Replace `self.n_iter_ = 1` with `int(np.mean(...))`. Verify: `python -c "...m=MajorGiftClassifier(max_iter=10,random_state=0).fit(X,y);assert m.n_iter_==10"`. Masking a value to pass a check is the pattern CLAUDE.md forbids.

**W3.11 — don't tighten a shared validator by accident.** `philanthropy/utils/_validation.py` is 30 lines with **zero imports** and only a `1 <= month <= 12` range check; `philanthropy/preprocessing/_encounter_recency.py:157` is a *method* that additionally does an `isinstance` check. Adding `isinstance` to the shared function would need `import numpy as np` and would start raising for `CRMCleaner` and `FiscalYearTransformer` (which already call it at `transformers.py:25`) — two classes this step does not claim to touch. Minimal: keep the method, delegate only its range check to the shared function. One validator for the range, zero behaviour change.

**W3.12 — the one meta-test worth keeping.** ~60 lines, introspection only, no fixtures. Assert: (1) every subpackage in `philanthropy.__all__` declares a non-empty `__all__`, every name resolves, and each has a `docs/reference/<name>.md` (this is the enforceable version of "no undocumented public symbol" — a substring scan cannot work because `docs/reference/models.md` is 44 bytes of bare `::: philanthropy.models`); (2) walking `models.__all__ + experimental.__all__`, every public `predict_*` beyond `predict`/`predict_proba`/`predict_log_proba` matches `^predict_\w+_(score|forecast)$`, is callable with X alone, and returns a 1-D ndarray of `len(X)` — **this is what would have caught all three of W3.5's renames at authoring time; the three newest estimators are exactly the three that deviated**; (3) every `preprocessing.__all__` class defines its own `get_feature_names_out(self, input_features=None)` with `len(...) == transform(X).shape[1]`; (4) `philanthropy.__all__` covers every non-underscore subpackage directory, so a new subpackage cannot ship unreachable. Two named exemptions with a one-line reason each: `RFMTransformer` (row-reducing aggregator, returns DataFrame), `UpliftTLearner` (`fit(X, y, treatment)`). This single list replaces the four separate allowlists the draft plans proposed — W2.4's `_EXCLUDED` accounting folds in here.

**W3.17 — the breaking release, with the test files the draft missed.** Delete the four alias methods and their decorators; delete the three dead params from `__init__`; make `donor_acquisition_cost`, `cost_per_dollar_raised`, `fundraising_roi` keyword-only by inserting `*,` (`fundraising_roi(total_raised, total_fundraising_expense)` inverts its siblings' expense-first order and silently returns a reciprocal today — keyword-only is the only validation that catches it). **That break touches `tests/test_metrics.py:36,40,91` (`donor_acquisition_cost(50_000, 200)`) as well as `tests/test_concentration.py:60-61,65-66`** — both files are in scope. Rename `metrics/scoring.py`→`_scoring.py`, `preprocessing/transformers.py`→`_transformers.py`, `models/propensity.py`→`_propensity_baseline.py`, `utils/testing.py`→`_testing.py`, updating the four subpackage `__init__.py` imports and `tests/conftest.py:6`, so 1.0 does not freeze an accidental second import path.

**W3.14 — JOSS artifacts, not ceremony.** JOSS requires a deposited archive with a DOI at acceptance; there is no `.zenodo.json`, no `codemeta.json`, and `CITATION.cff` carries only the author's IEEE DOI at `:40`. Add a minimal `.zenodo.json` (title, creators, license, upload_type) and an `identifiers:` stanza in `CITATION.cff` with the concept DOI, then extend the release checklist past "publish the GitHub Release" to include the Zenodo deposit.

**Out of scope**
- Coercing `RFMTransformer.transform` to ndarray — verified a downgrade: `.to_numpy().dtype` is `object`, because `get_feature_names_out` includes the string `donor_id`. It is a row-reducing aggregator that changes `n_samples`, so it is not a Pipeline transformer at all. W3.12 exempts it by name; a docstring is the honest fix.
- `predict_uplift_score` → `predict_uplift` for returning `[-1, 1]` — satisfies the enforceable contract, is Tier 3, and the range is already in its docstring. W3.13's scale column records it for free.
- Merging `utils.make_donor_dataset` and `datasets.generate_synthetic_donor_data` — two labels and two `random_state` defaults is a smell, not a defect; both are used (`tests/conftest.py:6` + 5 tests) and `README.md:200` documents one.
- Dropping `SolicitationWindowTransformer` (`_solicitation_window.py:10`, verified `A is B`) — zero maintenance cost, non-zero breakage cost. W3.13 documents it as a supported alias.
- A general sklearn-style `@deprecated` class decorator or `_deprecate_positional_args` — nothing in this ladder deprecates a whole class or a positional arg.
- Dependency floors, action pinning, `publish.yml` gate → **W1** (`W1.1`, `W1.14`). Branch protection → deferred until W1.12 settles job names.
- Battery composition, coverage gate, `MajorGiftClassifier` runtime → **W2**. Sequence W3.10 before W2.1's battery entry — it is one line.
- README/docs content → **W1.5** (README) and **W4** (docs pages). W3.13 touches `docs/reference/index.md` only.

---

## W4 — Documentation site, tutorials, worked examples

**Goal:** every public 0.5.0 symbol has a rendered reference page and one executed worked example, and every python fence in `docs/**` runs in CI.

**Exit criteria:**
- `mkdocs build --strict`
- `python -m pytest tests/test_doc_examples.py -q --no-cov`
- `for f in experimental utils cli; do test -f docs/reference/$f.md && grep -q "reference/$f.md" mkdocs.yml || exit 1; done`
- `for f in use_the_cli ingest_unischema_events recommend_ask_amounts score_matching_gift_eligibility measure_campaign_efficiency audit_score_fairness estimate_appeal_uplift; do test -f docs/how-to/$f.md && grep -q "how-to/$f.md" mkdocs.yml || exit 1; done`
- `python -m pytest tests/test_public_api_contract.py -q` (its reference-page assertion covers all 10 subpackages)

| # | Step | Files | Verify | Effort | Blocked by |
|---|---|---|---|---|---|
| 1 | 3 reference pages (`experimental`, `utils`, `cli`) + complete both index link lists | `docs/reference/{experimental,utils,cli,index}.md`, `docs/tutorials/index.md`, `mkdocs.yml` | `mkdocs build --strict && ls site/reference/utils/index.html` | xs | — |
| 2 | Grateful-patient tutorial: replace 3 `pd.read_csv` calls with inline DataFrames | `docs/tutorials/building_a_grateful_patient_pipeline.md` | `pytest tests/test_doc_examples.py -q -rs -k grateful` | s | W1.4 |
| 3 | First-model tutorial §5: `ColumnTransformer` routing + uncomment `fit`; assert `nunique > 1` | `docs/tutorials/building_your_first_model.md` | `pytest tests/test_doc_examples.py -q -k building_your_first` | s | W1.3, W1.4 |
| 4 | `use_the_cli.md` + `ingest_unischema_events.md` (self-contained; relocates the README UniSchema fence) | `docs/how-to/{use_the_cli,ingest_unischema_events,index}.md`, `mkdocs.yml` | `pytest tests/test_doc_examples.py -q -k 'use_the_cli or unischema'` | m | W1.4, W1.5 |
| 5 | `recommend_ask_amounts.md` (`AskAmountRecommender`, `action_priority`) + `score_matching_gift_eligibility.md` | `docs/how-to/{recommend_ask_amounts,score_matching_gift_eligibility,index}.md`, `mkdocs.yml` | `pytest tests/test_doc_examples.py -q -k 'ask or matching'` | m | W1.4 |
| 6 | `measure_campaign_efficiency.md` (`load_ciob_fundraising`, gini, ROI — keyword args) + `audit_score_fairness.md` | `docs/how-to/{measure_campaign_efficiency,audit_score_fairness,index}.md`, `mkdocs.yml` | `pytest tests/test_doc_examples.py -q -k 'efficiency or fairness'` | m | W1.4 |
| 7 | `estimate_appeal_uplift.md` — `UpliftTLearner.fit(X, y, treatment)`, `[-1,1]` scale, Tier 3 caveat | `docs/how-to/{estimate_appeal_uplift,index}.md`, `mkdocs.yml` | `pytest tests/test_doc_examples.py -q -k uplift` | s | W1.4 |
| 8 | `benchmarks.md`: 5-seed sweep with mean ± spread, or an explicit single-seed caveat | `docs/explanation/benchmarks.md`, `scripts/benchmark_models.py` | `python scripts/benchmark_models.py \| head -8` | s | — |
| 9 | Link the 7 new how-tos + the CLI from the shrunk README feature tables | `README.md` | `test "$(wc -l < README.md)" -lt 200 && grep -q use_the_cli README.md` | xs | W1.5, 4–7 |

### Detail

**W4.1 — the three pages, then W3.12 keeps them honest.** `docs/reference/experimental.md` = heading + the `experimental/__init__.py:2-3` no-guarantees line + `::: philanthropy.experimental`; `utils.md` = `::: philanthropy.utils` (that is `save_model`, `load_model`, `make_donor_dataset` — `README.md:200` points readers at this subpackage with no page to land on today); `cli.md` = `::: philanthropy.cli`, which renders the module docstring documenting `train`/`score`/`validate`. Append the 6 missing links to `docs/reference/index.md:5-9` (it lists 5 of 9 existing pages) and the 2 missing tutorials to `docs/tutorials/index.md:5`. No `tests/test_docs_api_coverage.py` — a substring scan over `docs/**` cannot pass, because a mkdocstrings page contains the module path and not the symbol names; W3.12's "every subpackage has a reference page" assertion is the version that actually works.

**W4.3 — the corrected flagship, with a regression assertion.** `FiscalYearTransformer.transform` returns a 2-column ndarray and *replaces* every other column, so it can never be chained ahead of a name-based transformer — the documented serial `Pipeline` makes `WealthScreeningImputer` no-op with a warning and hands `fiscal_year` (2020) to `DischargeToSolicitationWindowTransformer` as "days since discharge", which is outside `[90, 365]`, so both outputs are 0 for every row (`unique rows: [[0. 0.]]`). Route with `ColumnTransformer` — `["gift_date"]`→`FiscalYearTransformer`, `["estimated_net_worth"]`→`WealthScreeningImputer`, `["days_since_last_discharge"]`→`DischargeToSolicitationWindowTransformer` — mirroring the pattern `README.md:443-466` already gets right. Add `assert len(set(scores.round(6))) > 1` so a future constant-feature regression fails the doc test instead of exiting 0.

**W4.6 — write the metrics how-to with keyword args from day one.** `fundraising_roi(total_raised=..., total_fundraising_expense=...)` reads correctly today *and* survives W3.17 making the three money functions keyword-only. No throwaway paragraph explaining the inverted argument order, and no rework when the signature tightens.

**W4.8 — the published numbers.** `docs/explanation/benchmarks.md:34-40` reports AUC to three decimals from one split at one seed; an auditor reproduced it exactly, which proves determinism, not meaning. A reviewer will read `0.915` as a claim about the method. Loop `scripts/benchmark_models.py` over 5 seeds and publish mean ± range, or state "single seed, single split, illustrative only" in the table caption. Either is honest; the current text is not.

**Out of scope**
- `mike` / versioned docs — no dossier finding asked for it, it is the only new dependency proposed anywhere, and it would delete the working `upload-pages-artifact`/`deploy-pages` jobs plus require a manual Pages source flip. The real problem (PyPI users on 0.3.0 reading `main`'s docs) is solved by W3.16 shipping a release.
- `lychee-action` link checking — `mkdocs build --strict` (W1.10) already fails on every broken relative link under `docs/`, and a scripted check over README/CONTRIBUTING/`docs/**` verified zero broken relative links today.
- The `DischargeToSolicitationWindowTransformer` guard — W1.3. W4.3 fixes the documented pipeline; without the guard a user can still write the same silent failure.
- `__version__` single-sourcing and the persistence-how-to rewrite — W1.2/W1.9. W4 adds no new hardcoded version strings.
- Deleting the untracked local `docs/api/` directory — `git ls-files docs/api` is empty, mkdocs ignores it, no site output.
- Rewriting `docs/explanation/benchmarks.md` numbers themselves — verified to reproduce exactly; W4.8 adds spread, it does not correct values.
- `paper/paper.md`, `CITATION.cff`, `SECURITY.md` content — verified clean and consistently 0.5.x. The `paper.bib` DOIs and the `draft-pdf.yml` PR trigger are W3.15.

---

## Release ladder

| Version | Contents | Gate |
|---|---|---|
| **0.5.1** (patch, optional) | W1.1 floors, W1.2 version, W1.3 discharge guard, W1.8 NaN guard, W1.16 ingest error, W2.3 `PropensityScorer` | `pytest tests/ -q`; `uv --resolution lowest-direct` job green (W1.13). Ship this first if anyone is installing from source — W1.1 is a silent wrong answer. |
| **0.6.0** (W3.16) | All of W1, all of W2, W3.1–W3.15. Additive for Tier 1/2: every renamed method still works and warns. **Breaking:** `experimental.LapsePredictor` deleted (Tier 3, no runway required). CHANGELOG needs an explicit **Deprecated** block naming all 4 aliases + 3 dead params with `removed_in="0.7.0"`. | `pytest tests/ -q --cov` at the new branch gate; `pytest tests/test_public_api_contract.py tests/test_deprecations.py -q`; W1.14's tag↔pyproject↔CHANGELOG gate; `twine check --strict`. **First release actually published to PyPI since 0.3.0** — fixes `pypi-release-stale-vs-docs` and unbreaks 2 of 3 `examples/*.py`. |
| **0.7.0** (W3.17) | Remove all 4 `deprecated_alias` shims and the 3 dead `__init__` params; metrics keyword-only; 4 module renames to underscore paths. | 0.6.0 must be **on PyPI** — the shims need one full published minor of overlap, not one commit. `pytest tests/ -q -x`; `! grep -rn deprecated_alias philanthropy`. |
| **1.0.0** (W3.18) | `Development Status :: 5 - Production/Stable`; Tier 1 becomes semver-protected. | All five hold: 0.7.0 published; `tests/test_public_api_contract.py` green with no exemption added since 0.7.0; `grep -rn deprecated_alias philanthropy` empty (anything still warning at 1.0 becomes a 2.0 obligation); `__version__ == importlib.metadata.version(...)` and `py.typed` in the wheel; the W3.13 tier table lists every `__all__` symbol with no Tier 1 entry mid-deprecation. |

**Deprecation overlap:** shims added in 0.6.0 (W3.5, W3.6, W3.7) are removed in 0.7.0 (W3.17). That is the only overlap window, and it is one full published minor — the calendar wait between W3.16 and W3.17 is the plan's only real-world blocker.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| W1.3's `ValueError` guard breaks a downstream pipeline that relied on the positional fallback | User code that fed a DataFrame with the days column first now raises instead of silently scoring | Line 86 is currently uncovered, so no test depends on it; `check_estimator` verified passing with the guard. Ship in 0.6.0's **Breaking** section with the `ColumnTransformer` migration snippet from W4.3. |
| W2.16's per-file 90% floor is red on arrival | `_encounter_recency.py` (90%), `transformers.py` (91%), `_rfm.py` (92%), `_encounters.py` (93%) are already at the line and branch mode drops each 3–5 points; no W2 step adds a value test to any of them | Measure the tier number on the first `branch = true` run and pin at `measured − 1`, ratcheting to 90 when someone next touches those files. Do **not** pin 90 on day one — that produces a `# pragma: no cover` spree. |
| W2.3(d) flips `PropensityScorer`'s default-threshold prediction from class 1 to class 0 | Silent behaviour change for a public `threshold` param | Required by `estimator_checks.py:2833` (`argmax(predict_proba) == predict`), arbitrary either way for a constant 0.5 scorer. Update the docstring at `propensity.py:12,22` and name it in the 0.6.0 CHANGELOG. |
| W1.5 (README cut) and W4.9 (README links) touch the same file in opposite directions | Whichever lands second is a no-op or fails the `<200 lines` gate | Single owner: W1.5 does the whole structural cut; W4.9 only appends links to the already-shrunk tables and is explicitly blocked by it. No other workstream edits `README.md`. |
| W2.14 deletes `tests/test_transformers_property.py`, which W1.11 edits and W1.13's floor job runs | Editing a deleted file; floor job errors on a missing path | Strict order W1.11 → W1.13 → W2.14, and W1.13 runs `pytest tests/ -q` (not a file list) so the deletion is invisible to it. W2.14 moves `test_encounter_transformer_no_overflow_on_extreme_span` to `tests/test_audit_regressions.py` **before** removing the file. |
| The three meta-tests (W2.10, W3.12) ship green by construction | Their author-populated registries prove nothing about today's state | Accepted and stated: they exist to catch *future* drift, not to audit the present. W2.10's registry self-populates from a `self.x_ =` source scan so it cannot rot; W3.12's exemptions are 2 names with a reason each, and W3.18 gates on none being added since 0.7.0. |
