<p align="center">
  <img src="docs/assets/logo.png" alt="PhilanthroPy logo" width="180"/>
</p>

<p align="center">
  <strong>PhilanthroPy: Code for a cause—predictive analytics for advancement teams.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/philanthropy/"><img src="https://img.shields.io/pypi/v/philanthropy?color=blue" alt="PyPI version"/></a>
  <img src="https://img.shields.io/pypi/pyversions/philanthropy" alt="Python versions"/>
  <a href="https://github.com/PhilanthroPy-Project/PhilanthroPy/actions/workflows/ci.yml"><img src="https://github.com/PhilanthroPy-Project/PhilanthroPy/actions/workflows/ci.yml/badge.svg" alt="Tests"/></a>
  <img src="https://img.shields.io/badge/coverage-%E2%89%A592%25-brightgreen" alt="Coverage at least 92 percent"/>
  <img src="https://img.shields.io/badge/sklearn-compatible-orange" alt="sklearn compatible"/>
  <a href="https://PhilanthroPy-Project.github.io/PhilanthroPy/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-informational" alt="documentation"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/pypi/l/philanthropy?color=green" alt="License"/></a>
</p>

<p align="center">
  <strong><a href="https://PhilanthroPy-Project.github.io/PhilanthroPy/">🚀 View the Full Documentation Site</a></strong>
</p>

---

## What is PhilanthroPy?

PhilanthroPy is a Python library that slots directly into `sklearn.pipeline.Pipeline`. It covers the full predictive workflow for nonprofit and academic medical center (AMC) fundraising — from raw CRM cleaning and wealth imputation to major-gift propensity scoring, lapse prediction, and planned-giving intent.

### Who it's for

One toolkit, two audiences:

- **General nonprofit & university advancement teams** (no PHI in scope) — CRM cleaning, RFM segmentation, wealth-screening imputation, and donor-propensity / lapse / planned-giving scoring. Start with `CRMCleaner`, `RFMTransformer`, `WealthScreeningImputer`, and `DonorPropensityModel`.
- **Academic medical center (AMC) foundations running grateful-patient programs** (PHI in scope, higher scrutiny) — clinical-encounter featurization via `EncounterTransformer`, `GratefulPatientFeaturizer`, and `DischargeToSolicitationWindowTransformer`. **Before production use, read [Compliance Considerations](docs/explanation/compliance_considerations.md):** the PII handling here is a name-based heuristic, *not* formal HIPAA de-identification.

### Maturity

Single-maintainer MIT project. **`pip install philanthropy` gives you `0.6.0`**, the current release.

`main` is ahead of it: `0.7.0` (removes the `0.6.0` deprecations) and `1.0.0` (freezes the API) are merged and green but deliberately unreleased, so the `0.6.0` deprecation warnings get a real migration window rather than a token one. Read the [CHANGELOG](CHANGELOG.md) for what is queued.

Preprocessing and the core classifiers are Tier 1; grateful-patient featurization and `philanthropy.ingest` are Tier 2 (Beta); `FinancialForecastModel` and `philanthropy.experimental.*` are Tier 3 (Experimental) and carry no API guarantees. From `1.0.0`, Tier 1 becomes semver-protected — breaking one requires a major release preceded by a full published minor of `DeprecationWarning`. Per-symbol tiers are in the [API reference](docs/reference/index.md).

> **Maintenance:** maintained by one person on a best-effort basis. For vendor / OSS risk reviews: the bus factor is 1.

---

## Installation

```bash
pip install philanthropy
```

<details>
<summary>From source (for development)</summary>

```bash
git clone https://github.com/PhilanthroPy-Project/PhilanthroPy.git
cd PhilanthroPy
pip install -e ".[dev]"
```
</details>

---

## Quick Start

```python
from philanthropy.datasets import generate_synthetic_donor_data
from philanthropy.models import DonorPropensityModel

df = generate_synthetic_donor_data(n_samples=500, random_state=42)
X = df[["total_gift_amount", "years_active", "event_attendance_count"]].to_numpy()
y = df["is_major_donor"].to_numpy()

model = DonorPropensityModel(n_estimators=200, random_state=0)
model.fit(X, y)
scores = model.predict_affinity_score(X)   # 0–100 affinity scale

assert scores.shape == (500,)
assert len(set(scores.round(6))) > 1       # a constant score means a broken pipeline
```

> **Try it now — zero install:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PhilanthroPy-Project/PhilanthroPy/blob/main/examples/quickstart.ipynb)
>
> **Runnable scripts:** [`examples/quickstart.py`](examples/quickstart.py) and [`examples/unischema_to_scores.py`](examples/unischema_to_scores.py) run end to end and are smoke-tested in CI.

<p align="center">
  <img src="docs/assets/affinity_distribution.png" alt="Affinity score distribution separating major from non-major donors" width="640"/>
  <br/>
  <em>Output of <code>plot_affinity_distribution()</code>: the 0–100 affinity scores cleanly separate major from non-major donors.</em>
</p>

---

## From your CRM to scores

`philanthropy.ingest` is the on-ramp: it turns what your donor system already emits into the donor-level feature table the estimators expect, with no glue code in between.

**CiviCRM.** A contribution export (or an APIv4 `Contribution.get` result) → `read_civicrm_contributions()` → `civicrm_contributions_to_features()` → `predict_affinity_score()`. The bridge drops payment-processor test transactions and counts only `Completed` contributions, which is the difference between a lifetime-giving number you can brief a gift officer on and one inflated by refunds. Worked version: **[Ingest CiviCRM contributions](docs/how-to/ingest_civicrm_contributions.md)**.

**UniSchema.** PhilanthroPy is also the modeling half of an ecosystem. [UniSchema](https://github.com/PhilanthroPy-Project/UniSchema) normalizes fragmented advancement webhooks (GiveCampus, Slate, NPSP, Cvent, …) into a single `ConstituentEvent` stream. Webhooks → UniSchema egress → `read_constituent_events()` → `constituent_events_to_features()` → `predict_affinity_score()`. Worked, runnable version with the full diagram: **[Ingest UniSchema events](docs/how-to/ingest_unischema_events.md)**.

---

## Feature overview

Full parameter documentation for every symbol below is rendered in the [API reference](https://PhilanthroPy-Project.github.io/PhilanthroPy/reference/).

### 🧹 Preprocessing

| Transformer | Description |
|---|---|
| `CRMCleaner` | Standardise raw CRM exports — coerce `gift_date` to `datetime64` and `gift_amount` to `float64` |
| `WealthScreeningImputer` | Leakage-safe wealth imputation (median / mean / zero), fill stats frozen at `fit()` |
| `WealthScreeningImputerKNN` | Leakage-safe KNN imputation for wealth-screening vendor columns |
| `WealthPercentileTransformer` | Per-column wealth percentile rank (0–100); NaN-in → NaN-out |
| `FiscalYearTransformer` | Fiscal year & quarter from gift dates; configurable start month |
| `RFMTransformer` | Recency–Frequency–Monetary features for donor segmentation |
| `ShareOfWalletScorer` | Normalised Share-of-Wallet score + `capacity_tier` encoding |
| `MatchingGiftFeaturizer` | Employer matching-gift eligibility and expected-match features |
| `EncounterTransformer` | Bridge EHR encounters with the CRM; drops identifier-like columns by name |
| `EncounterRecencyTransformer` | Encounter-date columns → predictive recency features |
| `GratefulPatientFeaturizer` | Clinical gravity score + service-line capacity weights |
| `DischargeToSolicitationWindowTransformer` | `in_solicitation_window` (0/1) and `window_position_score` [0,1] |
| `SolicitationWindowTransformer` | Supported alias of `DischargeToSolicitationWindowTransformer` |
| `PlannedGivingSignalTransformer` | Bequest / legacy-gift intent vector |

### 🤖 Models

| Model | Description |
|---|---|
| `DonorPropensityModel` | Random Forest with `predict_affinity_score()` on a 0–100 scale |
| `MajorGiftClassifier` | Calibrated `HistGradientBoostingClassifier` — NaN-native |
| `LapsePredictor` | Random Forest for donor lapse, with `predict_lapse_score()` |
| `PlannedGivingIntentScorer` | Calibrated bequest-intent scorer, `predict_intent_score()` |
| `ShareOfWalletRegressor` | Total giving capacity and untapped-potential ratio |
| `AskAmountRecommender` | Conservative / target / stretch ask ladder via `ask_ladder()` |
| `MovesManagementClassifier` | Multi-class portfolio stage predictor |
| `FinancialForecastModel` | Hybrid LSTM-ARIMA revenue forecaster, dependency-free |
| `PropensityScorer` | Constant-probability baseline — a floor to beat, not a scorer |

### 📊 Metrics, splitters, and the rest

| Symbol | Module | Description |
|---|---|---|
| `donor_lifetime_value` | `metrics` | Discounted LTV annuity |
| `donor_retention_rate`, `donor_acquisition_cost` | `metrics` | Core campaign KPIs |
| `cost_per_dollar_raised`, `fundraising_roi` | `metrics` | Campaign efficiency |
| `gift_concentration_gini`, `top_donor_share` | `metrics` | Portfolio concentration |
| `disparate_impact_ratio`, `selection_rate_by_group` | `metrics` | Four-fifths-rule fairness audit |
| `FiscalYearGroupedSplitter` | `model_selection` | Walk-forward fiscal-year CV |
| `donor_feature_importance` | `inspection` | Permutation importance for any fitted estimator |
| `constituent_events_to_features`, `read_constituent_events` | `ingest` | UniSchema bridge |
| `civicrm_contributions_to_features`, `read_civicrm_contributions` | `ingest` | CiviCRM contribution-export bridge |
| `generate_synthetic_donor_data`, `load_ciob_fundraising` | `datasets` | Synthetic pool and a real CIOB series |
| `make_donor_dataset`, `save_model`, `load_model` | `utils` | Labelled fixtures and pipeline persistence |
| `plot_affinity_distribution`, `plot_retention_waterfall` | `visualisation` | Matplotlib is imported lazily, per function |
| `UpliftTLearner` | `experimental` | T-learner appeal uplift — no API guarantees |

---

## Guides

**Tutorials** — [Building your first model](docs/tutorials/building_your_first_model.md) · [Avoiding temporal data leakage](docs/tutorials/avoiding_temporal_data_leakage.md) · [Building a grateful-patient pipeline](docs/tutorials/building_a_grateful_patient_pipeline.md)

**How-to** — [Use the CLI](docs/how-to/use_the_cli.md) · [Ingest UniSchema events](docs/how-to/ingest_unischema_events.md) · [Ingest CiviCRM contributions](docs/how-to/ingest_civicrm_contributions.md) · [Handle missing wealth data](docs/how-to/handle_missing_wealth_data.md) · [Build grateful-patient features](docs/how-to/build_grateful_patient_features.md) · [Recommend ask amounts](docs/how-to/recommend_ask_amounts.md) · [Score matching-gift eligibility](docs/how-to/score_matching_gift_eligibility.md) · [Measure campaign efficiency](docs/how-to/measure_campaign_efficiency.md) · [Audit score fairness](docs/how-to/audit_score_fairness.md) · [Estimate appeal uplift](docs/how-to/estimate_appeal_uplift.md) · [Save and load models](docs/how-to/save_and_load_models.md) · [Develop and test](docs/how-to/develop_and_test.md)

**Explanation** — [Design principles](docs/explanation/design_principles.md) · [Capacity and loyalty](docs/explanation/capacity_and_loyalty.md) · [Fundraising metrics](docs/explanation/fundraising_metrics.md) · [Compliance considerations](docs/explanation/compliance_considerations.md) · [Benchmarks](docs/explanation/benchmarks.md)

---

## Roadmap

### 🔜 Next

- `philanthropy.visualisation.plot_capacity_heatmap()`
- `EnsemblePropensityModel` (stacked LapsePredictor + DonorPropensityModel)

---

## Research

S. A. Lalakiya, "AI for Advancement: Predictive Donor Analytics and Fundraising
Intelligence at Scale," *2025 IEEE 11th ICCED*, IEEE, 2025,
doi: [10.1109/ICCED68324.2025.11325064](https://doi.org/10.1109/ICCED68324.2025.11325064).

This is the library author's own related work on the same problem space, using a
different dataset and its own models. **It is not an independent evaluation or a
benchmark of PhilanthroPy.** To cite the software itself, see [`CITATION.cff`](CITATION.cff).

---

## Generative AI disclosure

AI assistance (Claude Code) was used during development of this package, across
implementation, tests, and documentation, in an agentic workflow rather than
line completion alone.

**What was not generated.** The design constraints are the author's and predate
any generated code: the leakage-safety contract (every fitted statistic is
computed on training data inside `fit` and frozen before `transform`/`predict`),
the dependency rule (scikit-learn, pandas, numpy, matplotlib, seaborn — no deep
learning frameworks), the estimator conventions, and the stability tiers.
Generated code that violated them was rejected rather than merged.

**Human review.** Nothing lands without the full gate green. Locally, `make ci`
runs flake8, mypy, the docstring examples, and the test suite against a 92%
coverage floor (`pyproject.toml`). CI additionally enforces a 93% coverage floor
on the risk-tier subtree, runs the suite across an OS and Python-version matrix,
installs at the declared dependency floors on Python 3.9, builds the
distributions and checks their metadata with `twine`, and verifies the package
imports without a plotting stack installed. Every public estimator passes
`sklearn.utils.estimator_checks.check_estimator`.

No approximate scale is attached to the AI use here. The author has not measured
the split and will not estimate one; the mechanism and the review gate above are
stated instead. This follows the
[pyOpenSci generative AI policy](https://www.pyopensci.org/blog/generative-ai-peer-review-policy.html).

---

## Contributing

Contributions are welcome, and a first PR does not need to be big — docs fixes,
missing tests, and clearer error messages all count.

**Start with a [good first issue](https://github.com/PhilanthroPy-Project/PhilanthroPy/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).**
Each one names the files to touch, the steps, and the single command that proves
it is done. Comment on the issue to claim it; ask there if anything is unclear.

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the fork-and-PR workflow, the full
local test gate, and pre-push hook setup. In short: fork, branch, run `make ci`
before every push, and never use `git push --no-verify`. Setup plus a first green
`make ci` takes about eight minutes.

Everyone who has landed a change is credited in
[CONTRIBUTORS.md](CONTRIBUTORS.md) — add yourself in the same PR.

Questions are welcome in
[Discussions](https://github.com/PhilanthroPy-Project/PhilanthroPy/discussions).

---

## License

MIT License — see `LICENSE` for details.
