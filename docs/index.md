---
description: "scikit-learn estimators for nonprofit and hospital fundraising analytics: donor propensity, lapse risk, wealth-screening imputation, leakage-safe by design."
hide:
  - toc
---

<div class="ap-hero" markdown>

<div class="ap-hero__copy" markdown>

# Find out whether a model beats the rule your shop already uses, before you pay for one. { .ap-hero__title }

<p class="ap-hero__sub">PhilanthroPy scores your donors for major-gift propensity, lapse risk, and more, then checks the score against the rule your shop already runs without a model (rank by past giving, ask for what they gave last time), so you can see whether a model is worth adopting before you commit to one.</p>

<div class="ap-cta" markdown>
[See the results](results/index.md){ .md-button }
[Try it on your file](tutorials/index.md){ .md-button .md-button--secondary }
[For developers](#for-developers){ .md-button .md-button--secondary }
</div>

</div>

<div class="ap-hero__ledger">
<table class="ap-ledger">
<caption>Sample ranked output</caption>
<colgroup>
<col style="width: 20%"><col style="width: 45%"><col style="width: 35%">
</colgroup>
<thead>
<tr><th scope="col">#</th><th scope="col">Donor ID</th><th scope="col">Gave again?</th></tr>
</thead>
<tbody>
<tr><td>1</td><td>D-04821</td><td>Yes</td></tr>
<tr><td>2</td><td>D-01193</td><td>Yes</td></tr>
<tr><td>3</td><td>D-07750</td><td>Yes</td></tr>
<tr><td>4</td><td>D-02264</td><td>No</td></tr>
<tr><td>5</td><td>D-05531</td><td>Yes</td></tr>
<tr><td>6</td><td>D-09902</td><td>No</td></tr>
</tbody>
</table>
<p class="ap-hero__caption">Six of the top picks from a held-out donor list, ranked by <code>predict_affinity_score</code>. See the <a href="results/index.md">Results</a> for whether that beats the rule your shop already uses.</p>
</div>

</div>

![How each model compares to the simple rule your shop already uses: one dot per dataset, left of center is worse than the rule, right is better](assets/results/scoreboard.png#only-light)
![How each model compares to the simple rule your shop already uses: one dot per dataset, left of center is worse than the rule, right is better](assets/results/scoreboard-dark.png#only-dark)

## What is PhilanthroPy?

PhilanthroPy is a production-ready Python library that slots directly into `sklearn.pipeline.Pipeline`. It covers the full predictive workflow for nonprofit and academic medical center (AMC) fundraising, from raw CRM cleaning and wealth imputation to major-gift propensity scoring, lapse prediction, and planned-giving intent.

## For developers

<div class="ap-specs">
  <div class="ap-specs__item"><span class="ap-specs__k ap-specs__k--ok">Leakage-safe</span><span class="ap-specs__v">train-only statistics, frozen before transform</span></div>
  <div class="ap-specs__item"><span class="ap-specs__k ap-specs__k--ok">check_estimator</span><span class="ap-specs__v">every Tier 1/2 estimator passes scikit-learn's compliance suite</span></div>
  <div class="ap-specs__item"><span class="ap-specs__k ap-specs__k--ok">Pipeline-ready</span><span class="ap-specs__v">drops into sklearn.pipeline.Pipeline</span></div>
  <div class="ap-specs__item"><span class="ap-specs__k ap-specs__k--ok">MIT</span><span class="ap-specs__v">open source, no vendor lock-in</span></div>
</div>

### A ranked call list, scored honestly

```python
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from philanthropy.datasets import generate_synthetic_donor_data
from philanthropy.models import DonorPropensityModel

df = generate_synthetic_donor_data(n_samples=2000, random_state=42)
X = df[["total_gift_amount", "years_active", "event_attendance_count"]].to_numpy()
y = df["is_major_donor"].to_numpy()

# Split BEFORE fitting. Scoring the rows you trained on tells you nothing.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

model = DonorPropensityModel(n_estimators=200, random_state=0)
model.fit(X_train, y_train)

scores = model.predict_affinity_score(X_test)   # 0-100, not a raw probability
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f"held-out ROC-AUC: {auc:.3f}")
print(pd.Series(scores).groupby(y_test).describe()[["count", "mean", "min", "max"]])
```

<figure class="ap-figure">
<p class="ap-stat"><span class="ap-stat__v">0.841</span><span class="ap-stat__l">held-out ROC-AUC, 500 donors the model never saw</span></p>
<svg viewBox="0 0 740 232" role="img" aria-labelledby="apc-title apc-desc">
  <title id="apc-title">Held-out affinity score by donor class</title>
  <desc id="apc-desc">Range and interquartile spread of the 0 to 100 affinity score for 500 held-out donors. Both groups span the full scale at their extremes, but the middle half of non-major donors sits between 3.0 and 35.0 while the middle half of major donors sits between 38.5 and 83.25, 3.5 points apart.</desc>

  <rect class="apc-band" x="337.75" y="56" width="16.3" height="100" rx="4"/>
  <text class="apc-callout" x="345.9" y="46" text-anchor="middle">middle halves 3.5 points apart</text>

  <g data-row="major">
    <title>Major donors: n = 183 · min 1 · Q1 38.5 · median 62 · Q3 83.25 · max 100</title>
    <text class="apc-name" x="160" y="74" text-anchor="end">Major donors</text>
    <text class="apc-sub" x="160" y="91" text-anchor="end">n = 183</text>
    <line class="apc-whisker apc-focus" x1="179.65" y1="78" x2="640" y2="78"/>
    <rect class="apc-focus" x="354.025" y="69" width="208.1" height="18" rx="4"/>
    <rect class="apc-median" x="462.3" y="69" width="2" height="18"/>
    <text class="apc-value" x="652" y="83">median 62</text>
  </g>

  <g data-row="non-major">
    <title>Non-major donors: n = 317 · min 0 · Q1 3 · median 14 · Q3 35 · max 96</title>
    <text class="apc-name" x="160" y="130" text-anchor="end">Non-major donors</text>
    <text class="apc-sub" x="160" y="147" text-anchor="end">n = 317</text>
    <line class="apc-whisker apc-muted" x1="175" y1="134" x2="621.4" y2="134"/>
    <rect class="apc-muted" x="188.95" y="125" width="148.8" height="18" rx="4"/>
    <rect class="apc-median" x="239.1" y="125" width="2" height="18"/>
    <text class="apc-value" x="652" y="139">median 14</text>
  </g>

  <line class="apc-axis" x1="175" y1="176" x2="640" y2="176"/>
  <text class="apc-tick" x="175" y="194" text-anchor="middle">0</text>
  <text class="apc-tick" x="291.3" y="194" text-anchor="middle">25</text>
  <text class="apc-tick" x="407.5" y="194" text-anchor="middle">50</text>
  <text class="apc-tick" x="523.8" y="194" text-anchor="middle">75</text>
  <text class="apc-tick" x="640" y="194" text-anchor="middle">100</text>
  <text class="apc-sub" x="407.5" y="220" text-anchor="middle">Affinity score</text>
</svg>
<figcaption>
Bar = interquartile range, notch = median, line = full min-to-max range.
The tails overlap: a few non-major donors score 96 and a few majors score 1.
The middles do not, and that is what a call list needs. Rank by score, work
down the list. Fit on the rows you score and the two groups separate perfectly,
which is the model reciting its training set, not a result.
</figcaption>
</figure>

??? note "Table view: the printed output and the quartiles behind the chart"

    What the snippet prints:

    ```text
    held-out ROC-AUC: 0.841
       count       mean  min   max
    0  317.0  22.069401  0.0  96.0
    1  183.0  58.704918  1.0  100.0
    ```

    The full five-number summary the chart is drawn from:

    | Group | n | Min | Q1 | Median | Q3 | Max |
    | --- | --- | --- | --- | --- | --- | --- |
    | Non-major donors | 317 | 0.0 | 3.0 | 14.0 | 35.0 | 96.0 |
    | Major donors | 183 | 1.0 | 38.5 | 62.0 | 83.25 | 100.0 |

[Run it in Colab, zero install](https://colab.research.google.com/github/PhilanthroPy-Project/PhilanthroPy/blob/main/examples/notebooks/01_quickstart_propensity.ipynb){ .md-button .md-button--secondary }

## Quick start

Get up and running in seconds:

!!! info "Current release: 0.7.0"
    `pip install philanthropy` gives you **0.7.0**. These docs are built from
    `main`, which also carries the merged-but-unreleased 1.0.0 work.
    See [Deprecations](reference/index.md#deprecations) for the handful of
    differences that affect you today.

=== "pip"
    ```bash
    pip install philanthropy
    ```

=== "from source"
    ```bash
    git clone https://github.com/PhilanthroPy-Project/PhilanthroPy.git
    cd PhilanthroPy
    pip install -e ".[dev]"
    ```

---

## Motivation

Predictive fundraising in nonprofits and healthcare foundations is often dominated by proprietary, black-box vendor tools, or brittle, ad-hoc Python scripts that suffer from subtle temporal data leakage across fiscal-year boundaries. Machine-learning code built for the nuances of philanthropic giving was mostly non-existent.

PhilanthroPy exists to change that: a rigorous, open-source, **scikit-learn-compatible** foundation for donor analytics. It puts advanced fundraising data science within reach of any team, so nonprofits can use their own data to safely and effectively identify their best prospects, without relying entirely on expensive outside vendors.

---

## Key features & capabilities

A comprehensive suite of tools, easy to understand and use:

<div class="ap-ledger-list" markdown>

<div class="ap-ledger-row" markdown>
**Messy data cleaning.** Standardises raw CRM exports (Salesforce NPSP, Raiser's Edge), fixing dates and currency amounts without crashing.

`CRMCleaner`
</div>

<div class="ap-ledger-row" markdown>
**Fiscal-calendar awareness.** Nonprofits run on fiscal years (e.g. July–June). PhilanthroPy understands these boundaries natively, preventing future data from leaking into historical models.

`FiscalYearTransformer`
</div>

<div class="ap-ledger-row" markdown>
**Smart wealth imputation.** Third-party wealth vendors rarely match every record. This estimates missing wealth capacity (like real-estate value) from similar donors using K-nearest neighbours.

`WealthScreeningImputerKNN`
</div>

<div class="ap-ledger-row" markdown>
**Grateful-patient featurization.** For academic medical centers, translates clinical-encounter histories into major-gift signals while decoupling them from explicit patient identifiers (PHI). This reduces compliance risk but is **not** formal HIPAA de-identification. See [Compliance Considerations](explanation/compliance_considerations.md).

`GratefulPatientFeaturizer`
</div>

<div class="ap-ledger-row" markdown>
**Propensity & share of wallet.** Estimators for capacity utilisation (what share of a donor's modelled wealth is estimated philanthropic capacity, **not** what share of their giving you receive) and the next best engagement step for a gift officer.

`ShareOfWalletScorer`
</div>

<div class="ap-ledger-row" markdown>
**Multi-source uploads, one upgrade model.** Maps and folds any number of tagged CRM/engagement exports (gifts, event attendance, volunteer hours, ...) into one donor table, then trains and scores who is likely to move from mid-level giving to your leadership threshold next fiscal year, all cut at an `as_of` date so nothing dated after it can leak in.

`build_leadership_snapshots`, `score_leadership_prospects`
</div>

</div>

!!! tip "Getting started"
    The quickest way to get familiar with PhilanthroPy is to dive into the **[Tutorials](tutorials/index.md)**.

## Explore the docs

<div class="ap-ledger-list ap-ledger-list--nav" markdown>

<div class="ap-ledger-row" markdown>
[**Tutorials**](tutorials/index.md)

Step-by-step, learning-oriented lessons for beginners.
</div>

<div class="ap-ledger-row" markdown>
[**How-To Guides**](how-to/index.md)

Goal-oriented recipes for specific tasks.
</div>

<div class="ap-ledger-row" markdown>
[**Explanation**](explanation/index.md)

Understanding-oriented concepts and architecture.
</div>

<div class="ap-ledger-row" markdown>
[**API Reference**](reference/index.md)

Information-oriented API docs.
</div>

</div>
