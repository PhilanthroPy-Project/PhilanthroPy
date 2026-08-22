<div class="ap-hero" markdown>
<span class="ap-hero__eyebrow">Open source · scikit-learn native</span>

# Predictive donor analytics, done right. { .ap-hero__title }

<p class="ap-hero__sub">A leakage-safe, pipeline-ready toolkit for nonprofit and academic-medical-center fundraising. Every Tier 1/2 estimator passes scikit-learn's <code>check_estimator</code>.</p>

<div class="ap-cta" markdown>
[Get started](tutorials/index.md){ .md-button }
[View on GitHub](https://github.com/PhilanthroPy-Project/PhilanthroPy){ .md-button .md-button--secondary }
</div>

</div>

<div class="ap-specs">
  <div class="ap-specs__item"><span class="ap-specs__k ap-specs__k--ok">Leakage-safe</span><span class="ap-specs__v">train-only statistics, frozen before transform</span></div>
  <div class="ap-specs__item"><span class="ap-specs__k ap-specs__k--ok">check_estimator</span><span class="ap-specs__v">every Tier 1/2 estimator passes scikit-learn's compliance suite</span></div>
  <div class="ap-specs__item"><span class="ap-specs__k ap-specs__k--ok">Pipeline-ready</span><span class="ap-specs__v">drops into sklearn.pipeline.Pipeline</span></div>
  <div class="ap-specs__item"><span class="ap-specs__k ap-specs__k--ok">MIT</span><span class="ap-specs__v">open source, no vendor lock-in</span></div>
</div>

## A ranked call list, scored honestly

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
<p class="ap-stat"><span class="ap-stat__v">0.932</span><span class="ap-stat__l">held-out ROC-AUC, 500 donors the model never saw</span></p>
<svg viewBox="0 0 740 232" role="img" aria-labelledby="apc-title apc-desc">
  <title id="apc-title">Held-out affinity score by donor class</title>
  <desc id="apc-desc">Range and interquartile spread of the 0 to 100 affinity score for 500 held-out donors. Both groups span the full scale at their extremes, but the middle half of non-major donors sits between 1.5 and 38.5 while the middle half of major donors sits between 85.5 and 99.5, 47 points apart.</desc>

  <rect class="apc-band" x="354" y="56" width="218.6" height="100" rx="4"/>
  <text class="apc-callout" x="463.3" y="46" text-anchor="middle">middle halves 47 points apart</text>

  <g data-row="major">
    <title>Major donors: n = 347 · min 18 · Q1 85.5 · median 96.5 · Q3 99.5 · max 100</title>
    <text class="apc-name" x="160" y="74" text-anchor="end">Major donors</text>
    <text class="apc-sub" x="160" y="91" text-anchor="end">n = 347</text>
    <line class="apc-whisker apc-focus" x1="258.7" y1="78" x2="640" y2="78"/>
    <rect class="apc-focus" x="572.6" y="69" width="65.1" height="18" rx="4"/>
    <rect class="apc-median" x="622.7" y="69" width="2" height="18"/>
    <text class="apc-value" x="652" y="83">median 96.5</text>
  </g>

  <g data-row="non-major">
    <title>Non-major donors: n = 153 · min 0 · Q1 1.5 · median 8.5 · Q3 38.5 · max 100</title>
    <text class="apc-name" x="160" y="130" text-anchor="end">Non-major donors</text>
    <text class="apc-sub" x="160" y="147" text-anchor="end">n = 153</text>
    <line class="apc-whisker apc-muted" x1="175" y1="134" x2="640" y2="134"/>
    <rect class="apc-muted" x="182" y="125" width="172" height="18" rx="4"/>
    <rect class="apc-median" x="213.5" y="125" width="2" height="18"/>
    <text class="apc-value" x="652" y="139">median 8.5</text>
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
The tails overlap: a few non-major donors score 100 and a few majors score 18.
The middles do not, and that is what a call list needs. Rank by score, work
down the list. Fit on the rows you score and the two groups separate perfectly,
which is the model reciting its training set, not a result.
</figcaption>
</figure>

??? note "Table view: the printed output and the quartiles behind the chart"

    What the snippet prints:

    ```text
    held-out ROC-AUC: 0.932
       count       mean   min    max
    0  153.0  25.019608   0.0  100.0
    1  347.0  88.665706  18.0  100.0
    ```

    The full five-number summary the chart is drawn from:

    | Group | n | Min | Q1 | Median | Q3 | Max |
    | --- | --- | --- | --- | --- | --- | --- |
    | Non-major donors | 153 | 0.0 | 1.5 | 8.5 | 38.5 | 100.0 |
    | Major donors | 347 | 18.0 | 85.5 | 96.5 | 99.5 | 100.0 |

[Run it in Colab, zero install](https://colab.research.google.com/github/PhilanthroPy-Project/PhilanthroPy/blob/main/examples/quickstart.ipynb){ .md-button .md-button--secondary }

## What is PhilanthroPy?

PhilanthroPy is a production-ready Python library that slots directly into `sklearn.pipeline.Pipeline`. It covers the full predictive workflow for nonprofit and academic medical center (AMC) fundraising, from raw CRM cleaning and wealth imputation to major-gift propensity scoring, lapse prediction, and planned-giving intent.

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

<div class="grid cards" markdown>

- :material-database-refresh: **Messy data cleaning**

    ---
    Standardises raw CRM exports (Salesforce NPSP, Raiser's Edge), fixing dates and currency amounts without crashing. *Uses `CRMCleaner`.*

- :material-calendar-range: **Fiscal-calendar awareness**

    ---
    Nonprofits run on fiscal years (e.g. July–June). PhilanthroPy understands these boundaries natively, preventing future data from leaking into historical models. *Uses `FiscalYearTransformer`.*

- :material-currency-usd: **Smart wealth imputation**

    ---
    Third-party wealth vendors rarely match every record. This estimates missing wealth capacity (like real-estate value) from similar donors using K-nearest neighbours. *Uses `WealthScreeningImputerKNN`.*

- :material-hospital-building: **Grateful-patient featurization**

    ---
    For academic medical centers, translates clinical-encounter histories into major-gift signals while decoupling them from explicit patient identifiers (PHI). This reduces compliance risk but is **not** formal HIPAA de-identification. See [Compliance Considerations](explanation/compliance_considerations.md). *Uses `GratefulPatientFeaturizer`.*

- :material-chart-bell-curve-cumulative: **Propensity & share of wallet**

    ---
    Estimators for capacity utilisation (what share of a donor's modelled wealth is estimated philanthropic capacity, **not** what share of their giving you receive) and the next best engagement step for a gift officer. *Uses `ShareOfWalletScorer`.*

</div>

!!! tip "Getting started"
    The quickest way to get familiar with PhilanthroPy is to dive into the **[Tutorials](tutorials/index.md)**.

## Explore the docs

<div class="grid cards" markdown>

- **[Tutorials](tutorials/index.md)**

    ---
    Step-by-step, learning-oriented lessons for beginners.

- **[How-To Guides](how-to/index.md)**

    ---
    Goal-oriented recipes for specific tasks.

- **[Explanation](explanation/index.md)**

    ---
    Understanding-oriented concepts and architecture.

- **[API Reference](reference/index.md)**

    ---
    Information-oriented API docs.

</div>
