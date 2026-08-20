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

## Ten lines to a ranked call list

```python
from philanthropy.datasets import generate_synthetic_donor_data
from philanthropy.models import DonorPropensityModel

df = generate_synthetic_donor_data(n_samples=500, random_state=42)
features = ["total_gift_amount", "years_active", "event_attendance_count"]
X = df[features].to_numpy()

model = DonorPropensityModel(n_estimators=200, random_state=0)
model.fit(X, df["is_major_donor"].to_numpy())
df["affinity_score"] = model.predict_affinity_score(X)   # 0-100, not a raw probability

summary = df.groupby("is_major_donor")["affinity_score"].describe()
print(summary[["count", "mean", "min", "max"]])
```

<figure class="ap-figure">
<svg viewBox="0 0 740 232" role="img" aria-labelledby="apc-title apc-desc">
  <title id="apc-title">Affinity score by donor class</title>
  <desc id="apc-desc">Range and interquartile spread of the 0 to 100 affinity score for 500 synthetic donors. Non-major donors span 0 to 39, major donors span 65 to 100, leaving an empty 26-point gap between the two groups.</desc>

  <rect class="apc-band" x="356.4" y="56" width="120.9" height="100" rx="4"/>
  <text class="apc-callout" x="416.8" y="46" text-anchor="middle">26-point gap, empty</text>

  <g data-row="major">
    <title>Major donors: n = 335 · min 65 · Q1 93.3 · median 98.5 · Q3 100 · max 100</title>
    <text class="apc-name" x="160" y="74" text-anchor="end">Major donors</text>
    <text class="apc-sub" x="160" y="91" text-anchor="end">n = 335</text>
    <line class="apc-whisker apc-focus" x1="477.3" y1="78" x2="640" y2="78"/>
    <rect class="apc-focus" x="608.6" y="69" width="31.4" height="18" rx="4"/>
    <rect class="apc-median" x="632" y="69" width="2" height="18"/>
    <text class="apc-value" x="652" y="83">65&#8211;100</text>
  </g>

  <g data-row="non-major">
    <title>Non-major donors: n = 165 · min 0 · Q1 1.5 · median 4 · Q3 16.5 · max 39</title>
    <text class="apc-name" x="160" y="130" text-anchor="end">Non-major donors</text>
    <text class="apc-sub" x="160" y="147" text-anchor="end">n = 165</text>
    <line class="apc-whisker apc-muted" x1="175" y1="134" x2="356.4" y2="134"/>
    <rect class="apc-muted" x="182" y="125" width="69.7" height="18" rx="4"/>
    <rect class="apc-median" x="192.6" y="125" width="2" height="18"/>
    <text class="apc-value" x="366" y="139">0&#8211;39</text>
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
Non-major donors top out at 39; no major donor scores below 65. That gap is the
whole product: a gift-officer call list, sorted.
</figcaption>
</figure>

??? note "Table view: the printed output and the quartiles behind the chart"

    What the snippet prints:

    ```text
                    count       mean   min    max
    is_major_donor
    0               165.0   9.636364   0.0   39.0
    1               335.0  94.865672  65.0  100.0
    ```

    The full five-number summary the chart is drawn from:

    | Group | n | Min | Q1 | Median | Q3 | Max |
    | --- | --- | --- | --- | --- | --- | --- |
    | Non-major donors | 165 | 0.0 | 1.5 | 4.0 | 16.5 | 39.0 |
    | Major donors | 335 | 65.0 | 93.25 | 98.5 | 100.0 | 100.0 |

[Run it in Colab, zero install](https://colab.research.google.com/github/PhilanthroPy-Project/PhilanthroPy/blob/main/examples/quickstart.ipynb){ .md-button .md-button--secondary }

## What is PhilanthroPy?

PhilanthroPy is a production-ready Python library that slots directly into `sklearn.pipeline.Pipeline`. It covers the full predictive workflow for nonprofit and academic medical center (AMC) fundraising — from raw CRM cleaning and wealth imputation to major-gift propensity scoring, lapse prediction, and planned-giving intent.

## Quick start

Get up and running in seconds:

!!! info "Current release: 0.6.0"
    `pip install philanthropy` gives you **0.6.0**. These docs are built from
    `main`, which also carries the merged-but-unreleased 0.7.0 and 1.0.0 work —
    see [Deprecations](reference/index.md#deprecations) for the handful of
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

Predictive fundraising in nonprofits and healthcare foundations is often dominated by proprietary, black-box vendor tools — or brittle, ad-hoc Python scripts that suffer from subtle temporal data leakage across fiscal-year boundaries. Machine-learning code built for the nuances of philanthropic giving was mostly non-existent.

PhilanthroPy exists to change that: a rigorous, open-source, **scikit-learn-compatible** foundation for donor analytics. It puts advanced fundraising data science within reach of any team, so nonprofits can use their own data to safely and effectively identify their best prospects — without relying entirely on expensive outside vendors.

---

## Key features & capabilities

A comprehensive suite of tools, easy to understand and use:

<div class="grid cards" markdown>

- :material-database-refresh: **Messy data cleaning**

    ---
    Standardises raw CRM exports (Salesforce NPSP, Raiser's Edge) — fixing dates and currency amounts without crashing. *Uses `CRMCleaner`.*

- :material-calendar-range: **Fiscal-calendar awareness**

    ---
    Nonprofits run on fiscal years (e.g. July–June). PhilanthroPy understands these boundaries natively, preventing future data from leaking into historical models. *Uses `FiscalYearTransformer`.*

- :material-currency-usd: **Smart wealth imputation**

    ---
    Third-party wealth vendors rarely match every record. This estimates missing wealth capacity (like real-estate value) from similar donors using K-nearest neighbours. *Uses `WealthScreeningImputerKNN`.*

- :material-hospital-building: **Grateful-patient featurization**

    ---
    For academic medical centers, translates clinical-encounter histories into major-gift signals while decoupling them from explicit patient identifiers (PHI). This reduces compliance risk but is **not** formal HIPAA de-identification — see [Compliance Considerations](explanation/compliance_considerations.md). *Uses `GratefulPatientFeaturizer`.*

- :material-chart-bell-curve-cumulative: **Propensity & share of wallet**

    ---
    Turn-key estimators for a donor's share of wallet (how much of their capacity goes to your cause) and the next best engagement step for a gift officer. *Uses `ShareOfWalletScorer`.*

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
