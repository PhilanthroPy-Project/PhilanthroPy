# PhilanthroPy vs. Commercial Vendors

Most advancement shops already pay for a wealth-screening or predictive-modeling
vendor — DonorSearch, WealthEngine (Blackbaud), iWave, Windfall, and similar
platforms. This page is an honest account of where PhilanthroPy fits alongside
them, and — just as importantly — where it deliberately does **not** compete.

The short version: PhilanthroPy is a **modeling library**, not a data service.
Commercial vendors sell you *data and a hosted product*. PhilanthroPy gives you
*transparent, self-hosted code* and expects you to bring your own data. These
are different purchases that happen to overlap in the word "predictive."

## What PhilanthroPy is (and isn't)

PhilanthroPy is an open-source, MIT-licensed, scikit-learn-native library that
runs entirely on your own infrastructure. Its entire dependency footprint is
`scikit-learn`, `pandas`, and `numpy` — there is no phone-home, no API key, no
account, and no per-record billing.

What that means in practice:

* **You bring your own data.** PhilanthroPy ships zero prospect-research
  records. The bundled `generate_synthetic_donor_data` produces *synthetic*
  donors for testing and tutorials — it is not a wealth append. Estimators like
  `WealthScreeningImputerKNN` fill gaps by borrowing from *your existing
  database's* neighbors, not from a proprietary wealth index.
* **You run it.** It is a `pip install`, not a subscription. There is no hosted
  UI, no dashboards, and no managed pipeline. If your organization has no one to
  write and schedule Python, that is a real cost PhilanthroPy does not absorb.
* **It is single-maintainer OSS.** One author maintains the project under an MIT
  license. There is no support contract, no SLA, and no account team. Community
  issues and pull requests are the support model.

We would rather say this plainly up front than have you discover it in month
three.

## Comparison

The table contrasts PhilanthroPy against the *category* of commercial
wealth-screening and predictive vendors. It intentionally does not rank named
products against each other — vendor feature sets, pricing, and data coverage
change often and vary by contract, and we will not make claims about them we
cannot verify.

| Dimension | PhilanthroPy | Commercial wealth/predictive vendors |
|---|---|---|
| **Cost** | Free (MIT). Your only spend is the compute and staff time to run it. | Recurring license, typically priced per record, per seat, or per screen. |
| **Data residency / self-hosting** | Runs entirely on your infrastructure; donor data never leaves your environment. | Data is generally uploaded to and processed by the vendor's platform. |
| **Transparency / auditability** | Full source is readable and inspectable. Every score traces to code you can step through; splits are leakage-safe by construction (`FiscalYearGroupedSplitter`). | Scoring models are typically proprietary and not open to line-by-line inspection. |
| **Customization** | Unlimited — subclass, retrain, re-weight, or fork any estimator. It is your code. | Configurable within the product's options; core model logic is generally fixed. |
| **Prospect-research data included** | **None.** You supply all wealth, capacity, and biographic data yourself. | A core part of the value: bundled wealth indicators, real-estate, giving history, and appended prospect records. |
| **Support / SLA** | Community best-effort via GitHub. No guaranteed response time. | Contractual support, onboarding, training, and account management. |
| **Compliance responsibility** | Entirely yours. The library is PII-aware (see [Design Principles](design_principles.md)), but you own data-handling, retention, and regulatory posture. | Shared: the vendor operates the platform under its own compliance program, though you remain the data controller. |

## When a commercial vendor is the better call

Being candid cuts both ways. Reach for a paid vendor when:

* **You need appended data you don't have.** PhilanthroPy cannot tell you a
  prospect's real-estate holdings or estimated net worth out of thin air — it
  models signals present in *your* data. If you need external wealth records,
  that is exactly what wealth-screening vendors sell, and PhilanthroPy is not a
  substitute.
* **You have no in-house Python capacity.** A hosted product with a UI and a
  support line will get a small shop to value faster than a library that
  assumes someone can write a pipeline and read a stack trace.
* **You require a contractual SLA.** Single-maintainer OSS cannot promise a
  four-hour response. If uptime or turnaround is contractually load-bearing, buy
  it.

Many organizations run both: a vendor for the data append, PhilanthroPy for
transparent, in-house scoring on top of it.

## When PhilanthroPy is the better call

Choose PhilanthroPy when the properties a hosted product cannot give you are the
ones that matter:

* **Auditability is non-negotiable.** When leadership, a board, or a regulator
  asks *why* a donor was scored a certain way, "the vendor's model is
  proprietary" is not an answer you can give. PhilanthroPy's is code you can
  show them.
* **Data must not leave your walls.** For institutions — academic medical
  centers especially — where donor data intersects with sensitive records,
  self-hosting removes an entire class of third-party data-transfer risk.
* **You want to own the model.** No per-record fees, no lock-in, no re-pricing
  at renewal. You can retrain on your own cadence and modify anything.
* **You already have the data.** If you have a CRM export and (optionally) an
  existing wealth append, PhilanthroPy turns it into leakage-safe,
  sklearn-native scores without a new subscription. Paired with
  [UniSchema](https://github.com/PhilanthroPy-Project/UniSchema), `philanthropy.ingest`
  will even assemble the feature table from your normalized event stream.

!!! note "Not a wealth screen"
    PhilanthroPy predicts *from* data you already hold. It does not identify,
    append, or enrich prospects with external records. If a vendor's core pitch
    is "we'll tell you who your wealthy prospects are," PhilanthroPy is
    solving the next problem — modeling behavior — not that one.

!!! warning "Know what you're adopting"
    This is open-source software maintained by one person, offered as-is under
    the MIT license. It carries no warranty, no SLA, and no bundled data.
    Evaluate it the way you would any dependency you plan to run in production:
    read the source, run the tests, and decide whether your team can own it.
