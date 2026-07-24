# Responsible Use & Compliance Considerations

PhilanthroPy's grateful-patient tooling touches data that may originate in a clinical system, so it sits close to regulated Protected Health Information (PHI). This page explains, in plain terms, what the library does and — just as importantly — what it does **not** do. It is written to help you scope a compliant deployment. It is **not legal advice**.

!!! danger "This is not legal advice"
    Nothing on this page is legal advice, and PhilanthroPy makes no representation that any workflow built with it is HIPAA-compliant. Regulations, institutional policies, and Business Associate Agreements vary. **Involve your organization's Privacy Officer and legal counsel before moving any patient-derived pipeline into production.**

## What the tooling does — and does not do — with PHI

The relevant components are `GratefulPatientFeaturizer` and `EncounterTransformer` in `philanthropy.preprocessing`. Their job is to turn clinical *encounter timing* (e.g., a most-recent discharge date and a count of encounters) into aggregate, non-identifying model features such as `days_since_last_discharge` and `encounter_frequency_score`.

**What the tooling does:**

* Derives temporal and frequency signals from encounter records you supply.
* Drops identifier-like columns from its output as a guard against accidental downstream leakage (see the next section).
* Freezes summary statistics at `fit()` time so identifiers are not carried into the fitted artifact.

**What the tooling does *not* do:**

* It does **not** de-identify your data. It never inspects cell *values*, applies a statistical disclosure method, or produces a dataset that meets any legal de-identification standard.
* It does **not** decide whether you are permitted to use clinical data for fundraising. That determination rests with you and your counsel.
* It does **not** establish, verify, or track patient authorization, opt-out status, or minimum-necessary scope.

In short: the library helps you *model* on data that you have **already** made lawful to use. Making it lawful is your responsibility, upstream of any transformer.

## Dropping columns by name is defense-in-depth, not de-identification

`EncounterTransformer` drops columns whose **names** match a set of identifier heuristics — the class attribute `PII_PATTERNS` (substrings such as `_id`, `mrn`, `ssn`, `name`, `dob`, `zip`, `patient`, `phone`, `email`, `address`), plus anything you add through the `id_cols_to_drop` parameter. As of v0.4.0 you can override the heuristic entirely with the `pii_patterns` parameter, which **replaces** (does not extend) the built-in default.

!!! warning "A name-based heuristic can only catch what it recognizes"
    This is **defense-in-depth**, not formal de-identification. The check inspects column *names* only — never the contents of any cell. It will silently miss:

    * an identifier stored under an unrecognized name (`record_key`, `acct`, `subscriber_no`);
    * an identifier hidden inside a free-text or note field;
    * quasi-identifiers (rare diagnoses, dates, small-cell geography) that are re-identifying in combination even though no single column "looks like" an ID.

    Treat the drop as a backstop that reduces the blast radius of a mistake, not as a control you can rely on to render data non-identifiable. Inspect `dropped_cols_` after `transform()` to audit exactly what was removed, and set `pii_patterns` / `id_cols_to_drop` to match your own schema.

## The two legal de-identification paths

Under the HIPAA Privacy Rule, PHI is considered de-identified only through one of two defined methods. PhilanthroPy implements **neither**; they are the responsibility of your data governance process, typically upstream of the pipeline.

* **Safe Harbor** — *45 CFR § 164.514(b).* Removal of 18 enumerated categories of identifiers (names, geographic subdivisions smaller than a state, all date elements more granular than year for dates directly related to an individual, contact details, MRNs, account numbers, biometric identifiers, and so on), *and* no actual knowledge that the residual data could identify an individual. Note that Safe Harbor's date restriction can conflict with the very encounter dates these features are built from — an example of why de-identification must be reasoned about deliberately, not assumed from a column drop.
* **Expert Determination** — *45 CFR § 164.514(a).* A person with appropriate statistical/scientific expertise determines and documents that the risk of re-identification is very small. This is the path most often used when analytically useful fields (like dates) must be retained.

If your workflow requires de-identified data, one of these methods must be applied and documented **before** the data reaches PhilanthroPy.

## The fundraising carve-out (45 CFR § 164.514(f))

HIPAA does contain a specific provision for fundraising. Under **45 CFR § 164.514(f)**, a covered entity may use or disclose a limited set of PHI to a business associate or institutionally related foundation for its own fundraising *without* individual authorization — but the provision is narrow and comes with obligations:

* The information usable for fundraising is limited (broadly: demographic data, dates of service, department of service, treating physician, outcome information, and health-insurance status) — again subject to the **minimum-necessary** principle. Pulling a full clinical record "because it might help the model" is outside the carve-out.
* Each fundraising communication must give the recipient a **clear and conspicuous opportunity to opt out**, and the covered entity may **not** condition treatment on a patient's fundraising choices.
* Once a patient opts out, further fundraising communications to that patient must stop.

PhilanthroPy does not enforce any of this. Honoring minimum-necessary scoping and opt-out status is an upstream data-governance obligation; the library will faithfully model whatever rows you hand it, opted-out or not.

## `allow_negative_days` models pre-/during-treatment solicitation

By default `EncounterTransformer(allow_negative_days=False)` coerces negative `days_since_last_discharge` values to `NaN`, discarding gifts that predate a discharge. Setting `allow_negative_days=True` **retains** those gifts — which means the resulting features can encode, and the downstream model can learn from, solicitation that occurred *before or during* active treatment.

!!! warning "`allow_negative_days=True` warns at fit time for a reason"
    As of v0.4.0, fitting with `allow_negative_days=True` emits a `UserWarning`. Soliciting a patient during an active care episode raises ethical concerns well beyond HIPAA. Professional-conduct guidance — the Association of Fundraising Professionals (AFP) *Donor Bill of Rights* and the Association for Healthcare Philanthropy (AHP) guidelines on grateful-patient programs — counsels care around the timing of clinician-referred or care-adjacent solicitation. Enable this flag only for legitimate retrospective analysis, and confirm the use with your donor-relations policy and Privacy Officer first.

## Wealth and capacity features can be proxies for protected characteristics

A model that never sees race, age, gender, or disability can still produce disparate outcomes, because wealth- and capacity-based features (estimated net worth, real-estate value, geography) frequently correlate with protected characteristics. A pipeline that scores prospects by capacity can therefore systematically advantage or disadvantage protected groups without anyone intending it.

PhilanthroPy ships a diagnostic to help you *detect* this: `philanthropy.metrics.disparate_impact_ratio`, which computes the EEOC "four-fifths rule" ratio (`min(selection_rate) / max(selection_rate)`) across groups you designate. A ratio below `0.8` is the conventional flag for adverse impact worth investigating. A companion helper, `selection_rate_by_group`, breaks the rates out per group.

!!! note "A diagnostic, not a clearance"
    `disparate_impact_ratio` surfaces disparity so a human can investigate it. It does **not** certify a model as fair or legally compliant — a passing ratio proves nothing, and both the choice of protected groups and the decision threshold materially change the number. Use it as one input to a human review, not as a gate that closes the question.

## The bottom line

PhilanthroPy does not de-identify data, does not adjudicate lawful use, and does not manage authorization or opt-out state. Its privacy features — name-based column dropping, fit-time freezing, fairness diagnostics — are defense-in-depth aids that reduce the cost of a mistake, not compliance controls you can rely on in their place.

**You are responsible** for ensuring that the data you feed the library is lawful to use for fundraising, appropriately de-identified or authorized, and scoped to the minimum necessary — and for involving your Privacy Officer and counsel before production.
