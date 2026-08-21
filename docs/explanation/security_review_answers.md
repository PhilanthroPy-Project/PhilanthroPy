# Security review Q&A

A one-page answer sheet for the questions an institutional security, privacy, or
procurement review asks before a nonprofit or academic medical center runs
PhilanthroPy against real donor data.

This page exists to be **forwarded**. If your privacy officer or IT security team
needs to sign off, send them this page rather than the whole documentation site.
For the regulatory reasoning behind the de-identification and fundraising-carve-out
answers, see [Compliance considerations](compliance_considerations.md).

---

## 1. Does the software send data anywhere?

**No.** PhilanthroPy never sends your data anywhere: no telemetry, no usage
analytics, no license check, no phone-home, and no third-party data append. It
also downloads nothing: there is no automatic model or dataset download, and
nothing is fetched at import time or inside `fit` / `transform`.

The package imports no network client at all: there is no `requests`,
`urllib.request`, `httpx`, `aiohttp`, or raw `socket` use anywhere in
`philanthropy/`. (`urllib.parse` does appear, in
`philanthropy.utils._validation.ensure_local_path`; it is pure string
manipulation and is what *rejects* remote paths, see question 1a.) The only URLs
in the source are citations in docstrings. The one bundled dataset
(`load_ciob_fundraising`) is a CSV vendored inside the wheel and read via
`importlib.resources`.

Both properties are enforced in CI rather than merely asserted here.
`tests/test_no_network.py` poisons every socket entry point and runs a full
train/score cycle plus a CRM ingest, and separately parses every module in the
package and fails the build if one imports a network-capable library without
appearing on an explicit allowlist (`_NETWORK_ALLOWED`). **That allowlist is
currently empty**, and adding an entry to it requires updating this answer,
`README.md` and `SECURITY.md` in the same pull request.

Why the allowlist exists at all: a read-only fetcher for a public research
dataset (for example the KDD Cup 1998 donor file, used to validate the library
against real data) is a plausible future addition. It would be an opt-in function
you call deliberately, never automatic, and it would still transmit none of your
data. Rather than let such a function quietly falsify this page, the allowlist
makes it a reviewed, documented change. If this paragraph still says the
allowlist is empty, nothing in the package can reach the network at all.

This is enforced, not just documented: `tests/test_no_network.py` monkeypatches
every socket entry point to raise, then runs a full train/score cycle, an imputation
pass, and a CRM ingest. CI fails if any of them tries to open a socket.

## 1a. What if someone passes it a remote path?

They get a `ValueError` before any read happens.

`pandas` readers will happily follow `https://`, `s3://` and `gs://` URIs, so
every user-supplied path is checked first. `EncounterTransformer(encounter_path=...)`,
`GratefulPatientFeaturizer(encounter_path=...)` and the CLI's `--data` argument
all reject any non-local scheme with a `ValueError` naming the parameter. Local
paths, including `file://`, are unaffected.

This closes a real gap rather than a theoretical one: before it was added, the
guarantee above was true of the library's own logic and false of its documented
public parameters, because an operator could hand a documented argument a remote
URI and pandas would fetch it. The check lives in
`philanthropy.utils._validation.ensure_local_path`, and
`tests/test_no_network.py` asserts both halves: that remote schemes raise, and
that ordinary local paths still load.

## 2. Do we need a Business Associate Agreement (BAA)?

**No.** A BAA covers a business associate who creates, receives, maintains, or
transmits protected health information on a covered entity's behalf. PhilanthroPy
is a library you install and run inside your own environment. No data is
transmitted to the maintainer or to any third party, and the maintainer has no
access to your data, your infrastructure, or your logs. There is no service, no
hosted component, and no vendor to contract with.

You remain the sole custodian of the data throughout.

## 3. What is the licence, and can we modify it?

MIT. You may use, modify, and redistribute it, including inside proprietary or
internal systems, with attribution. There is no separate commercial licence, no
per-seat cost, and no field-of-use restriction.

## 4. What third-party code does it pull in?

Four runtime dependencies, all long-established and widely vetted in the scientific
Python ecosystem:

| Package | Floor | Why |
|---|---|---|
| `scikit-learn` | `>=1.6` | every estimator subclasses its base classes |
| `pandas` | `>=2.0` | the feature-table and CRM-ingest layer |
| `numpy` | `>=1.23` | array backend |
| `joblib` | `>=1.2` | model persistence |

`matplotlib` and `seaborn` are needed only for the plotting helpers and are
optional; CI includes a job asserting the package imports without `matplotlib`
installed. Supported Python is 3.9 and newer.

There are no deep-learning dependencies. That is a deliberate constraint, not an
omission: heavier methods are approximated with the stack above so the install
stays small and auditable.

## 5. Is the code reviewed and tested?

Every push runs, in CI: `flake8`, `mypy`, docstring examples, the full test suite,
an overall branch-coverage floor, a higher coverage floor over the risk-tier
subtree (`preprocessing/`, `models/`, `ingest/`, the CLI, and model persistence),
declared dependency floors on the oldest supported Python, a packaging check
(`python -m build` plus `twine check`), and CodeQL static analysis.

Public estimators are also exercised by `sklearn.utils.estimator_checks`, a
third-party conformance suite rather than tests written by the same author: 20
configured instances, 1016 checks. Four classes are row-reducing or require a
constructor argument, so they cannot run the generated battery bare
(`RFMTransformer`, `MatchingGiftFeaturizer`, `EncounterTransformer`,
`GratefulPatientFeaturizer`); each has hand-written equivalent coverage and a
recorded reason, and a test fails the build if any public estimator is in
neither list. `UpliftTLearner` (Tier 3, `philanthropy.experimental`) is outside
the `fit(X, y)` contract altogether; see `docs/reference/experimental.md`.

## 6. What is the biggest security caveat we should know about?

**Model bundles are pickle-based.** Loading a `.joblib` bundle executes arbitrary
code during unpickling, exactly as scikit-learn's own persisted estimators do.

Only load model bundles produced by you or by a source you trust. Never run
`philanthropy score --model` against a bundle received from an untrusted party.
This is a property of the Python serialisation format, not of this package, and it
applies equally to any scikit-learn model you load. See
[SECURITY.md](https://github.com/PhilanthroPy-Project/PhilanthroPy/blob/main/SECURITY.md).

## 7. Does it de-identify our data for us?

**No, and you should not rely on it as though it did.** The `PII_PATTERNS`
column-dropping lives on `EncounterTransformer` (and is inherited by
`GratefulPatientFeaturizer`), not on `CRMCleaner`, which does no dropping at all.
It is a name-based heuristic, defence in depth against
obvious identifier columns being fed into a model, not de-identification under HIPAA
Safe Harbor (45 CFR 164.514(b)) or Expert Determination (164.514(a)). Note also
that `pii_patterns` **replaces** the default tuple rather than extending it, so
passing your own patterns switches off the built-in `mrn`/`ssn` entries unless you
repeat them.

De-identification remains your determination to make. The reasoning, including the
164.514(f) fundraising carve-out and its minimum-necessary and opt-out obligations,
is in [Compliance considerations](compliance_considerations.md).

## 8. Does it make decisions about patients or donors?

It produces **scores, not decisions.** The scoring methods return a ranking signal
for a human to act on: a prioritised call list for a gift officer. Nothing in the
package takes an action, sends a solicitation, or writes back to your CRM.

The package also ships a fairness diagnostic (`disparate_impact_ratio`, a
four-fifths-rule check). Treat it as a diagnostic that surfaces a problem, **not**
as a clearance that certifies its absence.

## 9. Who maintains it, and what happens if they stop?

One maintainer today. That is stated plainly in the README rather than hidden,
because for an institution running an OSS risk review it is a material fact.

What that means concretely: the code is MIT-licensed and fully public, so a fork
costs you nothing but time; there are no proprietary components, hosted services,
or license servers that could be switched off; and the test suite plus CI
configuration are in the repository, so a successor can verify the software still
works before changing it. There is no data or credential held by the maintainer
that you would need to recover.

If continuity matters to your review, say so in an issue; a second maintainer with
merge rights is a known gap being worked on.

## 10. How do we report a vulnerability?

Privately, via [GitHub's private vulnerability reporting](https://github.com/PhilanthroPy-Project/PhilanthroPy/security/advisories/new),
which is visible only to you and the maintainer. Email is a fallback. Please do not
open a public issue for a security problem. Expect acknowledgement within a few days
and a coordinated fix and disclosure. Full policy:
[SECURITY.md](https://github.com/PhilanthroPy-Project/PhilanthroPy/blob/main/SECURITY.md).
