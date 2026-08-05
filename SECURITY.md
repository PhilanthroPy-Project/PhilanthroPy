# Security Policy

## Supported versions

Security fixes land on the latest released minor. This table is updated as part of
the release checklist in [RELEASING.md](RELEASING.md) — if it names a version you
cannot `pip install`, that is a bug, please report it.

| Version | Supported          |
| ------- | ------------------ |
| 0.6.x   | :white_check_mark: |
| < 0.6   | :x:                |

## Loading models (trust boundary)

PhilanthroPy model bundles (`.joblib`) are **pickle-based**. Loading one —
`philanthropy score`/`validate --model X`, or `joblib.load` / the
`philanthropy.utils.load_model` helper — executes arbitrary code during
unpickling, exactly like scikit-learn's own persisted estimators.

**Only load model bundles from a source you trust.** Never run
`philanthropy score --model` on a `.joblib` file received from an untrusted
party. See scikit-learn's
[persistence security note](https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations).

## Reporting a vulnerability

Please **do not open a public issue** for security problems.

Use GitHub's private vulnerability reporting — [open a draft advisory](https://github.com/PhilanthroPy-Project/PhilanthroPy/security/advisories/new).
It is private to you and the maintainer, and it is the preferred channel.

If that is unavailable to you, email **shivamlalakiya151299@gmail.com** instead.

Either way, include a description and, if possible, a minimal reproduction. You can
expect an acknowledgement within a few days and a coordinated fix and disclosure.

## What this package does not do

PhilanthroPy makes **no network calls** — no telemetry, no license check, no
phone-home, and no third-party data append. It imports no HTTP client, and
`tests/test_no_network.py` enforces this in CI by making every socket raise.
Donor data you pass to it stays on the machine that ran it.

For the full set of questions an institutional security or privacy review asks,
see [security review Q&A](docs/explanation/security_review_answers.md).
