# Security Policy

## Supported versions

Security fixes land on the latest `0.5.x` release.

| Version | Supported          |
| ------- | ------------------ |
| 0.5.x   | :white_check_mark: |
| < 0.5   | :x:                |

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

Email **shivamlalakiya151299@gmail.com** with a description and, if possible, a
minimal reproduction. You can expect an acknowledgement within a few days and a
coordinated fix and disclosure.
