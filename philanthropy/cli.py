"""
philanthropy.cli
================
A CSV-in / CSV-out command line for analysts who aren't primarily Python
engineers. Three subcommands:

    philanthropy train    --data gifts.csv --target is_major_donor \\
                          --features total_gift_amount,years_active --out model.joblib
    philanthropy score    --model model.joblib --data prospects.csv --out scores.csv
    philanthropy validate --model model.joblib --data holdout.csv --target is_major_donor

`train` saves a self-describing bundle (the fitted model, the feature list, and
the scikit-learn / philanthropy versions used); `score` and `validate` reuse the
feature list stored in that bundle unless you override it with `--features`.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import __version__

_MODEL_CHOICES = (
    "DonorPropensityModel",
    "MajorGiftClassifier",
    "LapsePredictor",
    "PlannedGivingIntentScorer",
)


def _resolve_model(name):
    # argparse already rejects anything outside _MODEL_CHOICES, so there is no
    # unknown-name branch to handle here.
    from . import models

    return getattr(models, name)


def _read_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise SystemExit(f"Data file not found: {path}")


def _require_columns(df, columns, path):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Column(s) {missing} not found in {path}. "
            f"Available: {list(df.columns)}."
        )


def _split_features(features):
    if not features:
        return None
    return [f.strip() for f in features.split(",") if f.strip()]


def _load_bundle(path):
    from .utils import load_model

    try:
        return load_model(path)
    except FileNotFoundError:
        raise SystemExit(f"Model file not found: {path}")
    except ValueError as exc:
        raise SystemExit(str(exc))


def _score_array(model, X):
    if hasattr(model, "predict_affinity_score"):
        return model.predict_affinity_score(X)
    return model.predict_proba(X)[:, 1]


_CSV_INJECTION_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _neutralise_csv_injection(df):
    """Prefix an apostrophe to object-dtype cells beginning with a spreadsheet
    formula trigger (=, +, -, @, tab, CR).

    The scored CSV echoes donor-controlled string fields (name, email, …) that
    originate from third-party advancement webhooks. Without this, a cell like
    ``=cmd|'/c calc'!A1`` executes when an analyst opens scores.csv in Excel or
    Google Sheets (CSV formula injection, CWE-1236). Returns a copy; numeric
    columns are untouched.
    """
    out = df.copy()
    # Iterate every column and let the isinstance guard below decide. Selecting
    # by dtype is not stable across pandas versions: pandas 4 stops returning
    # `str`-dtype columns for include=["object"], which would silently switch
    # neutralisation off for exactly the donor-supplied text this protects.
    for col in out.columns:
        mask = out[col].map(
            lambda v: isinstance(v, str) and v.startswith(_CSV_INJECTION_PREFIXES)
        )
        if mask.any():
            out.loc[mask, col] = "'" + out.loc[mask, col]
    return out


def _cmd_train(args):
    features = _split_features(args.features)
    # Falsy, not `is None`: "--features ' , '" parses to [] and used to reach
    # fit() with a zero-column matrix.
    if not features:
        raise SystemExit("train requires --features (comma-separated column names).")
    df = _read_csv(args.data)
    _require_columns(df, features + [args.target], args.data)

    from .utils import save_model

    model = _resolve_model(args.model)(random_state=args.random_state)
    model.fit(df[features].to_numpy(), df[args.target].to_numpy())

    save_model(model, args.out, features=features, target=args.target)
    print(f"Trained {args.model} on {len(df)} rows; saved to {args.out}")


def _cmd_score(args):
    bundle = _load_bundle(args.model)
    features = _split_features(args.features) or bundle["features"]
    df = _read_csv(args.data)
    _require_columns(df, features, args.data)

    out = df.copy()
    out["score"] = _score_array(bundle["model"], df[features].to_numpy())
    out = _neutralise_csv_injection(out)
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"Wrote {len(out)} scored rows to {args.out}")
    else:
        out.to_csv(sys.stdout, index=False)


def _cmd_validate(args):
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    bundle = _load_bundle(args.model)
    features = _split_features(args.features) or bundle["features"]
    target = args.target or bundle.get("target")
    if target is None:
        raise SystemExit(
            "validate requires --target (no target stored in the model bundle)."
        )
    df = _read_csv(args.data)
    _require_columns(df, list(features) + [target], args.data)

    model = bundle["model"]
    X = df[features].to_numpy()
    y = df[target].to_numpy()
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    print(f"precision {precision_score(y, y_pred, zero_division=0):.3f}")
    print(f"recall    {recall_score(y, y_pred, zero_division=0):.3f}")
    print(f"f1        {f1_score(y, y_pred, zero_division=0):.3f}")
    print(f"roc_auc   {roc_auc_score(y, y_proba):.3f}")


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="philanthropy",
        description="Train, score, and validate PhilanthroPy models from CSV files.",
    )
    parser.add_argument(
        "--version", action="version", version=f"philanthropy {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a model from a labelled CSV and save it.")
    train.add_argument("--data", required=True, help="path to a labelled CSV")
    train.add_argument("--target", required=True, help="name of the label column")
    train.add_argument("--features", required=True, help="comma-separated feature columns")
    train.add_argument("--model", default="DonorPropensityModel", choices=_MODEL_CHOICES)
    train.add_argument("--out", required=True, help="output model bundle path (.joblib)")
    train.add_argument("--random-state", type=int, default=0, dest="random_state")
    train.set_defaults(func=_cmd_train)

    score = sub.add_parser("score", help="Score a CSV with a saved model.")
    score.add_argument(
        "--model",
        required=True,
        help="saved model bundle (.joblib). WARNING: a bundle is unpickled on "
        "load and can execute arbitrary code — only load bundles you trust.",
    )
    score.add_argument("--data", required=True, help="path to a CSV to score")
    score.add_argument("--features", default=None, help="override the bundle's features")
    score.add_argument("--out", default=None, help="output CSV path (default: stdout)")
    score.set_defaults(func=_cmd_score)

    validate = sub.add_parser(
        "validate", help="Report precision/recall/F1/ROC-AUC on a labelled CSV."
    )
    validate.add_argument(
        "--model",
        required=True,
        help="saved model bundle (.joblib). WARNING: a bundle is unpickled on "
        "load and can execute arbitrary code — only load bundles you trust.",
    )
    validate.add_argument("--data", required=True, help="path to a labelled CSV")
    validate.add_argument("--target", default=None, help="label column (else bundle's)")
    validate.add_argument("--features", default=None, help="override the bundle's features")
    validate.set_defaults(func=_cmd_validate)

    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
