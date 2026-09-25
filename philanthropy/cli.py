"""
philanthropy.cli
================
A CSV-in / CSV-out command line for analysts who aren't primarily Python
engineers. Four subcommands:

    philanthropy features --source raisers_edge --data gifts.csv --out features.csv
    philanthropy train    --data features.csv --target is_major_donor \\
                          --features total_gift_amount,years_active --out model.joblib
    philanthropy score    --model model.joblib --data prospects.csv --out scores.csv
    philanthropy validate --model model.joblib --data holdout.csv --target is_major_donor

`features` rolls a raw CRM gift export up into the donor-level table the models
consume, so the path from an export to a scored CSV needs no Python. It does not
invent a label: `train` still needs a `--target` column, and constructing one
from your own definition of a major donor stays your job. `features` also
accepts repeated `--activity TYPE=PATH` flags to fold engagement data in
alongside the gift features (see `activities_to_features`).

`train` saves a self-describing bundle (the fitted model, the feature list, and
the scikit-learn / philanthropy versions used); `score` and `validate` reuse the
feature list stored in that bundle unless you override it with `--features`.
`train --task upgrade` is a different shape entirely: it reads a raw gift
export (not a pre-built features CSV) and calls
`philanthropy.models.score_upgrade_prospects` directly, which trains on
history and scores today's prospects in one call, so `--out` there is a
scored CSV, not a saved model bundle.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import __version__

# Gift-export readers `features` can front. Each name maps to a
# (reader, aggregator) pair in _cmd_features.
_FEATURE_SOURCES = ("civicrm", "raisers_edge", "npsp")

# The donor-level columns `features` emits, in order. Named here so
# `philanthropy features --help` answers "what do I pass to --features?"
# without a guess; tests/test_cli.py checks it against the real frame.
_FEATURE_COLUMNS = (
    "contact_id, constituent_email, first_name, last_name, total_gift_amount, "
    "gift_count, largest_gift_amount, first_gift_date, last_gift_date, "
    "years_active, recency_days, distinct_financial_types"
)

_MODEL_CHOICES = (
    "DonorPropensityModel",
    "MajorGiftClassifier",
    "LapsePredictor",
    "PlannedGivingIntentScorer",
)


def _resolve_model(name: str) -> type:
    # argparse already rejects anything outside _MODEL_CHOICES, so there is no
    # unknown-name branch to handle here.
    from . import models

    return getattr(models, name)


def _read_csv(path: str) -> pd.DataFrame:
    from .utils._validation import ensure_local_path

    ensure_local_path(path, "data")
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise SystemExit(f"Data file not found: {path}")


def _require_columns(
    df: pd.DataFrame, columns: Sequence[str], path: str
) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Column(s) {missing} not found in {path}. "
            f"Available: {list(df.columns)}."
        )


def _split_features(features: Optional[str]) -> Optional[List[str]]:
    if not features:
        return None
    return [f.strip() for f in features.split(",") if f.strip()]


def _read_activities(activity_specs: Sequence[str]) -> pd.DataFrame:
    """Read every ``--activity TYPE=PATH`` flag into one long activity table.

    Each file is tagged with its own ``TYPE`` (overwriting any
    ``activity_type`` column it already has) before the files are
    concatenated, so a donor's engagement across sources lands in one table
    keyed like ``activities_to_features`` expects.
    """
    from .utils._validation import ensure_local_path

    frames = []
    for spec in activity_specs:
        if "=" not in spec:
            raise SystemExit(f"--activity must be TYPE=PATH, got {spec!r}.")
        activity_type, path = spec.split("=", 1)
        ensure_local_path(path, "activity")
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            raise SystemExit(f"Data file not found: {path}")
        df["activity_type"] = activity_type
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _read_raw_gifts(source: str, path: str) -> pd.DataFrame:
    """Read a raw CRM gift export and normalise it to ``donor_id`` /
    ``gift_date`` / ``gift_amount``, the shape ``score_upgrade_prospects``
    (via ``build_upgrade_snapshots``) needs.

    Reuses each source's own header-canonicalisation function (the same one
    its ``*_to_features`` aggregator calls internally) rather than
    re-deriving the CRM's column aliases here; renaming its three canonical
    columns to the gift-log names is the only new logic.
    """
    from . import ingest
    from .ingest._civicrm import _normalise_headers
    from .utils._validation import ensure_local_path

    ensure_local_path(path, "data")
    try:
        if source == "raisers_edge":
            from .ingest._raisers_edge import _canonical_raisers_edge

            raw = _normalise_headers(ingest.read_raisers_edge_gifts(path), _canonical_raisers_edge)
        elif source == "npsp":
            from .ingest._npsp import _canonical_npsp

            raw = _normalise_headers(ingest.read_npsp_opportunities(path), _canonical_npsp)
        else:
            raw = ingest.read_civicrm_contributions(path)
    except FileNotFoundError:
        raise SystemExit(f"Data file not found: {path}")

    missing = [c for c in ("contact_id", "receive_date", "total_amount") if c not in raw.columns]
    if missing:
        raise SystemExit(
            f"Column(s) {missing} not found in {path} after normalisation. "
            f"Available: {list(raw.columns)}."
        )
    return raw.rename(
        columns={"contact_id": "donor_id", "receive_date": "gift_date", "total_amount": "gift_amount"}
    )


def _load_bundle(path: str) -> Dict[str, Any]:
    from .utils import load_model

    try:
        return load_model(path)
    except FileNotFoundError:
        raise SystemExit(f"Model file not found: {path}")
    except ValueError as exc:
        raise SystemExit(str(exc))


def _score_array(model: Any, X: Any) -> np.ndarray:
    if hasattr(model, "predict_affinity_score"):
        return model.predict_affinity_score(X)
    return model.predict_proba(X)[:, 1]


_CSV_INJECTION_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _neutralise_csv_injection(df: pd.DataFrame) -> pd.DataFrame:
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


def _cmd_train(args: argparse.Namespace) -> None:
    if args.task == "upgrade":
        _cmd_train_upgrade(args)
        return

    features = _split_features(args.features)
    # Falsy, not `is None`: "--features ' , '" parses to [] and used to reach
    # fit() with a zero-column matrix.
    if not features:
        raise SystemExit("train requires --features (comma-separated column names).")
    if not args.target:
        raise SystemExit("train requires --target.")
    df = _read_csv(args.data)
    _require_columns(df, features + [args.target], args.data)

    from .utils import save_model

    model = _resolve_model(args.model)(random_state=args.random_state)
    model.fit(df[features].to_numpy(), df[args.target].to_numpy())

    save_model(model, args.out, features=features, target=args.target)
    print(f"Trained {args.model} on {len(df)} rows; saved to {args.out}")


def _cmd_train_upgrade(args: argparse.Namespace) -> None:
    if not args.source:
        raise SystemExit("train --task upgrade requires --source.")

    from .models import score_upgrade_prospects

    gifts = _read_raw_gifts(args.source, args.data)
    activities = _read_activities(args.activity) if args.activity else None

    donors = None
    if args.donors:
        donors = _read_csv(args.donors)
        if "donor_id" not in donors.columns:
            raise SystemExit(
                f"--donors file must have a 'donor_id' column; got {list(donors.columns)}."
            )
        donors = donors.set_index("donor_id")

    scores, report = score_upgrade_prospects(
        gifts,
        activities=activities,
        donors=donors,
        threshold=args.threshold,
        band=tuple(args.band),
        fiscal_year_start=args.fiscal_year_start,
        as_of=args.as_of,
        random_state=args.random_state,
    )

    out = _neutralise_csv_injection(scores.reset_index())
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"Wrote {len(out)} scored rows to {args.out}")
    else:
        out.to_csv(sys.stdout, index=False)

    for key, value in report.items():
        print(f"{key}: {value}")


def _cmd_score(args: argparse.Namespace) -> None:
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


def _cmd_validate(args: argparse.Namespace) -> None:
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


def _cmd_features(args: argparse.Namespace) -> None:
    from . import ingest
    from .utils._validation import ensure_local_path

    read: Callable[..., pd.DataFrame]
    to_features: Callable[..., pd.DataFrame]
    if args.source == "raisers_edge":
        read = ingest.read_raisers_edge_gifts
        to_features = ingest.raisers_edge_gifts_to_features
    elif args.source == "npsp":
        read = ingest.read_npsp_opportunities
        to_features = ingest.npsp_opportunities_to_features
    else:
        read = ingest.read_civicrm_contributions
        to_features = ingest.civicrm_contributions_to_features

    ensure_local_path(args.data, "data")
    try:
        gifts = read(args.data)
    except FileNotFoundError:
        raise SystemExit(f"Data file not found: {args.data}")
    try:
        features = to_features(gifts)
    except KeyError as exc:
        # The aggregators name the missing export fields; that message is the
        # whole answer, so surface it instead of a traceback.
        raise SystemExit(str(exc.args[0] if exc.args else exc))

    if args.activity:
        activities = _read_activities(args.activity)
        as_of = args.as_of or pd.to_datetime(activities["activity_date"], errors="coerce").max()
        act_feats = ingest.activities_to_features(activities, as_of=as_of, donors=features)
        # Both sides already key on contact_id: a plain left join, no rename.
        features = features.join(act_feats, how="left")

    # reset_index first: contact_id is the index, and _neutralise_csv_injection
    # only walks columns, so an index left in place would skip the escaping and
    # then be written to the CSV anyway by to_csv(index=True).
    out = _neutralise_csv_injection(features.reset_index())
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"Wrote {len(out)} donor rows to {args.out}")
    else:
        out.to_csv(sys.stdout, index=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="philanthropy",
        description="Train, score, and validate PhilanthroPy models from CSV files.",
    )
    parser.add_argument(
        "--version", action="version", version=f"philanthropy {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    features = sub.add_parser(
        "features",
        help="Roll a CRM gift export up into the donor-level feature table.",
        description=(
            "Aggregate a raw CRM gift export into the one-row-per-donor table "
            "the models consume, then feed that to `train` and `score`. "
            "Emitted columns, in order: " + _FEATURE_COLUMNS + ". Commitment "
            "rows (pledges, recurring gift templates) are dropped for "
            "raisers_edge, Pledged instalment rows for npsp, and test-mode "
            "and non-Completed rows for civicrm, so a committed dollar is "
            "not counted twice. No label is produced: `train --target` "
            "needs a column you define yourself."
        ),
    )
    features.add_argument(
        "--source", required=True, choices=_FEATURE_SOURCES,
        help="which CRM the export came from",
    )
    features.add_argument(
        "--data", required=True, help="gift export CSV, or a directory of them"
    )
    features.add_argument(
        "--activity", action="append", default=[], metavar="TYPE=PATH",
        help="an activity log CSV tagged with its type, e.g. event=events.csv; "
        "repeatable. Adds activities_to_features columns to the output.",
    )
    features.add_argument(
        "--as-of", default=None, dest="as_of",
        help="cutoff date for --activity features (default: the latest "
        "activity_date in the combined activity log)",
    )
    features.add_argument("--out", default=None, help="output CSV path (default: stdout)")
    features.set_defaults(func=_cmd_features)

    train = sub.add_parser("train", help="Train a model from a labelled CSV and save it.")
    train.add_argument("--task", choices=("plain", "upgrade"), default="plain",
                        help="'plain': fit --model on --features/--target (default). "
                        "'upgrade': read a raw gift export via --source and call "
                        "score_upgrade_prospects, writing scored donors to --out.")
    train.add_argument("--data", required=True, help="labelled CSV (plain) or raw gift export (upgrade)")
    train.add_argument("--target", default=None, help="name of the label column (plain)")
    train.add_argument("--features", default=None, help="comma-separated feature columns (plain)")
    train.add_argument("--model", default="DonorPropensityModel", choices=_MODEL_CHOICES)
    train.add_argument("--out", required=True, help="output model bundle (plain) or scored CSV (upgrade)")
    train.add_argument("--random-state", type=int, default=0, dest="random_state")
    train.add_argument("--source", default=None, choices=_FEATURE_SOURCES,
                        help="which CRM the gift export came from (upgrade)")
    train.add_argument("--threshold", type=float, default=1000.0,
                        help="leadership-giving level an upgrade crosses into (upgrade)")
    train.add_argument("--band", type=float, nargs=2, default=(100.0, 999.0),
                        metavar=("LOW", "HIGH"),
                        help="upgrade-candidate FY-total band (upgrade)")
    train.add_argument("--fiscal-year-start", type=int, default=7, dest="fiscal_year_start",
                        help="month (1-12) the fiscal year begins (upgrade)")
    train.add_argument("--activity", action="append", default=[], metavar="TYPE=PATH",
                        help="an activity log CSV tagged with its type; repeatable (upgrade)")
    train.add_argument("--donors", default=None,
                        help="optional donor-attributes CSV with a donor_id column (upgrade)")
    train.add_argument("--as-of", default=None, dest="as_of",
                        help="scoring cutoff (default: the latest gift date) (upgrade)")
    train.set_defaults(func=_cmd_train)

    score = sub.add_parser("score", help="Score a CSV with a saved model.")
    score.add_argument(
        "--model",
        required=True,
        help="saved model bundle (.joblib). WARNING: a bundle is unpickled on "
        "load and can execute arbitrary code; only load bundles you trust.",
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
        "load and can execute arbitrary code; only load bundles you trust.",
    )
    validate.add_argument("--data", required=True, help="path to a labelled CSV")
    validate.add_argument("--target", default=None, help="label column (else bundle's)")
    validate.add_argument("--features", default=None, help="override the bundle's features")
    validate.set_defaults(func=_cmd_validate)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Execute the philanthropy command-line interface.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments passed to the parser. If None, defaults to sys.argv[1:].
    """
    args = _build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
