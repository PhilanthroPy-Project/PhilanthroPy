"""tests/test_cli.py: end-to-end CLI (train -> score -> validate)."""

import pandas as pd
import pytest

from philanthropy.cli import main
from philanthropy.datasets import generate_synthetic_donor_data

FEATURES = "total_gift_amount,years_active,event_attendance_count"


def _make_csv(tmp_path, name, n=300):
    df = generate_synthetic_donor_data(n_samples=n, random_state=1)
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def test_cli_train_score_validate(tmp_path, capsys):
    data = _make_csv(tmp_path, "train.csv")
    model_path = tmp_path / "m.joblib"

    main(["train", "--data", str(data), "--target", "is_major_donor",
          "--features", FEATURES, "--out", str(model_path)])
    assert model_path.exists()

    scores_path = tmp_path / "scores.csv"
    main(["score", "--model", str(model_path), "--data", str(data),
          "--out", str(scores_path)])
    scored = pd.read_csv(scores_path)
    assert "score" in scored.columns
    assert len(scored) == 300

    main(["validate", "--model", str(model_path), "--data", str(data),
          "--target", "is_major_donor"])
    out = capsys.readouterr().out
    assert "roc_auc" in out


def test_cli_uses_bundle_features_and_target(tmp_path, capsys):
    # score/validate should work without re-specifying --features/--target
    data = _make_csv(tmp_path, "d.csv")
    model_path = tmp_path / "m.joblib"
    main(["train", "--data", str(data), "--target", "is_major_donor",
          "--features", FEATURES, "--out", str(model_path)])

    main(["validate", "--model", str(model_path), "--data", str(data)])
    assert "precision" in capsys.readouterr().out


def test_cli_missing_target_column_errors(tmp_path):
    data = _make_csv(tmp_path, "d.csv")
    model_path = tmp_path / "m.joblib"
    with pytest.raises(SystemExit):
        main(["train", "--data", str(data), "--target", "does_not_exist",
              "--features", FEATURES, "--out", str(model_path)])


def test_cli_missing_model_file_errors(tmp_path):
    data = _make_csv(tmp_path, "d.csv")
    with pytest.raises(SystemExit):
        main(["score", "--model", str(tmp_path / "nope.joblib"),
              "--data", str(data), "--features", FEATURES])


def test_cli_score_writes_to_stdout_by_default(tmp_path, capsys):
    data = _make_csv(tmp_path, "d.csv", n=20)
    model_path = tmp_path / "m.joblib"
    main(["train", "--data", str(data), "--target", "is_major_donor",
          "--features", FEATURES, "--out", str(model_path)])
    capsys.readouterr()

    main(["score", "--model", str(model_path), "--data", str(data)])
    out = capsys.readouterr().out
    assert out.splitlines()[0].endswith(",score")
    assert len(out.strip().splitlines()) == 21  # header + 20 rows


@pytest.mark.parametrize("model_name", [
    "DonorPropensityModel",
    "MajorGiftClassifier",
    "LapsePredictor",
    "PlannedGivingIntentScorer",
])
def test_cli_train_accepts_every_documented_model(tmp_path, capsys, model_name):
    data = _make_csv(tmp_path, "d.csv", n=120)
    model_path = tmp_path / f"{model_name}.joblib"
    main(["train", "--data", str(data), "--target", "is_major_donor",
          "--features", FEATURES, "--model", model_name, "--out", str(model_path)])
    assert model_path.exists()
    assert f"Trained {model_name}" in capsys.readouterr().out

    from philanthropy.utils import load_model
    assert type(load_model(model_path)["model"]).__name__ == model_name


def test_cli_train_without_features_exits(tmp_path):
    data = _make_csv(tmp_path, "d.csv", n=20)
    # argparse enforces --features on the command line, so reach _cmd_train with
    # an empty string, which _split_features resolves to None.
    with pytest.raises(SystemExit, match="train requires --features"):
        main(["train", "--data", str(data), "--target", "is_major_donor",
              "--features", "  ,  ", "--out", str(tmp_path / "m.joblib")])


def test_cli_missing_data_file_exits(tmp_path):
    with pytest.raises(SystemExit, match="Data file not found"):
        main(["train", "--data", str(tmp_path / "absent.csv"),
              "--target", "is_major_donor", "--features", FEATURES,
              "--out", str(tmp_path / "m.joblib")])


def test_cli_non_bundle_model_file_exits(tmp_path):
    import joblib

    data = _make_csv(tmp_path, "d.csv", n=20)
    junk = tmp_path / "junk.joblib"
    joblib.dump({"not": "a bundle"}, junk)
    with pytest.raises(SystemExit, match="not a PhilanthroPy model bundle"):
        main(["score", "--model", str(junk), "--data", str(data),
              "--features", FEATURES])


def test_cli_validate_without_a_target_anywhere_exits(tmp_path):
    from philanthropy.models import DonorPropensityModel
    from philanthropy.utils import save_model

    data = _make_csv(tmp_path, "d.csv", n=60)
    df = pd.read_csv(data)
    features = FEATURES.split(",")
    model = DonorPropensityModel(n_estimators=5, random_state=0).fit(
        df[features].to_numpy(), df["is_major_donor"].to_numpy()
    )
    bundle_path = tmp_path / "no_target.joblib"
    save_model(model, bundle_path, features=features)  # target left as None

    with pytest.raises(SystemExit, match="validate requires --target"):
        main(["validate", "--model", str(bundle_path), "--data", str(data)])


def test_cli_missing_feature_column_lists_available(tmp_path):
    data = _make_csv(tmp_path, "d.csv", n=20)
    with pytest.raises(SystemExit, match=r"not found in .*Available:"):
        main(["train", "--data", str(data), "--target", "is_major_donor",
              "--features", "nope_col", "--out", str(tmp_path / "m.joblib")])


# --------------------------------------------------------------------------- #
# `features`: gift export in, donor-level feature table out
# --------------------------------------------------------------------------- #
_RE_HEADER = "Constituent ID,Gift Date,Gift Amount,Gift Type,Fund,Email\n"


def _make_gift_export(tmp_path, name="gifts.csv", n_donors=60):
    rows = [_RE_HEADER]
    for i in range(n_donors):
        rows.append(f"{i},2025-01-10,{1000 + i * 10}.00,Pledge,Annual Fund,d{i}@amc.edu\n")
        rows.append(f"{i},2025-02-10,{100 + i}.00,Pay-Cash,Annual Fund,d{i}@amc.edu\n")
        rows.append(f"{i},2025-03-10,{50 + i}.00,Pay-Cash,Annual Fund,d{i}@amc.edu\n")
    path = tmp_path / name
    path.write_text("".join(rows))
    return path


def test_cli_features_raisers_edge_drops_the_pledge_rows(tmp_path, capsys):
    data = _make_gift_export(tmp_path, n_donors=3)
    out_path = tmp_path / "features.csv"
    main(["features", "--source", "raisers_edge", "--data", str(data),
          "--out", str(out_path)])
    feats = pd.read_csv(out_path)
    assert len(feats) == 3
    # Donor 0: 100 + 50 in payments, and not the 1000 pledge on top.
    row = feats.loc[feats["contact_id"] == 0].iloc[0]
    assert row["total_gift_amount"] == 150.0
    assert row["gift_count"] == 2
    assert "Wrote 3 donor rows" in capsys.readouterr().out


def test_cli_features_writes_contact_id_as_a_column(tmp_path):
    data = _make_gift_export(tmp_path, n_donors=2)
    out_path = tmp_path / "features.csv"
    main(["features", "--source", "raisers_edge", "--data", str(data),
          "--out", str(out_path)])
    assert list(pd.read_csv(out_path).columns)[0] == "contact_id"


def test_cli_features_writes_to_stdout_by_default(tmp_path, capsys):
    data = _make_gift_export(tmp_path, n_donors=2)
    main(["features", "--source", "raisers_edge", "--data", str(data)])
    out = capsys.readouterr().out
    assert out.startswith("contact_id,")
    assert len(out.strip().splitlines()) == 3


def test_cli_features_civicrm_source(tmp_path):
    path = tmp_path / "contributions.csv"
    path.write_text(
        "Contact ID,Contribution Date,Total Amount,Contribution Status\n"
        "101,2025-01-15,250.00,Completed\n"
        "101,2025-02-15,99.00,Failed\n"
    )
    out_path = tmp_path / "features.csv"
    main(["features", "--source", "civicrm", "--data", str(path),
          "--out", str(out_path)])
    feats = pd.read_csv(out_path)
    assert feats.iloc[0]["total_gift_amount"] == 250.0


def test_cli_features_neutralises_csv_injection(tmp_path):
    path = tmp_path / "gifts.csv"
    path.write_text(_RE_HEADER + "1,2025-01-10,100.00,Cash,Annual Fund,=cmd|calc\n")
    out_path = tmp_path / "features.csv"
    main(["features", "--source", "raisers_edge", "--data", str(path),
          "--out", str(out_path)])
    assert "'=cmd|calc" in out_path.read_text()


def test_cli_features_missing_export_field_exits_with_the_field_names(tmp_path, capsys):
    path = tmp_path / "gifts.csv"
    path.write_text("Constituent ID,Gift Type\n1,Cash\n")
    with pytest.raises(SystemExit) as excinfo:
        main(["features", "--source", "raisers_edge", "--data", str(path)])
    assert "Gift Date" in str(excinfo.value)


def test_cli_features_missing_data_file_exits(tmp_path):
    with pytest.raises(SystemExit):
        main(["features", "--source", "raisers_edge",
              "--data", str(tmp_path / "nope.csv")])


def test_cli_features_help_lists_the_columns_it_actually_emits(tmp_path):
    """The help text is what tells an analyst what to pass to `train
    --features`, so it must not drift from the real frame."""
    from philanthropy.cli import _FEATURE_COLUMNS

    data = _make_gift_export(tmp_path, n_donors=2)
    out_path = tmp_path / "features.csv"
    main(["features", "--source", "raisers_edge", "--data", str(data),
          "--out", str(out_path)])
    emitted = list(pd.read_csv(out_path).columns)
    assert [c.strip() for c in _FEATURE_COLUMNS.split(",")] == emitted


def test_cli_features_then_train_then_score(tmp_path, capsys):
    """The path the CLI advertises, end to end: a raw Raiser's Edge gift export
    in, a scored CSV out, no Python. The label is still the analyst's: `train`
    needs a --target column that no gift export contains."""
    data = _make_gift_export(tmp_path, n_donors=80)
    feature_path = tmp_path / "features.csv"
    main(["features", "--source", "raisers_edge", "--data", str(data),
          "--out", str(feature_path)])

    labelled = pd.read_csv(feature_path)
    labelled["is_major_donor"] = (labelled["total_gift_amount"] > 200).astype(int)
    labelled_path = tmp_path / "labelled.csv"
    labelled.to_csv(labelled_path, index=False)

    model_path = tmp_path / "m.joblib"
    main(["train", "--data", str(labelled_path), "--target", "is_major_donor",
          "--features", "total_gift_amount,gift_count,recency_days",
          "--out", str(model_path)])

    scores_path = tmp_path / "scores.csv"
    main(["score", "--model", str(model_path), "--data", str(feature_path),
          "--out", str(scores_path)])
    scored = pd.read_csv(scores_path)
    assert "score" in scored.columns
    assert len(scored) == 80
