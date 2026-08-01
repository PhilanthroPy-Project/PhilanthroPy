"""tests/test_cli.py — end-to-end CLI (train -> score -> validate)."""

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
