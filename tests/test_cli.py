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
