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


def test_cli_validate_reports_ranking_metrics(tmp_path, capsys):
    data = _make_csv(tmp_path, "d.csv")
    model_path = tmp_path / "m.joblib"
    main(["train", "--data", str(data), "--target", "is_major_donor",
          "--features", FEATURES, "--out", str(model_path)])

    main(["validate", "--model", str(model_path), "--data", str(data)])
    out = capsys.readouterr().out
    assert "at threshold 0.5" in out
    assert "average_precision" in out
    assert "base_rate" in out
    # one decile line per decile, numbered 1 through 10
    for decile in range(1, 11):
        assert f"\n{decile:>6}  " in out
    assert "top 30 of 300" in out  # default --top-n is 10% of rows


def test_cli_validate_top_n_accepts_count_and_percentage(tmp_path, capsys):
    data = _make_csv(tmp_path, "d.csv")
    model_path = tmp_path / "m.joblib"
    main(["train", "--data", str(data), "--target", "is_major_donor",
          "--features", FEATURES, "--out", str(model_path)])

    main(["validate", "--model", str(model_path), "--data", str(data), "--top-n", "25"])
    assert "top 25 of 300" in capsys.readouterr().out

    main(["validate", "--model", str(model_path), "--data", str(data), "--top-n", "20%"])
    assert "top 60 of 300" in capsys.readouterr().out


def test_cli_train_score_validate_do_not_warn_on_feature_names(tmp_path, recwarn):
    data = _make_csv(tmp_path, "d.csv")
    model_path = tmp_path / "m.joblib"
    main(["train", "--data", str(data), "--target", "is_major_donor",
          "--features", FEATURES, "--out", str(model_path)])
    main(["score", "--model", str(model_path), "--data", str(data),
          "--out", str(tmp_path / "scores.csv")])
    main(["validate", "--model", str(model_path), "--data", str(data)])
    assert not any("feature names" in str(w.message) for w in recwarn.list)


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


_NPSP_HEADER = "Account ID,Close Date,Amount,Stage\n"


def _make_opportunity_export(tmp_path, name="opportunities.csv", n_donors=60):
    rows = [_NPSP_HEADER]
    for i in range(n_donors):
        rows.append(f"{i},2025-01-10,{100 + i}.00,Pledged\n")
        rows.append(f"{i},2025-01-10,{100 + i}.00,Closed Won\n")
        rows.append(f"{i},2025-02-10,{50 + i}.00,Closed Won\n")
    path = tmp_path / name
    path.write_text("".join(rows))
    return path


def test_cli_features_npsp_drops_the_pledged_rows(tmp_path, capsys):
    data = _make_opportunity_export(tmp_path, n_donors=3)
    out_path = tmp_path / "features.csv"
    main(["features", "--source", "npsp", "--data", str(data),
          "--out", str(out_path)])
    feats = pd.read_csv(out_path)
    assert len(feats) == 3
    # Donor 0: the two Closed Won rows (100 + 50), not the duplicate 100 pledge.
    row = feats.loc[feats["contact_id"] == 0].iloc[0]
    assert row["total_gift_amount"] == 150.0
    assert row["gift_count"] == 2
    assert "Wrote 3 donor rows" in capsys.readouterr().out


def test_cli_features_npsp_missing_export_field_exits_with_the_field_names(tmp_path):
    path = tmp_path / "opportunities.csv"
    path.write_text("Account ID,Stage\n1,Closed Won\n")
    with pytest.raises(SystemExit) as excinfo:
        main(["features", "--source", "npsp", "--data", str(path)])
    assert "Close Date" in str(excinfo.value)


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


# --------------------------------------------------------------------------- #
# `features --activity` / `--as-of`
# --------------------------------------------------------------------------- #
def test_cli_features_activity_flag_adds_engagement_columns(tmp_path):
    data = _make_gift_export(tmp_path, n_donors=3)
    activity_path = tmp_path / "activities.csv"
    activity_path.write_text("contact_id,activity_date\n0,2025-01-01\n1,2025-02-01\n")
    out_path = tmp_path / "features.csv"
    main(["features", "--source", "raisers_edge", "--data", str(data),
          "--activity", f"event={activity_path}", "--out", str(out_path)])
    feats = pd.read_csv(out_path)
    assert "event_count_12m" in feats.columns
    row = feats.loc[feats["contact_id"] == 0].iloc[0]
    assert row["event_count_12m"] == 1


def test_cli_features_activity_as_of_cutoff(tmp_path):
    data = _make_gift_export(tmp_path, n_donors=2)
    activity_path = tmp_path / "activities.csv"
    activity_path.write_text("contact_id,activity_date\n0,2025-06-01\n")
    out_path = tmp_path / "features.csv"
    main(["features", "--source", "raisers_edge", "--data", str(data),
          "--activity", f"event={activity_path}", "--as-of", "2025-01-01",
          "--out", str(out_path)])
    feats = pd.read_csv(out_path)
    # The activity is after --as-of: cut before anything is counted, so the
    # event type has no rows left anywhere and contributes no columns.
    assert "event_count_12m" not in feats.columns


def test_cli_features_activity_bad_spec_exits(tmp_path):
    data = _make_gift_export(tmp_path, n_donors=2)
    with pytest.raises(SystemExit, match="--activity must be TYPE=PATH"):
        main(["features", "--source", "raisers_edge", "--data", str(data),
              "--activity", "no-equals-sign"])


# --------------------------------------------------------------------------- #
# `train --task upgrade`
# --------------------------------------------------------------------------- #
_UPGRADE_YEARS = ["2020-08-01", "2021-08-01", "2022-08-01", "2023-08-01", "2024-08-01"]
_UPGRADE_ARCHETYPES = {
    "flat_high": [900, 900, 900, 900, 900],
    "early_riser": [600, 1050, 1200, 1400, 1600],
    "late_riser": [300, 500, 700, 1100, 1300],
    "flat_low": [400, 400, 400, 400, 400],
}


def _make_upgrade_gift_export(tmp_path, name="upgrade_gifts.csv", n_per_group=20):
    rows = ["Constituent ID,Gift Date,Gift Amount\n"]
    for archetype, amounts in _UPGRADE_ARCHETYPES.items():
        for i in range(n_per_group):
            donor_id = f"{archetype}_{i}"
            for year, amount in zip(_UPGRADE_YEARS, amounts):
                rows.append(f"{donor_id},{year},{amount}.00\n")
    path = tmp_path / name
    path.write_text("".join(rows))
    return path


def test_cli_train_task_upgrade_writes_scored_csv_and_prints_report(tmp_path, capsys):
    data = _make_upgrade_gift_export(tmp_path)
    out_path = tmp_path / "scored.csv"
    main(["train", "--task", "upgrade", "--source", "raisers_edge",
          "--data", str(data), "--out", str(out_path), "--random-state", "0"])

    scored = pd.read_csv(out_path)
    assert len(scored) == 40  # flat_high + flat_low, still band-qualifying
    assert list(scored.columns) == [
        "donor_id", "fiscal_year", "affinity_score", "rank", "decile",
        "top_reasons", "suggested_ask",
    ]
    out = capsys.readouterr().out
    assert "Wrote 40 scored rows" in out
    assert "n_training_rows" in out


def test_cli_train_task_upgrade_requires_source(tmp_path):
    data = _make_upgrade_gift_export(tmp_path)
    with pytest.raises(SystemExit, match="requires --source"):
        main(["train", "--task", "upgrade", "--data", str(data),
              "--out", str(tmp_path / "scored.csv")])


def test_cli_train_task_upgrade_npsp_source(tmp_path):
    rows = ["Account ID,Close Date,Amount,Stage\n"]
    for archetype, amounts in _UPGRADE_ARCHETYPES.items():
        for i in range(10):
            donor_id = f"{archetype}_{i}"
            for year, amount in zip(_UPGRADE_YEARS, amounts):
                rows.append(f"{donor_id},{year},{amount}.00,Closed Won\n")
    data = tmp_path / "opportunities.csv"
    data.write_text("".join(rows))

    out_path = tmp_path / "scored.csv"
    main(["train", "--task", "upgrade", "--source", "npsp", "--data", str(data),
          "--out", str(out_path), "--random-state", "0"])
    scored = pd.read_csv(out_path)
    assert len(scored) == 20  # flat_high + flat_low, 10 each


def test_cli_train_task_upgrade_with_donors_csv(tmp_path):
    data = _make_upgrade_gift_export(tmp_path)
    donors_path = tmp_path / "donors.csv"
    donor_ids = [f"{a}_{i}" for a in _UPGRADE_ARCHETYPES for i in range(20)]
    lines = ["donor_id,wealth_rating\n"] + [f"{d},A\n" for d in donor_ids]
    donors_path.write_text("".join(lines))

    out_path = tmp_path / "scored.csv"
    main(["train", "--task", "upgrade", "--source", "raisers_edge",
          "--data", str(data), "--donors", str(donors_path),
          "--out", str(out_path), "--random-state", "0"])
    assert len(pd.read_csv(out_path)) == 40


def test_cli_train_task_upgrade_donors_csv_requires_donor_id_column(tmp_path):
    data = _make_upgrade_gift_export(tmp_path)
    donors_path = tmp_path / "donors.csv"
    donors_path.write_text("not_donor_id,wealth_rating\n1,A\n")
    with pytest.raises(SystemExit, match="donor_id"):
        main(["train", "--task", "upgrade", "--source", "raisers_edge",
              "--data", str(data), "--donors", str(donors_path),
              "--out", str(tmp_path / "scored.csv")])


def test_cli_train_plain_task_unaffected_by_new_flags(tmp_path):
    """--task defaults to 'plain' and every existing train behaviour is
    untouched: this is the same call test_cli_train_score_validate makes."""
    data = _make_csv(tmp_path, "train.csv")
    model_path = tmp_path / "m.joblib"
    main(["train", "--data", str(data), "--target", "is_major_donor",
          "--features", FEATURES, "--out", str(model_path)])
    assert model_path.exists()


def test_python_and_cli_upgrade_paths_produce_identical_scores(tmp_path):
    """The brief's acceptance test: score_upgrade_prospects called directly
    on the parsed raw export must match `train --task upgrade`'s CLI output
    for the same three CSVs, to floating-point tolerance."""
    from philanthropy.cli import _read_activities, _read_raw_gifts
    from philanthropy.models import score_upgrade_prospects

    data = _make_upgrade_gift_export(tmp_path)
    activity_path = tmp_path / "activities.csv"
    donor_ids = [f"{a}_{i}" for a in _UPGRADE_ARCHETYPES for i in range(20)]
    activity_path.write_text(
        "contact_id,activity_date\n"
        + "\n".join(f"{d},2024-08-01" for d in donor_ids[:10])
        + "\n"
    )
    donors_path = tmp_path / "donors.csv"
    donors_path.write_text(
        "donor_id,wealth_rating\n" + "\n".join(f"{d},A" for d in donor_ids) + "\n"
    )

    gifts = _read_raw_gifts("raisers_edge", str(data))
    activities = _read_activities([f"event={activity_path}"])
    donors = pd.read_csv(donors_path).set_index("donor_id")
    direct_scores, direct_report = score_upgrade_prospects(
        gifts, activities=activities, donors=donors, random_state=0
    )

    scored_path = tmp_path / "scored.csv"
    main([
        "train", "--task", "upgrade", "--source", "raisers_edge",
        "--data", str(data), "--activity", f"event={activity_path}",
        "--donors", str(donors_path), "--out", str(scored_path),
        "--random-state", "0",
    ])
    cli_scores = pd.read_csv(scored_path).set_index("donor_id").sort_index()
    direct_sorted = direct_scores.sort_index()
    cli_scores.index = cli_scores.index.astype(str)
    direct_sorted.index = direct_sorted.index.astype(str)

    assert list(cli_scores.index) == list(direct_sorted.index)
    pd.testing.assert_series_equal(
        cli_scores["affinity_score"].astype(float),
        direct_sorted["affinity_score"].astype(float),
        check_names=False, check_exact=False,
    )
    assert list(cli_scores["rank"]) == list(direct_sorted["rank"])
    assert list(cli_scores["decile"]) == list(direct_sorted["decile"])
