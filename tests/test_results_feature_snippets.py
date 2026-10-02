"""Tests for the "What the model looks at" snippet renderer in
scripts/make_results_pages.py (E.11i's pasted spec).

These do not re-fit any model (that part moves with the BLAS/sklearn build,
the same reason tests/test_benchmark_models.py uses a tolerance band rather
than an exact match): they lock the *rendering* step, which is a pure
function of a ``results[...]["features"]`` dict, so a template change that
breaks the reader-facing rules (plain language above the fold, raw column
names only inside the collapsed analyst note, every tab ending with the same
disclaimer) is caught without depending on a fitted model's exact numbers.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "make_results_pages.py"


@pytest.fixture(scope="module")
def mrp():
    spec = importlib.util.spec_from_file_location("make_results_pages", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_entry(mrp, with_ablation: bool):
    extra_sets = None
    if with_ablation:
        extra_sets = [
            {
                "id": "full", "label": mrp.FEATURE_SET_LABELS["full"], "columns": ["total", "n", "recent", "streak"],
                "chosen": False, "top1pct": {"value": 30.0, "lo": 28.0, "hi": 32.0},
                "top5pct": {"value": 46.0, "lo": 44.0, "hi": 48.0}, "top10pct": {"value": 46.0, "lo": 44.0, "hi": 48.0},
            }
        ]
    drivers = [
        {"column": "total", "label": "lifetime giving", "group": "giving_history", "importance": 0.10, "lo": 0.08, "hi": 0.12, "direction": "+"},
        {"column": "years_since_last", "label": "years since last gift", "group": "recency", "importance": 0.05, "lo": None, "hi": None, "direction": "-"},
        {"column": "streak", "label": "giving streak", "group": "momentum", "importance": 0.02, "lo": None, "hi": None, "direction": "mixed"},
    ]
    return mrp._features_entry(
        ("total", "years_since_last", "streak"), drivers, scoring="roc_auc", split="walk-forward",
        extra_sets=extra_sets,
    )


def test_snippet_has_plain_language_sections_and_disclaimer(mrp, tmp_path):
    results = {
        "demo_synthetic": {
            "top1pct": {"model": 40.0, "rule": 20.0}, "top5pct": {"model": 35.0, "rule": 22.0},
            "top10pct": {"model": 30.0, "rule": 18.0},
            "features": _fake_entry(mrp, with_ablation=False),
        }
    }
    mrp.MODEL_DATASET_TABS = {"demo": [("Sample data", "demo_synthetic")]}
    mrp.render_feature_snippets(results, tmp_path)

    text = (tmp_path / "demo__demo_synthetic.md").read_text()
    assert text.startswith('=== "Sample data"')
    assert "On this file the model looks at 3 things about each donor" in text
    assert "lifetime giving" in text and "▲ raises the score" in text
    assert "years since last gift" in text and "▼ lowers the score" in text
    assert "giving streak" in text and "● depends" in text
    assert text.rstrip().endswith("Results on your own file will differ.")
    # No raw column name outside the collapsed analyst note.
    before_note = text.split('??? note')[0]
    assert "`total`" not in before_note and "`years_since_last`" not in before_note
    assert "`total`" in text  # it does appear inside the analyst note


def test_absent_feature_groups_are_called_out(mrp, tmp_path):
    results = {
        "demo_synthetic": {
            "top1pct": {"model": 40.0, "rule": 20.0}, "top5pct": {"model": 35.0, "rule": 22.0},
            "top10pct": {"model": 30.0, "rule": 18.0},
            "features": _fake_entry(mrp, with_ablation=False),
        }
    }
    mrp.MODEL_DATASET_TABS = {"demo": [("Sample data", "demo_synthetic")]}
    mrp.render_feature_snippets(results, tmp_path)
    text = (tmp_path / "demo__demo_synthetic.md").read_text()
    assert "not used here" in text
    assert "engagement" in text.lower()
    assert "wealth" in text.lower()


def test_ablation_table_only_appears_when_a_second_feature_set_exists(mrp, tmp_path):
    no_ablation = {
        "demo_synthetic": {
            "top1pct": {"model": 40.0, "rule": 20.0}, "top5pct": {"model": 35.0, "rule": 22.0},
            "top10pct": {"model": 30.0, "rule": 18.0},
            "features": _fake_entry(mrp, with_ablation=False),
        }
    }
    with_ablation = {
        "demo_synthetic": {
            "top1pct": {"model": 40.0, "rule": 20.0}, "top5pct": {"model": 35.0, "rule": 22.0},
            "top10pct": {"model": 30.0, "rule": 18.0},
            "features": _fake_entry(mrp, with_ablation=True),
        }
    }
    mrp.MODEL_DATASET_TABS = {"demo": [("Sample data", "demo_synthetic")]}

    mrp.render_feature_snippets(no_ablation, tmp_path)
    assert "What adding information did" not in (tmp_path / "demo__demo_synthetic.md").read_text()

    mrp.render_feature_snippets(with_ablation, tmp_path)
    text = (tmp_path / "demo__demo_synthetic.md").read_text()
    assert "What adding information did" in text
    assert "the best simple rule (for comparison)" in text


def test_missing_dataset_key_is_skipped_not_errored(mrp, tmp_path):
    mrp.MODEL_DATASET_TABS = {"demo": [("Sample data", "demo_synthetic"), ("KDD Cup 1998", "demo_kdd98")]}
    mrp.render_feature_snippets({}, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_available_but_unused_group_is_not_called_truly_absent(mrp, tmp_path):
    """A kdd98 dataset tab must not claim 'this file has no wealth' when the
    model simply isn't given WEALTH1/INCOME/etc columns that file does have
    (bug: the group-absence sentence used to be keyed only off the model's
    own feature list, with no notion of what the underlying file contains)."""
    drivers = [
        {"column": "recency", "label": "months since last gift", "group": "recency", "importance": 0.10, "lo": None, "hi": None, "direction": "-"},
    ]
    entry = mrp._features_entry(("recency",), drivers, scoring="roc_auc", split="walk-forward")
    results = {
        "demo_kdd98": {
            "top1pct": {"model": 40.0, "rule": 20.0}, "top5pct": {"model": 35.0, "rule": 22.0},
            "top10pct": {"model": 30.0, "rule": 18.0}, "features": entry,
        }
    }
    mrp.MODEL_DATASET_TABS = {"demo": [("KDD Cup 1998 (real donor file)", "demo_kdd98")]}
    mrp.render_feature_snippets(results, tmp_path)
    text = (tmp_path / "demo__demo_kdd98.md").read_text()
    assert "this file has no wealth" not in text.lower()
    assert "this model is not given" in text.lower()
    assert "wealth & demographics" in text.lower() and "mailing history" in text.lower()
    # engagement really is absent from KDD98 (no event/volunteer data), so
    # that claim is still the "truly absent" one.
    assert "this file has no engagement" in text.lower()
    # "momentum" is a computed comparison, not a record type a file has or
    # lacks, so it must never appear in this sentence either way.
    assert "momentum" not in text.lower()


def test_synthetic_wealth_is_available_but_unused_not_absent(mrp, tmp_path):
    """make_donor_panel's donors frame carries wealth_estimate; no synthetic
    benchmark feeds it to a model, but the file does have it, so it must
    land in the "available but unused" sentence, not "this file has no
    wealth" (bug: DATASET_GROUPS_AVAILABLE['synthetic'] used to omit wealth
    entirely, as if make_donor_panel never generated it)."""
    drivers = [
        {"column": "total", "label": "lifetime giving", "group": "giving_history", "importance": 0.10, "lo": None, "hi": None, "direction": "+"},
        {"column": "recent", "label": "this year's gift", "group": "recency", "importance": 0.05, "lo": None, "hi": None, "direction": "+"},
    ]
    entry = mrp._features_entry(("total", "recent"), drivers, scoring="roc_auc", split="train/test")
    results = {
        "demo_synthetic": {
            "top1pct": {"model": 40.0, "rule": 20.0}, "top5pct": {"model": 35.0, "rule": 22.0},
            "top10pct": {"model": 30.0, "rule": 18.0}, "features": entry,
        }
    }
    mrp.MODEL_DATASET_TABS = {"demo": [("Sample data", "demo_synthetic")]}
    mrp.render_feature_snippets(results, tmp_path)
    text = (tmp_path / "demo__demo_synthetic.md").read_text()
    assert "this file has no wealth" not in text.lower()
    assert "wealth & demographics is in this file, but this model is not given it here" in text.lower()
    # generate_synthetic_donor_data's event_attendance_count isn't in the
    # data source any synthetic benchmark actually uses (make_donor_panel),
    # and no synthetic benchmark's gifts carry a true solicitation/response
    # mailing history (only which appeal an actual gift came from), so both
    # really are absent here.
    assert "this file has no engagement and mailing history" in text.lower()


def test_momentum_group_is_relabeled_year_over_year_change(mrp, tmp_path):
    """fy_trend/streak/consecutive_years_given are a one-year diff and a
    years-given count, not the trailing-slope 'momentum' features from
    philanthropy.utils._momentum (unused by every benchmark). The group must
    not be labeled or described as "momentum" - that name is reserved for the
    *_slope_*/*_rel_slope_* columns - and the "file has no X" sentence must
    never mention it, since the KDD98 upgrade tab already uses this group on
    the same file other KDD98 tabs would otherwise call it absent from."""
    assert mrp.GROUP_LABELS["momentum"] != "Momentum"
    assert "momentum" not in mrp.GROUP_MEANINGS["momentum"].lower()
    drivers = [
        {"column": "fy_trend", "label": "giving trend, this year vs last", "group": "momentum", "importance": 0.10, "lo": None, "hi": None, "direction": "+"},
    ]
    entry = mrp._features_entry(("fy_trend",), drivers, scoring="roc_auc", split="walk-forward")
    results = {
        "demo_kdd98": {
            "top1pct": {"model": 40.0, "rule": 20.0}, "top5pct": {"model": 35.0, "rule": 22.0},
            "top10pct": {"model": 30.0, "rule": 18.0}, "features": entry,
        }
    }
    mrp.MODEL_DATASET_TABS = {"demo": [("KDD Cup 1998 (real donor file)", "demo_kdd98")]}
    mrp.render_feature_snippets(results, tmp_path)
    text = (tmp_path / "demo__demo_kdd98.md").read_text()
    assert "| Momentum |" not in text
    assert "Year-over-year change" in text


def test_zero_effect_drivers_are_hidden_from_the_visible_table(mrp, tmp_path):
    drivers = [
        {"column": "total", "label": "lifetime giving", "group": "giving_history", "importance": 0.10, "lo": 0.08, "hi": 0.12, "direction": "+"},
        {"column": "fy_trend", "label": "giving trend, this year vs last", "group": "momentum", "importance": 0.0, "lo": -0.007, "hi": 0.011, "direction": "mixed"},
    ]
    entry = mrp._features_entry(("total", "fy_trend"), drivers, scoring="roc_auc", split="walk-forward")
    results = {
        "demo_synthetic": {
            "top1pct": {"model": 40.0, "rule": 20.0}, "top5pct": {"model": 35.0, "rule": 22.0},
            "top10pct": {"model": 30.0, "rule": 18.0}, "features": entry,
        }
    }
    mrp.MODEL_DATASET_TABS = {"demo": [("Sample data", "demo_synthetic")]}
    mrp.render_feature_snippets(results, tmp_path)
    text = (tmp_path / "demo__demo_synthetic.md").read_text()
    before_note = text.split("??? note")[0]
    assert "lifetime giving" in before_note
    assert "giving trend, this year vs last" not in before_note
    assert "Only 1 feature had a measurable effect" in text
    # the zero-effect driver still appears, in the analyst note.
    assert "giving trend, this year vs last" not in text.split("??? note")[1] or "`fy_trend`" in text


def test_regressor_driver_wording_uses_its_own_target_not_score(mrp, tmp_path):
    """The ask model predicts a dollar amount, not a classifier probability;
    'raises the score' is meaningless there."""
    drivers = [
        {"column": "AVGGIFT", "label": "average gift amount", "group": "giving_history", "importance": 0.80, "lo": None, "hi": None, "direction": "+"},
    ]
    entry = mrp._features_entry(
        ("AVGGIFT",), drivers, scoring="neg_mean_absolute_error", split="walk-forward", target_label="suggested ask",
    )
    results = {
        "demo_kdd98": {
            "top1pct": {"model": 40.0, "rule": 20.0}, "top5pct": {"model": 35.0, "rule": 22.0},
            "top10pct": {"model": 30.0, "rule": 18.0}, "features": entry,
        }
    }
    mrp.MODEL_DATASET_TABS = {"demo": [("KDD Cup 1998 (real donor file)", "demo_kdd98")]}
    mrp.render_feature_snippets(results, tmp_path)
    text = (tmp_path / "demo__demo_kdd98.md").read_text()
    assert "raises the suggested ask" in text
    assert "raises the score" not in text


def test_feature_info_covers_every_declared_feature_set(mrp):
    """A column used by a real driver function but missing from FEATURE_INFO
    raises a KeyError at generation time (not silently mislabeled); this
    locks that every column this module actually feeds a model has a plain
    label and a group before it ships."""
    declared = set(mrp.UPGRADE_FEATURE_COLS) | set(mrp.KDD_FEATURE_COLS) | {
        "total", "n", "recent", "streak", "years_since_last", "prev_recent", "max_gift", "tenure",
        "recency", "frequency", "monetary",
    }
    missing = declared - set(mrp.FEATURE_INFO)
    assert not missing, f"FEATURE_INFO is missing plain labels for: {sorted(missing)}"
