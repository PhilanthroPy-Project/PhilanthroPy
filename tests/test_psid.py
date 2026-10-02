"""
tests/test_psid.py
===================
Unit tests for the PSID individual-level cross-year extract reader. No real
PSID data anywhere here: every fixture is synthetic, built in the same
fixed-width ``.txt`` + Stata ``infix`` ``.do`` layout as a real PSID Data
Center download, but with fabricated households and fabricated values.
"""

import pandas as pd
import pytest

from philanthropy.datasets import load_psid_philanthropy

# (variable, field width) for the columns this test fixture supplies, one
# tuple per PSID wave. Values are fabricated; widths are chosen for a
# compact fixture, not copied from the real file.
_WAVE_1_2001 = [
    ("ER30001", 4), ("ER30002", 3),  # household key (shared every wave)
    ("ER33603", 2),  # relation to head (10 = head)
    ("ER20047", 6), ("ER20053", 6), ("ER20059", 6), ("ER20065", 6),
    ("ER20071", 6), ("ER20083", 6),  # giving categories + 2001 checkpoint
    ("ER19162", 6),  # G102A itemized
    ("ER20456", 7),  # family income
    ("S516", 7), ("S517", 7),  # wealth1, wealth2
    ("ER20089", 4), ("ER20097", 4),  # head/spouse annual volunteer hours
]
_WAVE_2003_REGULAR_HOURS_VARS = [
    "ER23563", "ER23573", "ER23582", "ER23591", "ER23600", "ER23609",
    "ER23618", "ER23627", "ER23636", "ER23645", "ER23654", "ER23663",
    "ER23673", "ER23683",
]
_WAVE_2003 = [
    ("ER33703", 2),
    ("ER23483", 6), ("ER23489", 6), ("ER23495", 6), ("ER23501", 6),
    ("ER23507", 6), ("ER23513", 6), ("ER23519", 6), ("ER23525", 6),
    ("ER23531", 6), ("ER23537", 6), ("ER23543", 6),
    ("ER22535", 6), ("ER24099", 7), ("S616", 7), ("S617", 7),
] + [(var, 5) for var in _WAVE_2003_REGULAR_HOURS_VARS]
_WAVE_2017 = [
    ("ER34503", 2),
    ("ER71042", 6), ("ER71044", 6), ("ER71046", 6), ("ER71048", 6),
    ("ER71050", 6), ("ER71052", 6), ("ER71054", 6), ("ER71056", 6),
    ("ER71058", 6), ("ER71060", 6), ("ER71063", 6),
    ("ER67757", 6), ("ER71426", 7), ("ER71483", 7), ("ER71485", 7),
    ("ER66720", 4), ("ER66733", 4),
]
_COLUMNS = _WAVE_1_2001 + _WAVE_2003 + _WAVE_2017


def _build_fixture(tmp_path, rows):
    """rows: list of dicts, var -> value. Missing vars default to 0."""
    positions = []
    pos = 1
    for var, width in _COLUMNS:
        positions.append((var, pos, pos + width - 1))
        pos += width

    do_lines = ["infix"]
    do_lines += [f"  {var} {start} - {end}" for var, start, end in positions]
    do_lines.append("using fake.txt, clear")
    do_path = tmp_path / "fixture.do"
    do_path.write_text("\n".join(do_lines) + "\n;\n")

    data_lines = []
    for row in rows:
        line = ""
        for var, width in _COLUMNS:
            value = row.get(var, 0)
            line += str(value).rjust(width)
        data_lines.append(line)
    data_path = tmp_path / "fixture.txt"
    data_path.write_text("\n".join(data_lines) + "\n")

    return str(data_path), str(do_path)


def _sentinel(width, dk=False):
    return int("9" * (width - 1) + "8") if dk else int("9" * width)


def test_returns_expected_columns(tmp_path):
    rows = [{"ER30001": 1, "ER30002": 1, "ER33603": 10, "ER33703": 10, "ER34503": 10}]
    data_path, do_path = _build_fixture(tmp_path, rows)

    df = load_psid_philanthropy(data_path, do_path)

    assert "household_key" in df.columns
    assert "year" in df.columns
    assert "total_giving" in df.columns
    assert "giving_religious" in df.columns
    assert "head_volunteer_hours_annual" in df.columns
    assert "household_volunteer_hours_regular" in df.columns
    assert "head_volunteer_hours_typical_week" in df.columns


def test_household_linked_across_waves(tmp_path):
    rows = [{"ER30001": 42, "ER30002": 7, "ER33603": 10, "ER33703": 10, "ER34503": 10}]
    data_path, do_path = _build_fixture(tmp_path, rows)

    df = load_psid_philanthropy(data_path, do_path)

    assert set(df["household_key"]) == {42007}
    assert sorted(df["year"]) == [2001, 2003, 2017]


def test_total_giving_sums_available_categories(tmp_path):
    rows = [{
        "ER30001": 1, "ER30002": 1, "ER33603": 10,
        "ER20047": 500, "ER20053": 100, "ER20059": 0, "ER20065": 0,
        "ER20071": 0, "ER20083": 25,
    }]
    data_path, do_path = _build_fixture(tmp_path, rows)

    df = load_psid_philanthropy(data_path, do_path)
    row_2001 = df[df["year"] == 2001].iloc[0]

    assert row_2001["total_giving"] == 625
    assert row_2001["giving_religious"] == 500
    assert pd.isna(row_2001["giving_youth"])  # 2001 has no per-category youth amount


def test_missing_code_recoded_to_nan(tmp_path):
    rows = [{
        "ER30001": 1, "ER30002": 1, "ER33603": 10,
        "ER20047": _sentinel(6), "ER20053": 300,
    }]
    data_path, do_path = _build_fixture(tmp_path, rows)

    df = load_psid_philanthropy(data_path, do_path)
    row_2001 = df[df["year"] == 2001].iloc[0]

    assert pd.isna(row_2001["giving_religious"])
    assert row_2001["giving_combo"] == 300
    assert row_2001["total_giving"] == 300  # the NA category is excluded, not zeroed


def test_non_head_rows_excluded_per_wave(tmp_path):
    rows = [
        # Head in 2001 only (e.g. a spouse, relation 20, in 2003/2017).
        {"ER30001": 1, "ER30002": 1, "ER33603": 10, "ER33703": 20, "ER34503": 20},
        # Head in 2003 only (e.g. not yet in the family in 2001/2017: inap, code 0).
        {"ER30001": 2, "ER30002": 1, "ER33603": 0, "ER33703": 10, "ER34503": 0},
    ]
    data_path, do_path = _build_fixture(tmp_path, rows)

    df = load_psid_philanthropy(data_path, do_path)

    h1 = df[df["household_key"] == 1001]
    h2 = df[df["household_key"] == 2001]
    assert sorted(h1["year"]) == [2001]
    assert sorted(h2["year"]) == [2003]


def test_volunteer_hours_kept_as_separate_measures(tmp_path):
    rows = [{
        "ER30001": 1, "ER30002": 1, "ER33603": 10, "ER33703": 10, "ER34503": 10,
        "ER20089": 50, "ER20097": 20,  # 2001 annual hours
        "ER66720": 5, "ER66733": 3,  # 2017 typical-week hours
    }]
    data_path, do_path = _build_fixture(tmp_path, rows)

    df = load_psid_philanthropy(data_path, do_path)
    row_2001 = df[df["year"] == 2001].iloc[0]
    row_2003 = df[df["year"] == 2003].iloc[0]
    row_2017 = df[df["year"] == 2017].iloc[0]

    assert row_2001["head_volunteer_hours_annual"] == 50
    assert pd.isna(row_2001["head_volunteer_hours_typical_week"])
    assert pd.isna(row_2003["head_volunteer_hours_annual"])
    assert pd.isna(row_2003["head_volunteer_hours_typical_week"])
    assert pd.isna(row_2017["head_volunteer_hours_annual"])
    assert row_2017["head_volunteer_hours_typical_week"] == 5


def test_regular_volunteer_hours_summed_for_2003_only(tmp_path):
    rows = [{
        "ER30001": 1, "ER30002": 1, "ER33603": 10, "ER33703": 10, "ER34503": 10,
        "ER23563": 10, "ER23573": 5, "ER23582": _sentinel(5, dk=True),
    }]
    data_path, do_path = _build_fixture(tmp_path, rows)

    df = load_psid_philanthropy(data_path, do_path)
    row_2001 = df[df["year"] == 2001].iloc[0]
    row_2003 = df[df["year"] == 2003].iloc[0]
    row_2017 = df[df["year"] == 2017].iloc[0]

    assert row_2003["household_volunteer_hours_regular"] == 15  # DK slot excluded, not zeroed
    assert pd.isna(row_2001["household_volunteer_hours_regular"])
    assert pd.isna(row_2017["household_volunteer_hours_regular"])


def test_family_income_not_recoded_as_missing(tmp_path):
    # A legitimate 7-digit income that happens to equal the giving-field
    # sentinel pattern must stay a real number: family_income has no
    # missing-code convention, unlike giving/itemized/hours amounts.
    rows = [{
        "ER30001": 1, "ER30002": 1, "ER33603": 10,
        "ER20456": 9999999,
    }]
    data_path, do_path = _build_fixture(tmp_path, rows)

    df = load_psid_philanthropy(data_path, do_path)
    row_2001 = df[df["year"] == 2001].iloc[0]

    assert row_2001["family_income"] == 9999999


def test_wave_missing_from_extract_is_skipped(tmp_path):
    # A fixture that only ever includes the 2001 columns: 2003 and 2017 (and
    # every other wave in _WAVES) simply never appear in the output.
    positions = []
    pos = 1
    for var, width in _WAVE_1_2001:
        positions.append((var, pos, pos + width - 1))
        pos += width
    do_lines = ["infix"]
    do_lines += [f"  {var} {start} - {end}" for var, start, end in positions]
    do_lines.append("using fake.txt, clear")
    do_path = tmp_path / "fixture.do"
    do_path.write_text("\n".join(do_lines) + "\n;\n")

    line = "".join(str({"ER30001": 1, "ER30002": 1, "ER33603": 10}.get(var, 0)).rjust(width)
                    for var, width in _WAVE_1_2001)
    data_path = tmp_path / "fixture.txt"
    data_path.write_text(line + "\n")

    df = load_psid_philanthropy(str(data_path), str(do_path))

    assert sorted(df["year"].unique().tolist()) == [2001]
