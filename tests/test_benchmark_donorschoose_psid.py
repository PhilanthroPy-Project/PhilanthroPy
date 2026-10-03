"""
tests/test_benchmark_donorschoose_psid.py
==========================================
Unit tests for the PSID helpers in scripts/benchmark_models_vs_baselines.py.
No real PSID data: a tiny synthetic fixture in the same fixed-width
``.txt`` + Stata ``infix`` ``.do`` layout a real PSID Data Center download
uses, fabricated households and values only.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_models_vs_baselines as bm  # noqa: E402

_WAVE_1_2001 = [
    ("ER30001", 4), ("ER30002", 3),
    ("ER33603", 2),
    ("ER20047", 6), ("ER20053", 6), ("ER20059", 6), ("ER20065", 6),
    ("ER20071", 6), ("ER20083", 6),
    ("ER19162", 6), ("ER20456", 7),
    ("S516", 7), ("S517", 7),
    ("ER20089", 4), ("ER20097", 4),
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
_COLUMNS = _WAVE_1_2001 + _WAVE_2003
_2003_GIVING_VARS = ["ER23483", "ER23489", "ER23495", "ER23501", "ER23507", "ER23513",
                      "ER23519", "ER23525", "ER23531", "ER23537", "ER23543"]


def _build_fixture(tmp_path, rows):
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
        line = "".join(str(row.get(var, 0)).rjust(width) for var, width in _COLUMNS)
        data_lines.append(line)
    data_path = tmp_path / "fixture.txt"
    data_path.write_text("\n".join(data_lines) + "\n")
    return str(data_path), str(do_path)


def test_attrition_and_nan_next_wave_excluded_not_counted_as_lapsed(tmp_path):
    rows = [
        # hh1: Head both waves, gives in both -> included, not lapsed.
        {"ER30001": 1, "ER30002": 1, "ER33603": 10, "ER33703": 10,
         "ER20047": 300, "ER23483": 400},
        # hh2: Head in 2001 only (not Head in 2003, code 0 = inapplicable):
        # attrition, not observed in the next wave at all.
        {"ER30001": 2, "ER30002": 1, "ER33603": 10, "ER33703": 0,
         "ER20047": 300},
        # hh3: Head both waves, but every 2003 giving category is the
        # Don't-Know sentinel (99999 8 = width-6 DK code), so total_giving
        # in 2003 is NaN, not $0.
        {"ER30001": 3, "ER30002": 1, "ER33603": 10, "ER33703": 10,
         "ER20047": 300, **{v: 999998 for v in _2003_GIVING_VARS}},
    ]
    data_path, do_path = _build_fixture(tmp_path, rows)

    snap = bm._psid_wave_period_snapshots(data_path, do_path, "lapse", False, 1000.0, (100.0, 999.0))
    wave_2001 = snap[snap["wave"] == 2001]

    assert set(wave_2001["household_key"]) == {1001}
    assert int(wave_2001.loc[wave_2001["household_key"] == 1001, "target"].iloc[0]) == 0
