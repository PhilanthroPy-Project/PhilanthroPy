"""Lock the benchmark table to scripts/benchmark_models.py output."""

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN = REPO_ROOT / "docs" / "explanation" / "benchmark_results.txt"
SCRIPT = REPO_ROOT / "scripts" / "benchmark_models.py"

# The metrics come from fitted sklearn models, so they move with the
# BLAS/sklearn build. CI showed means drifting up to ~0.006 and seed range
# endpoints up to ~0.015 across Python versions on the same commit; anything
# past these bands means the benchmark itself broke.
MEAN_TOLERANCE = 0.02
RANGE_TOLERANCE = 0.05
DECIMAL = re.compile(r"\d+\.\d+")
CELL = re.compile(r"(\d+\.\d+) \((\d+\.\d+)-(\d+\.\d+)\)")


def test_benchmark_table_is_reproducible():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    golden_lines = GOLDEN.read_text().splitlines()
    actual_lines = result.stdout.splitlines()
    assert len(actual_lines) == len(golden_lines), (
        "benchmark output changed shape; regenerate "
        "docs/explanation/benchmark_results.txt and commit verbatim"
    )
    for actual, golden in zip(actual_lines, golden_lines):
        assert DECIMAL.sub("#", actual) == DECIMAL.sub("#", golden), (
            f"non-metric part of the benchmark table drifted:\n"
            f"expected: {golden}\nactual:   {actual}"
        )
        actual_cells = CELL.findall(actual)
        golden_cells = CELL.findall(golden)
        assert len(actual_cells) == len(golden_cells), (
            f"metric cell count drifted:\nexpected: {golden}\nactual:   {actual}"
        )
        for actual_cell, golden_cell in zip(actual_cells, golden_cells):
            for actual_value, golden_value, tolerance in zip(
                actual_cell, golden_cell, (MEAN_TOLERANCE,) + (RANGE_TOLERANCE,) * 2
            ):
                assert abs(float(actual_value) - float(golden_value)) <= tolerance, (
                    f"benchmark metric drifted beyond {tolerance}: "
                    f"expected {golden_value}, got {actual_value}\n"
                    f"line: {actual}"
                )
