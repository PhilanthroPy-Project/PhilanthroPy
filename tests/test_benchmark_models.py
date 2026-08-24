"""Lock the benchmark table to scripts/benchmark_models.py output."""

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN = REPO_ROOT / "docs" / "explanation" / "benchmark_results.txt"
SCRIPT = REPO_ROOT / "scripts" / "benchmark_models.py"

# The metrics come from fitted sklearn models, so the third decimal moves with
# the BLAS/sklearn build. CI showed 0.732 vs 0.731 drift across Python versions
# on the same commit; anything past this band means the benchmark itself broke.
TOLERANCE = 0.01
DECIMAL = re.compile(r"\d+\.\d+")


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
        actual_numbers = [float(value) for value in DECIMAL.findall(actual)]
        golden_numbers = [float(value) for value in DECIMAL.findall(golden)]
        for actual_value, golden_value in zip(actual_numbers, golden_numbers):
            assert abs(actual_value - golden_value) <= TOLERANCE, (
                f"benchmark metric drifted beyond {TOLERANCE}: "
                f"expected {golden_value}, got {actual_value}\n"
                f"line: {actual}"
            )
