"""Lock the benchmark table to scripts/benchmark_models.py output."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN = REPO_ROOT / "docs" / "explanation" / "benchmark_results.txt"
SCRIPT = REPO_ROOT / "scripts" / "benchmark_models.py"


def test_benchmark_table_is_reproducible():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    golden = GOLDEN.read_text()
    assert result.stdout == golden, (
        "benchmark output drifted from docs/explanation/benchmark_results.txt; "
        "regenerate on Linux CI and commit verbatim"
    )
