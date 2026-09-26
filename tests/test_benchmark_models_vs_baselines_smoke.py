"""A tiny, offline smoke check for scripts/benchmark_models_vs_baselines.py.

Not a golden-file lock like tests/test_benchmark_models.py: that script's
numbers are fitted-model metrics that move with the BLAS/sklearn build, and
this one adds a KDD Cup 1998 section that must never run under test (it is
network-gated, see tests/test_no_network.py). This just proves ``--fast
--skip-kdd98`` runs to completion, offline, and produces the shape of table
the module docstring promises.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "benchmark_models_vs_baselines.py"


def test_fast_synthetic_run_is_offline_and_reports_a_verdict():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--fast", "--skip-kdd98"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "kdd98" not in result.stdout, "--skip-kdd98 must not touch the KDD98 section"
    assert "synthetic_panel" in result.stdout
    assert "verdict" in result.stdout
    assert any(v in result.stdout for v in ("wins", "modest", "loses"))
    assert "Runtime:" in result.stdout
