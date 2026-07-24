"""tests/test_doc_examples.py

Executes the fenced ``python`` blocks in the tutorial and how-to docs so that a
broken public API in the documentation fails CI instead of a user's copy-paste
(see the P0-2 grateful-patient tutorial regression).

Each doc file's python blocks are concatenated in order and executed in a single
fresh namespace (later blocks may rely on earlier imports/variables). A file is
skipped — visibly — when it cannot run in CI because its code reads external
data files or the network, or when it carries an explicit ``<!-- docs-notest -->``
marker.
"""

import os
import re
from pathlib import Path

import pytest

try:  # keep any plotting in the docs headless
    import matplotlib

    matplotlib.use("Agg")
except Exception:  # pragma: no cover - matplotlib is a dev/docs dependency
    pass

DOCS_ROOT = Path(__file__).resolve().parent.parent / "docs"
DOC_DIRS = ("tutorials", "how-to")

_PY_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
# If any of these appear in the file, its code touches state CI can't provide.
_UNRUNNABLE_MARKERS = (
    "read_csv",
    "read_parquet",
    "read_excel",
    "open(",
    "requests",
    "urlopen",
    "docs-notest",
)


def _doc_files():
    files = []
    for d in DOC_DIRS:
        files.extend(sorted((DOCS_ROOT / d).glob("*.md")))
    return files


@pytest.mark.parametrize(
    "path", _doc_files(), ids=lambda p: f"{p.parent.name}/{p.name}"
)
def test_doc_python_blocks_execute(path, tmp_path):
    text = path.read_text()
    code = "\n\n".join(_PY_FENCE.findall(text))
    if not code.strip():
        pytest.skip("no python code blocks")
    if any(marker in text for marker in _UNRUNNABLE_MARKERS):
        pytest.skip("requires external data/network or marked docs-notest")

    # Execute inside a throwaway cwd so any files a doc writes (e.g. joblib
    # artifacts) land in the temp dir, not the repo.
    namespace = {"__name__": "__doc_example__"}
    prev_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        exec(compile(code, str(path), "exec"), namespace)
    finally:
        os.chdir(prev_cwd)
