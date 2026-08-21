"""tests/test_doc_examples.py

Executes the fenced ``python`` blocks in every documentation page (``docs/**``
recursively plus the top-level ``README.md``), so a broken public API in the
documentation fails CI instead of a user's copy-paste.

Each file's python blocks are concatenated in order and executed in a single
fresh namespace (later blocks may rely on earlier imports/variables). Files with
no python fence are not collected at all. A file can only be muted by adding an
explicit ``<!-- docs-notest -->`` marker *and* naming it in ``_NOTEST`` below;
``test_notest_allowlist_is_exact`` makes muting a page a visible diff.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = REPO_ROOT / "docs"

_PY_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_NOTEST_MARKER = "<!-- docs-notest -->"

DOC_FILES = sorted(DOCS_ROOT.rglob("*.md")) + [REPO_ROOT / "README.md"]

# Pages that cannot execute in CI. Every entry needs the marker in the file too.
_NOTEST: set = set()


def _doc_files():
    return [p for p in DOC_FILES if "```python" in p.read_text()]


def test_notest_allowlist_is_exact():
    marked = {p.name for p in DOC_FILES if _NOTEST_MARKER in p.read_text()}
    assert marked == _NOTEST


@pytest.mark.parametrize(
    "path", _doc_files(), ids=lambda p: f"{p.parent.name}/{p.name}"
)
def test_doc_python_blocks_execute(path, tmp_path):
    text = path.read_text()
    if _NOTEST_MARKER in text:
        pytest.skip("docs-notest")

    code = "\n\n".join(_PY_FENCE.findall(text))

    # Execute inside a throwaway cwd so any files a doc writes (e.g. joblib
    # artifacts) land in the temp dir, not the repo.
    namespace = {"__name__": "__doc_example__"}
    prev_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        exec(compile(code, str(path), "exec"), namespace)
    finally:
        os.chdir(prev_cwd)
