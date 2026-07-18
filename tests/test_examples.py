"""Smoke tests — every script in ``examples/`` must run end to end.

Keeps the examples honest: if the public API drifts, ``main()`` breaks here
before it breaks for a user copy-pasting from the docs.
"""

import importlib.util
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_SCRIPTS = sorted(p.stem for p in EXAMPLES_DIR.glob("*.py"))


@pytest.mark.parametrize("name", EXAMPLE_SCRIPTS)
def test_example_runs(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()  # must complete without raising
