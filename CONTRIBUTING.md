# Contributing to PhilanthroPy

Thanks for helping improve PhilanthroPy! This guide covers the local checks
every change must pass before it reaches CI. By participating you agree to abide
by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Setup

```bash
git clone https://github.com/PhilanthroPy-Project/PhilanthroPy.git
cd PhilanthroPy
pip install -e ".[dev]"          # editable install so the working tree is what's tested
sh scripts/install_hooks.sh      # pre-push hook: runs the suite before every push
```

Install editable — a non-editable copy in site-packages will shadow your edits
under pytest and silently run stale code.

## Before pushing any commit

Always run the full local gate first:

```bash
make ci
```

This runs in the exact same order as GitHub Actions:
1. Collection check — catches missing imports immediately
2. Full test suite
3. Coverage gate (≥ 85%)

If `make ci` passes, your push will pass CI.
Never use `git push --no-verify`.

When adding a new test file that imports a new class:
- Implement the class FIRST
- Add the export to `__init__.py` FIRST
- Verify: `python -c "from philanthropy.X import Y; print('OK')"`
- THEN write the test file
- THEN run `make ci`
- THEN git add + commit + push

A single test file must never assert contradictory shapes or column counts for
the same transformer. Before committing a test file, run:

```bash
grep -n "shape\|columns\|n_by" tests/<file>.py
```

and confirm all shape assertions are consistent with each other.

## Additional checks

After cloning, run `sh scripts/install_hooks.sh` to install the pre-push hook.
This runs the full test suite before every push, preventing collection errors
from reaching CI.

Before committing a new test file, always verify:

```bash
python -m pytest <new_test_file.py> --collect-only -q
# Must show: X tests collected, 0 errors
```

Use `git push --no-verify` only in an emergency to bypass the hook.
