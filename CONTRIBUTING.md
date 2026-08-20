# Contributing to PhilanthroPy

Thanks for helping improve PhilanthroPy! This guide covers the local checks
every change must pass before it reaches CI. By participating you agree to abide
by our [Code of Conduct](CODE_OF_CONDUCT.md).

New here? Start with a
[good first issue](https://github.com/PhilanthroPy-Project/PhilanthroPy/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
each one names the files to touch and the single command that proves it is
done. A first PR does not need to be big; docs fixes and missing tests are
genuinely wanted. Comment on the issue to claim it, and ask there if anything in
this guide does not work; a question is a valid contribution too.

## Setup

Fork the repository on GitHub first; you will not have push access to
`PhilanthroPy-Project/PhilanthroPy`.

```bash
git clone https://github.com/<your-username>/PhilanthroPy.git
cd PhilanthroPy
git remote add upstream https://github.com/PhilanthroPy-Project/PhilanthroPy.git
git switch -c my-change         # never work on main
pip install -e ".[dev]"          # editable install so the working tree is what's tested
sh scripts/install_hooks.sh      # pre-push hook: runs the suite before every push
```

Install editable: a non-editable copy in site-packages will shadow your edits
under pytest and silently run stale code.

Expect the install plus a first green `make ci` to take about eight minutes.

## Before pushing any commit

Always run the full local gate first:

```bash
make ci
```

This runs, in order:
1. Lint (flake8, real defects only)
2. Type check (mypy)
3. Collection check, catches missing imports immediately
4. Docstring examples (`--doctest-modules`)
5. Full test suite
6. Coverage gate (branch coverage ≥ 92% overall)

If `make ci` passes, the lint, type, test, and overall-coverage jobs will pass
CI. CI checks four more things `make ci` does not:

- a **risk-tier coverage floor** over `preprocessing/`, `models/`, `ingest/`,
  `cli.py`, and `utils/_persistence.py`; reproduce it locally with `make riskcov`
  (run it after `make ci`, which produces the coverage data). The include list and
  the floor are defined once in the `Makefile`; CI runs the same target, so the two
  cannot disagree.
- the declared **dependency floors** on Python 3.9 (`uv pip install --resolution lowest-direct`)
- `python -m build` plus `twine check` on the built distributions
- a **minimal install** (`pip install .`) that must import without matplotlib

Never use `git push --no-verify`.

When adding a new test file that imports a new class:
- Implement the class FIRST
- Add the export to `__init__.py` FIRST
- Verify: `python -c "from philanthropy.X import Y; print('OK')"`
- THEN write the test file
- THEN run `make ci`
- THEN git add + commit

A single test file must never assert contradictory shapes or column counts for
the same transformer. Before committing a test file, run:

```bash
grep -n "shape\|columns\|n_by" tests/<file>.py
```

and confirm all shape assertions are consistent with each other.

## Opening the pull request

```bash
git push origin my-change        # your fork, not upstream
```

Then open a pull request against `PhilanthroPy-Project/PhilanthroPy` `main`.
GitHub offers the button on your fork right after the push. Fill in the
[PR template](.github/PULL_REQUEST_TEMPLATE.md): what changed and why, and add a
`CHANGELOG.md` entry under `[Unreleased]`. Add yourself to
[CONTRIBUTORS.md](CONTRIBUTORS.md) in the same PR: one line, name or handle and
what you did.

A body of just `Resolves #N` is not enough; say what changed and why even
when an issue is linked; reviewers read the PR body first, not the issue.

CI runs on pull requests from forks. If a job fails, push another commit to the
same branch; the PR updates itself.

## Additional checks

After cloning, run `sh scripts/install_hooks.sh` to install the pre-push hook.
This runs the full test suite before every push, preventing collection errors
from reaching CI.

Before committing a new test file, always verify:

```bash
python -m pytest <new_test_file.py> --collect-only -q
# Must show: X tests collected, 0 errors
```

## Versioning & deprecation

PhilanthroPy follows [Semantic Versioning](https://semver.org). Breaking changes
are called out under a **Breaking** heading in [CHANGELOG.md](CHANGELOG.md), and
a deprecated public API emits a `DeprecationWarning` for at least one minor
release before removal. Per-symbol stability tiers are in
[docs/reference/index.md](docs/reference/index.md).

Cutting a release is a maintainer task and needs PyPI, Zenodo, and GitHub
release permissions; the runbook lives in [RELEASING.md](RELEASING.md).
