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

### Claiming an issue

Comment on the issue before you open a pull request, and wait for a maintainer
to assign it to you; we assign within 24 hours. This is not bureaucracy:
issues #175 and #176 were two people solving the same four-line problem on the
same evening, and one of them wasted their time. If an issue already has an
assignee, pick another one; the `good first issue` label always has open ones.

## Setup

Fork the repository on GitHub first; you will not have push access to
`PhilanthroPy-Project/PhilanthroPy`.

```bash
git clone https://github.com/<your-username>/PhilanthroPy.git
cd PhilanthroPy
git remote add upstream https://github.com/PhilanthroPy-Project/PhilanthroPy.git
git switch -c my-change         # never work on main
pip install -e ".[dev]"          # editable install so the working tree is what's tested
sh scripts/install_hooks.sh      # pre-push hook: collection check + flake8, a few seconds
```
> **Windows contributors:** `make` is not available in PowerShell by default,
> so `make ci` and `make riskcov` below will not run as written. See
> [Testing on Windows](#testing-on-windows) for the equivalent commands.
> `sh scripts/install_hooks.sh` also requires Git Bash; the installed
> pre-push hook does **not** fire when pushing from plain PowerShell, only
> from a POSIX-compatible shell (confirmed by testing).

Install editable: a non-editable copy in site-packages will shadow your edits
under pytest and silently run stale code.

Expect the install plus a first green `make ci` to take about eight minutes.

## Before opening a pull request

Run the full local gate once before you open the PR (or mark it ready for
review). While iterating, running the tests for the files you touched is
enough; CI runs the full suite on every push to the PR.

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
- the declared **dependency floors** on Python 3.10 (`uv pip install --resolution lowest-direct`)
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

## Adding a CRM ingest reader

A reader PR gives one CRM's gift export its own bridge module, registers it so
`read_gifts(..., source="<crm>")` finds it, and tests it. Work the steps below
in order; each one is something a reviewer asked for on #312, #314 or #321.
Copy the shape from `philanthropy/ingest/_raisers_edge.py` and
`tests/test_raisers_edge.py` rather than inventing a new one.

### 1. Cite the vendor's own docs first

Collect the vendor's own documentation for the export: the field names, and
every type/status term the filter matches (`Gift Type` values, contribution
statuses, opportunity stages, revenue types). The module docstring ends with a
`Sources:` block listing each document with a link, and every field name and
every type/status term in the file must come from one of them. The PR body
cites the same sources. #314 stayed blocked until its vocabulary was traced
back to a Neon Support Center article: a `DEFAULT_` set that is not in the
cited vocabulary produces a wrong total, not a wording preference.

### 2. Build the bridge module

All in `philanthropy/ingest/_<crm>.py`:

- **Header aliases.** A `_HEADER_ALIASES` dict mapping the export's own labels
  onto the canonical `contact_id`, `receive_date` and `total_amount` names,
  plus the type/status column (e.g. `gift_type`), applied by a local
  `_canonical_<crm>()` that falls through to `_canonical` from `._civicrm`.
  Cover every spelling the vendor documents for the same field: the desktop
  export, the database columns and the API rarely agree.
- **A `DEFAULT_...` constant.** `DEFAULT_EXCLUDED_...` or
  `DEFAULT_INCLUDED_...`, a tuple documented with `#:` comments and cited to
  the vendor docs, used as the default of the aggregator's filter parameter.
  `None` (or an empty sequence) disables the filter, so the parameter
  docstring says that explicitly.
- **The private row filter.** `_<crm>_received_rows(...)` normalises headers,
  raises `KeyError` naming the fields the export is missing, and applies the
  filter. When the filter was requested but the export has no type/status
  column it calls `warnings.warn(..., stacklevel=3)` instead of silently
  summing commitments with the payments made against them.
- **The typed empty frame.** `<crm>_..._to_features` returns
  `_empty_feature_frame()` from `._civicrm` when nothing survives the filter,
  so an all-pledge input yields the same columns and dtypes as a normal frame
  instead of an untyped empty one.
- **Delegation.** The aggregator ends with
  `civicrm_contributions_to_features(df, reference_date=..., statuses=None)` —
  `statuses=None` because this bridge already dropped everything that is not
  money received; do not reimplement the roll-up. The loader
  `read_<crm>_...` delegates to `read_civicrm_contributions` for the same
  reason (BOM handling and path hardening are already solved there).

### 3. Register it

- `philanthropy/ingest/_read_gifts.py`: import the reader and aggregator
  (and the private `_<crm>_received_rows`), add `"<crm>"` to `GIFT_SOURCES`,
  add the `(reader, aggregator)` pair to `_REGISTRY`, add the
  `(reader, row filter)` pair to `_ROW_FILTERS`, add the source's filter kwarg
  to the `read_gifts` docstring's `**kwargs` list, and name the bridge in the
  module docstring's list. `GIFT_SOURCES` is compared against a hardcoded set
  in `tests/test_read_gifts.py` and against `_ROW_FILTERS` in
  `tests/test_read_gift_rows.py`, so a half-registered source fails tests
  rather than review.
- `philanthropy/ingest/__init__.py`: the import in module order, and the
  `__all__` entry in its alphabetical place (the `DEFAULT_`/`GIFT_` capitals
  sort ahead of the lowercase names — do not append at the end).
- `docs/reference/index.md`: one Tier 2 row with the three public symbols, the
  `ingest` module, and why it is not Tier 1.

### 4. Wire the CLI

`check` and `train --task leadership` read a raw export through
`_read_raw_gifts` in `philanthropy/cli.py`, which calls
`ingest.read_gift_rows`, which looks the source up in `_ROW_FILTERS`. So:

- The private `_<crm>_received_rows` helper from step 2 is the row filter
  `_<crm>_..._to_features` calls: one filter, so the CLI and `features` count
  exactly the same gifts.
- Register `(read_<crm>_..., _<crm>_received_rows)` in `_ROW_FILTERS` in
  `philanthropy/ingest/_read_gifts.py` — that table is the CLI's raw-gift
  reader table (#341) and it is what `check` and `train --task leadership`
  use.
- `features --source` needs nothing: its choices are `GIFT_SOURCES` (#338
  has landed).
- Add one case to `_RAW_GIFT_EXPORTS` in `tests/test_cli.py`: a row the source
  counts and one it drops (a pledge, an open stage, an unpaid schedule), so
  `test_raw_gift_log_drops_the_rows_features_drops` proves the raw gift log
  drops what `features` drops.

### 5. Test it

- `tests/test_<crm>.py`, one file per CRM, following
  `tests/test_raisers_edge.py`: the default filter, `None` and an empty
  sequence disabling it, the missing-type-column warning, every documented
  alias, the typed empty frame, the missing-field `KeyError`, and the
  `read_<crm>_...` single-file and directory cases.
- Two `read_gifts` tests in `tests/test_read_gifts.py`:
  `test_<crm>_path_matches_direct_call` (a path through the registry gives
  the same frame as the direct reader + aggregator call) and
  `test_<crm>_kwarg_reaches_the_underlying_aggregator` (the filter kwarg is
  not swallowed). Import the new names in the alphabetical import block.
- The hardcoded `GIFT_SOURCES` set in `test_gift_sources_lists_all_..._presets`
  in `tests/test_read_gifts.py`: add `"<crm>"` and bump the count in the test
  name.
- The `_EXPORTS` entry in `tests/test_read_gift_rows.py`; without it
  `test_every_gift_source_is_covered` fails, because it compares
  `_ROW_FILTERS`, `GIFT_SOURCES` and `_EXPORTS` for equality.

Before committing, run:

```bash
python -m pytest tests/test_<crm>.py tests/test_read_gifts.py tests/test_read_gift_rows.py tests/test_cli.py -q
```

### 6. Add exactly one CHANGELOG bullet

One bullet under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md), naming the
new public symbols and the preset. Never run an editor formatter over
`CHANGELOG.md`: #312 was held up by a formatter rewriting about 100 unrelated
lines. Check the diff before you push:

```bash
git diff upstream/main -- CHANGELOG.md
```

It must show your one bullet and nothing else.

### 7. Build the docs

`docs/reference/index.md` is part of the site, so run the same command CI
runs:

```bash
mkdocs build --strict
```

It must pass. Then run `make ci` (or the [Windows
commands](#testing-on-windows)) as for any other change.

If `main` gains another reader while yours is in review, merge `main` and add
your source alongside theirs. Taking your own side of `GIFT_SOURCES`,
`_REGISTRY`, the `read_gifts` docstring, `ingest/__init__.py` and
`tests/test_read_gifts.py` drops the reader that landed first — that is the
round-trip #312, #314 and #321 each needed.

## Additional checks

After cloning, run `sh scripts/install_hooks.sh` to install the pre-push hook.
It checks for test collection errors and runs flake8 before every push, which
takes a few seconds. The full suite runs in CI.

Before committing a new test file, always verify:

```bash
python -m pytest <new_test_file.py> --collect-only -q
# Must show: X tests collected, 0 errors
```
## Testing on Windows

`make` and `sh` are not available in PowerShell by default. Run the
equivalent commands directly instead of `make ci` / `make riskcov`:

```powershell
python -m flake8 philanthropy tests examples
python -m mypy philanthropy
python -m pytest philanthropy --doctest-modules -q --no-cov
python -m pytest tests/ --collect-only -q
python -m coverage run -m pytest tests/ -q
python -m coverage report
```

`coverage report` reads the overall floor from `[tool.coverage.report]` in
`pyproject.toml`, so the bare command enforces exactly what CI does. Do not
pass `--fail-under` here: the number is defined once and copying it into a
second place is how the two drift apart.

For the risk-tier coverage floor, see the `RISK_TIER`/`RISK_FLOOR` values
in the `Makefile` and run:

```powershell
python -m coverage report --include='<RISK_TIER from Makefile>' --fail-under=<RISK_FLOOR from Makefile>
```

If `pytest --cov=...` (via `pytest-cov`) raises
`ImportError: cannot load module more than once per process` on numpy
import, use `coverage run -m pytest` followed by `coverage report` instead.
Same result, different import path.

The `sh scripts/install_hooks.sh` pre-push hook requires Git Bash to
install, and does not fire when pushing from plain PowerShell. Run it via
Git Bash, or run `pytest tests/ --collect-only -q` and
`python -m flake8 philanthropy tests examples` manually before pushing.

## Versioning & deprecation

PhilanthroPy follows [Semantic Versioning](https://semver.org). Breaking changes
are called out under a **Breaking** heading in [CHANGELOG.md](CHANGELOG.md), and
a deprecated public API emits a `DeprecationWarning` for at least one minor
release before removal. Per-symbol stability tiers are in
[docs/reference/index.md](docs/reference/index.md).

Cutting a release is a maintainer task and needs PyPI, Zenodo, and GitHub
release permissions; the runbook lives in [RELEASING.md](RELEASING.md).
