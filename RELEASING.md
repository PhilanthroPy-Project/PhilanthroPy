# Releasing PhilanthroPy

Maintainer runbook. Cutting a release needs PyPI, Zenodo, and GitHub release
permissions; contributors do not need anything on this page. For contributing,
see [CONTRIBUTING.md](CONTRIBUTING.md).

## Versioning & deprecation

PhilanthroPy follows [Semantic Versioning](https://semver.org). While the
project is pre-1.0, minor releases (`0.x.0`) may contain breaking changes; these
are always called out under a **Breaking** heading in
[CHANGELOG.md](CHANGELOG.md). Where feasible, a deprecated public API is kept for
at least one minor release and emits a `DeprecationWarning` pointing at its
replacement before removal. Supported versions are listed in
[SECURITY.md](SECURITY.md).

Nothing is deprecated at 0.7.0. When you next need to deprecate something,
reintroduce the one mechanism 0.6.0 used: a `deprecated_alias(new_name,
removed_in=...)` decorator in `philanthropy/utils/_deprecation.py` for a renamed
method, and an inline `warnings.warn(..., DeprecationWarning)` in `fit`/`split`
for a parameter that no longer does anything, plus a `tests/test_deprecations.py`
whose registry meta-test fails when a shim ships untested. Per-symbol stability
tiers live in [docs/reference/index.md](docs/reference/index.md).

## RELEASE CHECKLIST

Run in order. Steps 1–6 are the gate `publish.yml` enforces; 7–9 are manual.

1. `make ci` is green, and so is `pytest tests/test_public_api_contract.py -q`.
   Also skim the `## [Unreleased]` block in `CHANGELOG.md`. `.gitattributes`
   sets `merge=union` on that file so concurrent PRs stop conflicting on it, and
   the cost is that git will never again flag a problem in this file. Two things
   to look for: a bullet under the wrong heading (union is line-based, not
   section-aware), and a duplicated or contradictory bullet (if two branches
   edited the same entry, union silently keeps both). Cheap to eyeball once per
   release, annoying to find later.
2. Bump `version` in `pyproject.toml`. Nothing else carries the version:
   `philanthropy.__version__` reads it from installed metadata.
3. Add a `## [X.Y.Z] - YYYY-MM-DD` section to `CHANGELOG.md`. The date is
   required: `publish.yml` rejects a heading still carrying `- TBD`, which is
   what a release staged ahead of its window looks like. A release that removes anything
   needs a **Breaking** heading; a release that adds a shim needs a
   **Deprecated** heading naming every alias and dead parameter with the version
   that removes it.
4. Update the "Deprecations" section of `docs/reference/index.md`.
5. `python -m build && python -m twine check --strict dist/*`.
6. Tag `vX.Y.Z` and push it. Publishing is gated on clicking **Publish release**
   in the GitHub UI; `publish.yml` then re-checks that the tag, `pyproject.toml`
   and `CHANGELOG.md` agree before it builds.
   - **Paste the CHANGELOG section into the release body.** An empty release body
     wastes the only page most people ever read about a version.
   - **Add a `### Thanks` line naming every external contributor** in that
     section. A named credit on a permanent release page is worth more to a
     drive-by contributor than a line in a markdown file.
   - Update the supported-versions table in `SECURITY.md` so it names a version
     that can actually be installed.
   - Re-run `examples/quickstart.ipynb` if the public API moved: it installs from
     PyPI (`!pip install philanthropy -q`) while living on `main`, so it is the
     one artifact that can silently break on a rename.
7. Confirm the release landed on PyPI and that `pip install philanthropy==X.Y.Z`
   works in a clean venv.
8. **Deposit to Zenodo.** The GitHub–Zenodo integration picks up the published
   release and reads `.zenodo.json`. JOSS requires a deposited archive with a DOI
   at acceptance.
9. On the first deposit only: copy the Zenodo **concept** DOI (the one that
   resolves to the latest version, not the per-version DOI) into the
   `identifiers:` stanza of `CITATION.cff`, replacing
   `10.5281/zenodo.PENDING`, and commit.

A shim added in `X.Y.0` may only be removed in `X.(Y+1).0` **after `X.Y.0` is
published on PyPI**: one full published minor of overlap, not one commit.

### Cutting a release `main` has already moved past

`main` can carry more than one unreleased version; 0.7.0 and 1.0.0 are both
merged and both still `- TBD`. The gate compares the tag against the
`pyproject.toml` **of the commit the tag points at**, so once a later version
has bumped that file at the tip, the older release can no longer be cut from the
tip: `v0.7.0` against a `main` reading `version = "1.0.0"` fails with
`tag != pyproject 1.0.0`, dated changelog or not. Step 2 above assumes one
version in flight; with several staged, tag the last commit that still reads the
older version, `e1713c4` for 0.7.0:

```bash
git switch -c release/0.7.0 e1713c4
# step 3: date the '## [0.7.0]' heading in CHANGELOG.md
git commit -am "chore: date the 0.7.0 release"
git push origin release/0.7.0 && git tag v0.7.0 && git push origin v0.7.0
```

The branch is not a fork of the release pipeline. A `release` event only fires
for a workflow file that "exists on the default branch", so `publish.yml` is
always the one on `main`; `actions/checkout` with no `ref` "defaults to the
reference or SHA for that event", which is the tagged commit. The old tree gets
the current gate and the current action pins.

Then date the same heading on `main`, so its changelog stops claiming `- TBD`
for something that shipped.
