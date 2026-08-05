# Issue draft template

Not in `.github/ISSUE_TEMPLATE/` on purpose — that directory is the **public** issue
chooser, and an outside bug reporter would see and pick this. These are maintainer
drafts for stocking the `good first issue` feed.

## Why the shape matters

Both external contributions to date (PRs #30 and #36) were agent fleets that landed
a working PR within ~2h of the label going on. They succeeded because the title named
an exact file and symbol. Two other issues shipped **wrong line numbers** and one
shipped an acceptance criterion that was impossible to satisfy — see the repair
history on #23 and #33.

So the rules are:

1. **Never freeze a line number or a test count into the body.** Line numbers drift
   the moment anyone edits the file above them. Use a `Locate with` column holding a
   `grep`. A grep cannot drift.
2. **End with a shell block that exits non-zero until the issue is fixed.** The
   contributor then self-verifies before opening a PR instead of guessing at intent.
3. **For code-shaped issues, name the test file *and* the test function names.** CI
   enforces two coverage floors; an agent that adds a branch without a matching test
   turns CI red and walks. Pre-solve coverage in the spec.
4. **Title must sell and scope the task alone.** CodeTriage's daily email renders
   only `repo#number` plus the title — the body never appears in the inbox.
5. Run `python scripts/check_issue_lines.py` before filing or re-labelling anything.

## Template

```markdown
### What's wrong

One or two sentences. Name the user-visible symptom, not the internal cause.

### Where

| File | Locate with |
|---|---|
| `philanthropy/preprocessing/_example.py` | `grep -n "def get_feature_names_out" philanthropy/preprocessing/_example.py` |

### What to change

1. Numbered, mechanical steps. No judgement calls — if a step needs a decision
   about correct behaviour, this is not a good first issue.
2. ...

### Tests to add or extend

- File: `tests/test_example.py`
- Add: `test_get_feature_names_out_returns_declared_names`
- (Code changes need a test or the coverage floor fails. See AGENTS.md.)

### Done when

```bash
# Exits 0 only once the issue is fixed. Run it before opening the PR.
python -c "from philanthropy.preprocessing import Example; assert Example.get_feature_names_out.__doc__"
make ci
make riskcov
```

### First time here?

Read [CONTRIBUTING.md](../../CONTRIBUTING.md) and [AGENTS.md](../../AGENTS.md).
Docs-only fix? Use GitHub's web editor and open the PR from there — no local setup
needed. Add a `## [Unreleased]` CHANGELOG entry and yourself to `CONTRIBUTORS.md` in
the same PR.
```

## Keeping the feed stocked

- **Never let the open labelled count drop below 5.** Check with
  `gh issue list --label 'good first issue' --state open --json number --jq 'length'`.
- Apply a label the same day you merge a PR. The search feed both contributors used
  orders by **label-application time**, not creation time — labelling an existing
  issue re-surfaces it.
- Target **2 code : 1 doc**. Five of seven were docs-only, so drive-bys never touched
  an estimator.
- Generate code issues from `pytest --cov-report=term-missing` output, never from
  imagination. Do not file issues against unreachable code — a contributor writes a
  passing test, watches the lines stay red, and gives up.
- **Do not fix a labelled issue yourself.** Every one you close is destroyed
  inventory on the only channel with a nonzero conversion rate.
