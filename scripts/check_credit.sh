#!/usr/bin/env bash
# Credit guard (issue #113): a PR that changes code under philanthropy/ must
# also touch CHANGELOG.md, and its author must already be credited in
# CONTRIBUTORS.md. Runs on pull_request events only; direct pushes to main
# and history that predates the guard are not checked.
set -euo pipefail

base=$1
head=$2
author=${3:-}

changed=$(git diff --name-only "$base" "$head")

if ! echo "$changed" | grep -q '^philanthropy/'; then
  echo "credit guard: no source changes under philanthropy/ - nothing to check"
  exit 0
fi

if ! echo "$changed" | grep -qx 'CHANGELOG.md'; then
  echo "::error::This PR touches files under philanthropy/ but not CHANGELOG.md. Add a line under [Unreleased] describing the change."
  exit 1
fi

if [ -z "$author" ]; then
  echo "::error::PR author could not be determined; cannot run the CONTRIBUTORS.md check."
  exit 1
fi

if ! grep -qE "@${author}([^A-Za-z0-9_-]|$)" CONTRIBUTORS.md; then
  echo "::error::This PR touches files under philanthropy/ but its author (@$author) is not listed in CONTRIBUTORS.md. Add yourself in the same pull request (see 'Getting listed')."
  exit 1
fi

echo "credit guard OK: changelog touched and @$author is credited"
