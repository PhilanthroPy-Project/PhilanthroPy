import os

# CHANGELOG
p = "CHANGELOG.md"
data = open(p, "rb").read()


def crlf(s: str) -> bytes:
    return s.replace("\n", "\r\n").encode("utf-8")


old = """### Added
- Test covering the CLI's `_score_array` fallback to `predict_proba(X)[:, 1]`
"""
new = """### Added
- `philanthropy check` and `philanthropy train --task leadership` accept all
  seven `read_gifts` presets (adding `bloomerang`, `donorperfect` and
  `blackbaud_crm`), and read each one through the same row filter `features`
  uses (closes #341).
- Test covering the CLI's `_score_array` fallback to `predict_proba(X)[:, 1]`
"""
assert crlf(old) in data, "CHANGELOG anchor not found"
data = data.replace(crlf(old), crlf(new), 1)
open(p, "wb").write(data)
print("CHANGELOG updated")

# CONTRIBUTORS
p = "CONTRIBUTORS.md"
data = open(p, "rb").read()
anchor = "## Getting listed"
entry = (
    "- [@Sudhakar7J](https://github.com/Sudhakar7J): extended "
    "`philanthropy check` and `philanthropy train --task leadership` to every "
    "`read_gifts` source, reading each through the same row filter `features` "
    "uses (closes #341).\n\n"
)
assert crlf(anchor) in data, "CONTRIBUTORS anchor not found"
data = data.replace(crlf(anchor), crlf(entry + anchor), 1)
open(p, "wb").write(data)
print("CONTRIBUTORS updated")
