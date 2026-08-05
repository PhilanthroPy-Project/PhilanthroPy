"""Print every `path:line` cited in open good-first-issue bodies with the real line.

Usage: python scripts/check_issue_lines.py
Eyeball the output before labelling or re-surfacing an issue: a shifted line
number sends a contributor (or their agent) at the wrong symbol.
"""

import json
import pathlib
import re
import subprocess

EXT = r"(?:py|md|yml|toml|cff|in)"
# inline `path.py:123`, and markdown table rows `| path.py | 123, 456 |`
PAT = re.compile(rf"([\w/]+\.{EXT})[:\s]*[`(]?:?(\d+)")
ROW = re.compile(rf"\|\s*`?([\w/]+\.{EXT})`?\s*\|\s*([\d,\s]+?)\s*\|")


def cited(body):
    for path, num in PAT.findall(body):
        yield path, num
    for path, nums in ROW.findall(body):
        for num in nums.split(","):
            yield path, num.strip()

issues = json.loads(
    subprocess.run(
        ["gh", "issue", "list", "--label", "good first issue",
         "--state", "open", "--json", "number,body"],
        capture_output=True, text=True, check=True,
    ).stdout
)

for it in issues:
    print(f"--- #{it['number']}")
    for path, num in dict.fromkeys(cited(it["body"] or "")):
        p = pathlib.Path(path)
        if not p.is_file():
            continue
        lines = p.read_text().splitlines()
        n = int(num)
        text = lines[n - 1].strip() if 0 < n <= len(lines) else "<out of range>"
        print(f"  {path}:{n}  {text[:80]}")
