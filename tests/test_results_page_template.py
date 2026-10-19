"""Every model page on the Results site follows one template.

In order: the question as the reader would ask it; the bold bottom line, one
of five fixed phrases and the only bold on the page; results by dataset as
tabs, sample data last; "What this means for your file"; "Try it on your own
donors" with a code block; then the collapsed "How we tested" and "Numbers
for analysts". Where a page answers a question on the index, its bottom line
is the one the index computes from results.json.
"""

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGES = REPO_ROOT / "docs" / "results"
RESULTS = json.loads((REPO_ROOT / "docs" / "assets" / "results" / "results.json").read_text())

# page -> its question's key in results.json "_index" (None: not on the index)
MODEL_PAGES = {
    "leadership.md": "upgrade",
    "response.md": "response",
    "lapse.md": "lapse",
    "ask.md": "ask",
    "planned_giving.md": "planned_giving",
    "who_to_mail.md": "who_to_mail",
    "intervals.md": None,
}
BOTTOM_LINES = (
    "Use the model", "Use the model for your top slice only", "Use the simple rule",
    "Use the retention list", "Can't tell yet",
)
SECTIONS = [
    "## What this means for your file",
    "## Try it on your own donors",
    '??? note "How we tested"',
    '??? note "Numbers for analysts"',
]


def _without_code(text):
    return re.sub(r"^\s*```.*?^\s*```", "", text, flags=re.S | re.M)


def _paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


@pytest.mark.parametrize("page", sorted(MODEL_PAGES))
def test_page_opens_with_question_then_bottom_line(page):
    paras = _paragraphs(_without_code((PAGES / page).read_text()))
    assert paras[0].startswith("# ") and "\n" not in paras[0], page
    has_question_line = "\n" not in paras[1] and paras[1].endswith("?")
    assert paras[0].endswith("?") or has_question_line, f"{page}: no one-line question in or under the title"
    bottom = paras[2] if has_question_line else paras[1]
    m = re.match(r"\*\*([^*]+?)\.?\*\*", bottom)
    assert m, f"{page}: the bottom line must come straight after the question, in bold"
    phrase = m.group(1)
    assert any(phrase == p or phrase.startswith(p + " (") for p in BOTTOM_LINES), (page, phrase)
    key = MODEL_PAGES[page]
    if key is not None:
        assert phrase == RESULTS["_index"][key]["bottom_line"], (page, phrase)


@pytest.mark.parametrize("page", sorted(MODEL_PAGES))
def test_bottom_line_is_the_only_bold(page):
    text = _without_code((PAGES / page).read_text())
    assert len(re.findall(r"\*\*[^*]+\*\*", text)) == 1, page


@pytest.mark.parametrize("page", sorted(MODEL_PAGES))
def test_sections_in_template_order(page):
    text = (PAGES / page).read_text()
    top = [line for line in _without_code(text).splitlines() if re.match(r"(#{2,} |=== |\?\?\? )", line)]
    tabs = [line for line in top if line.startswith("=== ")]
    assert top == tabs + SECTIONS, f"{page}: top-level sections are {top}"
    sample = [i for i, t in enumerate(tabs) if t.startswith('=== "Sample data')]
    assert sample == list(range(len(tabs) - len(sample), len(tabs))), f"{page}: sample data must be the last tab"
    try_it = text.split("## Try it on your own donors", 1)[1].split('??? note "How we tested"', 1)[0]
    assert "```python" in try_it, f"{page}: no code block under Try it on your own donors"
