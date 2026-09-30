"""tests/test_results_docs_images.py

Every chart referenced from ``docs/results/*.md`` needs non-empty Markdown
alt text: it is what a screen reader announces and what search engines and
chat-shared screenshots fall back to when the image itself is not visible.
This is a repo convention, not something ``mkdocs build --strict`` checks,
so it needs its own test (E.13, D1: "Every chart has ... alt text").
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DOCS = REPO_ROOT / "docs" / "results"

# Matches a Markdown image: ![alt text](path)
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def _images_by_page():
    for path in sorted(RESULTS_DOCS.glob("*.md")):
        for alt, target in _IMAGE_RE.findall(path.read_text()):
            yield path.name, alt, target


IMAGES = list(_images_by_page())


def test_results_pages_reference_at_least_one_image():
    assert IMAGES, "expected docs/results/*.md to embed at least one chart"


@pytest.mark.parametrize(
    "page,alt,target", IMAGES, ids=[f"{p}:{t}" for p, _, t in IMAGES]
)
def test_results_image_has_alt_text(page, alt, target):
    assert alt.strip(), f"{page} embeds {target} with empty alt text"


@pytest.mark.parametrize(
    "page,alt,target", IMAGES, ids=[f"{p}:{t}" for p, _, t in IMAGES]
)
def test_results_image_file_exists(page, alt, target):
    resolved = (RESULTS_DOCS / target).resolve()
    assert resolved.is_file(), f"{page} references missing image {target}"
