"""What the published page says about itself, outside the arena.

The page carries four authorship marks: a `<meta name="author">`, an HTML
comment at the top of the source, one `console.info` line, and the footer
credit. They live in `site/template.html`, never in the built file, so every
build carries them. These tests hold them in three places: the template, a
fresh build, and the committed `docs/index.html` that GitHub Pages serves.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from redlight import export

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "site" / "template.html"
PUBLISHED = REPO_ROOT / "docs" / "index.html"

AUTHOR_META = '<meta name="author" content="Safdar Hussain"'
SOURCE_COMMENT = (
    "<!-- Red Light, Green Light · built by Safdar Hussain · "
    "https://github.com/safdar-hussain1 -->"
)
CONSOLE_LINE = (
    'console.info("Red Light, Green Light — built by Safdar Hussain · '
    'https://github.com/safdar-hussain1/red-light-green-light-cv")'
)
FOOTER_LINK = '<a href="https://github.com/safdar-hussain1">Safdar Hussain</a>'


@pytest.fixture(scope="module")
def fresh_build(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("page") / "index.html"
    export.build_site(str(out))
    return out.read_text(encoding="utf-8")


@pytest.fixture(scope="module", params=("template", "fresh build", "docs/index.html"))
def page(request, fresh_build) -> str:
    if request.param == "template":
        return TEMPLATE.read_text(encoding="utf-8")
    if request.param == "fresh build":
        return fresh_build
    return PUBLISHED.read_text(encoding="utf-8")


def test_author_meta_is_present(page):
    assert page.count(AUTHOR_META) == 1


def test_source_comment_is_present(page):
    assert page.count(SOURCE_COMMENT) == 1


def test_source_comment_sits_at_the_top_of_the_file(page):
    """Anyone who opens the page source sees it first, not 800 KB later."""
    assert page.index(SOURCE_COMMENT) < page.index("<head>")


def test_console_signature_is_present_once(page):
    assert page.count(CONSOLE_LINE) == 1


def test_console_signature_runs_as_a_plain_script(page):
    """The line has to be inside an ordinary `<script>` block to run at all."""
    blocks = re.findall(r"<script>(.*?)</script>", page, re.S)
    assert any(CONSOLE_LINE in block for block in blocks)


def test_footer_credits_the_author(page):
    footer = re.search(r"<footer>(.*?)</footer>", page, re.S)
    assert footer, "the page has no footer"
    text = footer.group(1)
    assert FOOTER_LINK in text
    assert "Built by" in text
