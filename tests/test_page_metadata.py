"""What the published page says about itself, outside the arena.

Two things live here. The page carries four authorship marks: a
`<meta name="author">`, an HTML comment at the top of the source, one
`console.info` line, and the footer credit. And its `<head>` carries the
search and social metadata: title, description, canonical URL, Open Graph
and Twitter cards, the Search Console token, a JSON-LD graph and a favicon.

Both are written in `site/template.html`, never in the built file, so every
build carries them. These tests hold them in three places: the template, a
fresh build, and the committed `docs/index.html` that GitHub Pages serves —
plus the files the metadata points at beside it in `docs/`.
"""

from __future__ import annotations

import json
import re
import struct
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

import pytest

from redlight import export

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "site" / "template.html"
PUBLISHED = REPO_ROOT / "docs" / "index.html"

AUTHOR_META = re.compile(r'<meta\s+name="author"\s+content="Safdar Hussain"\s*/?>')
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
    assert len(AUTHOR_META.findall(page)) == 1


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


# --------------------------------------------------------------------------
# Search and social metadata
# --------------------------------------------------------------------------
#
# The block in <head> that decides how the page looks in a search result and
# in a shared link: the title and description Google shows, the Open Graph
# and Twitter cards, the canonical URL, the Search Console token, the JSON-LD
# graph, and the favicon. It is written in the template and has to arrive
# intact in the page GitHub Pages serves, next to the three files it points
# at: `og-image.png`, `favicon.svg` and `sitemap.xml`.

SITE_URL = "https://safdar-hussain1.github.io/red-light-green-light-cv/"
OG_IMAGE_URL = SITE_URL + "og-image.png"
SEARCH_CONSOLE_TOKEN = "0SIEfExLTSQj1qvnHWF5A5fY58KVl2lpIEnePP9CtI0"
DOCS = REPO_ROOT / "docs"

AUTHOR = {
    "@type": "Person",
    "name": "Safdar Hussain",
    "url": "https://github.com/safdar-hussain1",
    "sameAs": [
        "https://github.com/safdar-hussain1",
        "https://www.linkedin.com/in/safdar-hussain-a8a61b248",
    ],
}


class _Head(HTMLParser):
    """Collects what a crawler reads: the head's title, meta, links, JSON-LD.

    The title is the first `<title>` inside `<head>` only — the inline SVGs in
    the body carry `<title>` elements of their own.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = None
        self.meta: dict[str, list[str]] = {}
        self.links: list[dict] = []
        self.json_ld: list[str] = []
        self.h1 = 0
        self._in_head = self._in_title = self._in_ld = False
        self._buf = ""

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "head":
            self._in_head = True
        elif tag == "title" and self._in_head and self.title is None:
            self._in_title, self._buf = True, ""
        elif tag == "meta" and (a.get("name") or a.get("property")):
            key = (a.get("name") or a.get("property")).lower()
            self.meta.setdefault(key, []).append(a.get("content") or "")
        elif tag == "link":
            self.links.append(a)
        elif tag == "script" and a.get("type") == "application/ld+json":
            self._in_ld, self._buf = True, ""
        elif tag == "h1":
            self.h1 += 1

    def handle_endtag(self, tag):
        if tag == "head":
            self._in_head = False
        elif tag == "title" and self._in_title:
            self._in_title, self.title = False, " ".join(self._buf.split())
        elif tag == "script" and self._in_ld:
            self._in_ld = False
            self.json_ld.append(self._buf)

    def handle_data(self, data):
        if self._in_title or self._in_ld:
            self._buf += data

    def one(self, key: str) -> str:
        values = self.meta.get(key, [])
        assert len(values) == 1, f"expected exactly one {key!r} meta tag, found {len(values)}"
        return values[0].strip()


@pytest.fixture(scope="module")
def head(page) -> _Head:
    parser = _Head()
    parser.feed(page)
    return parser


def test_title_fits_in_a_search_result(head):
    assert head.title, "no <title> in <head>"
    assert len(head.title) <= 60, f"title is {len(head.title)} characters: {head.title!r}"


def test_description_is_120_to_160_characters(head):
    description = head.one("description")
    assert 120 <= len(description) <= 160, f"description is {len(description)} characters"


def test_social_cards_repeat_the_title_and_description(head):
    assert head.one("og:title") == head.title
    assert head.one("twitter:title") == head.title
    assert head.one("og:description") == head.one("description")
    assert head.one("twitter:description") == head.one("description")


def test_urls_point_at_the_live_page(head):
    canonical = [link.get("href") for link in head.links if link.get("rel") == "canonical"]
    assert canonical == [SITE_URL]
    assert head.one("og:url") == SITE_URL
    assert head.one("og:image") == OG_IMAGE_URL
    assert head.one("twitter:image") == OG_IMAGE_URL


def test_card_types_and_image_metadata(head):
    assert head.one("og:type") == "website"
    assert head.one("og:site_name") == "Safdar Hussain"
    assert head.one("twitter:card") == "summary_large_image"
    assert head.one("og:image:width") == "1200"
    assert head.one("og:image:height") == "630"
    alt = head.one("og:image:alt")
    assert len(alt) > 40, "the image alt text should say what the picture shows"
    assert head.one("twitter:image:alt") == alt


def test_author_and_search_console_meta(head):
    assert head.one("author") == "Safdar Hussain"
    assert head.one("google-site-verification") == SEARCH_CONSOLE_TOKEN


def test_favicon_link_resolves_to_a_committed_svg(head):
    icons = [link for link in head.links if "icon" in (link.get("rel") or "").split()]
    assert len(icons) == 1, f"expected one icon link, found {icons}"
    href = icons[0].get("href")
    assert not re.match(r"https?://|//", href), "the favicon must be served next to the page"
    favicon = DOCS / href
    assert favicon.is_file(), f"docs/{href} is not committed"
    assert favicon.read_text(encoding="utf-8").lstrip().startswith("<svg")
    assert icons[0].get("type") == "image/svg+xml"


def test_json_ld_is_a_graph_of_the_game_and_its_source(head):
    assert len(head.json_ld) == 1, "expected exactly one JSON-LD block"
    graph = json.loads(head.json_ld[0])
    assert graph["@context"] == "https://schema.org"
    nodes = {node["@type"]: node for node in graph["@graph"]}
    assert set(nodes) == {"VideoGame", "SoftwareSourceCode"}

    game = nodes["VideoGame"]
    assert game["gamePlatform"] == "Web browser"
    assert game["url"] == SITE_URL
    assert game["image"] == OG_IMAGE_URL
    assert game["description"] == head.one("description")
    assert game["isAccessibleForFree"] is True

    source = nodes["SoftwareSourceCode"]
    assert source["codeRepository"] == "https://github.com/safdar-hussain1/red-light-green-light-cv"
    assert source["license"] == "https://opensource.org/licenses/MIT"

    for node in nodes.values():
        assert node["name"] == "Red Light, Green Light"
        assert node["author"] == AUTHOR, f"{node['@type']} does not carry the full author"


def test_page_has_exactly_one_h1(head):
    assert head.h1 == 1


def test_og_image_is_a_1200_by_630_png_under_500_kb():
    data = (DOCS / "og-image.png").read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "og-image.png is not a PNG"
    assert struct.unpack(">II", data[16:24]) == (1200, 630)
    assert len(data) < 500 * 1024, f"og-image.png is {len(data) // 1024} KB"


def test_sitemap_lists_the_page_with_a_literal_lastmod():
    text = (DOCS / "sitemap.xml").read_text(encoding="utf-8")
    root = ElementTree.fromstring(text)
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    assert [loc.text for loc in root.findall("s:url/s:loc", ns)] == [SITE_URL]
    lastmod = root.findall("s:url/s:lastmod", ns)
    assert len(lastmod) == 1
    # A literal date, written when the content changed. Stamping today() at
    # build time would make every build produce a different file.
    date.fromisoformat(lastmod[0].text)


def test_no_per_project_robots_txt():
    """Crawlers only read robots.txt at the host root, so one here does nothing."""
    assert not (DOCS / "robots.txt").exists()
