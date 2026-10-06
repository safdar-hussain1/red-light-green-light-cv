"""The arena has to survive being a single file on a static host.

`build_site` collapses the whole browser arena — markup, styles, the JS
referee, the benchmark numbers, the chant, the golden fixtures — into one
HTML file that can be dropped on GitHub Pages and opened with no build step
and no server. These tests hold that promise to its edges: nothing left
unresolved in the template, nothing fetched from a third party without a
hash pinning what may be fetched, and a file small enough to open on a
phone.

The determinism test is the one that keeps the repository honest. A build
that varies run to run makes every commit of `docs/index.html` a diff full
of noise, and hides the one line that actually changed.
"""

import base64
import json
import re
from pathlib import Path

import pytest

from redlight import export

REPO_ROOT = Path(__file__).resolve().parents[1]

MAX_SITE_BYTES = 2 * 1024 * 1024
"""Two megabytes: the arena still has to open on a phone on hotel wifi."""

# The one documented exception to the subresource-integrity rule. MediaPipe's
# pose model and its wasm runtime are resolved by the tasks-vision loader at
# run time, from URLs the loader derives itself, so there is no fixed set of
# bytes to hash at build time. It is loaded only after the visitor opts into
# the camera, and it is pinned to an exact version so the URL cannot drift.
SRI_EXEMPT_HOSTS = ("cdn.jsdelivr.net/npm/@mediapipe/tasks-vision",)


@pytest.fixture(scope="module")
def built_site(tmp_path_factory) -> Path:
    """Build the site once and let the whole module inspect it."""
    out = tmp_path_factory.mktemp("site") / "index.html"
    export.build_site(str(out))
    return out


@pytest.fixture(scope="module")
def html(built_site) -> str:
    return built_site.read_text(encoding="utf-8")


def external_resources(html: str) -> list[tuple[str, str]]:
    """Every tag in the page that pulls bytes from another origin.

    Returns (tag_kind, full_tag) pairs for `<script src>` and
    `<link href>` tags whose URL is absolute.
    """
    found = []
    for match in re.finditer(r"<script\b[^>]*\bsrc\s*=\s*[\"']([^\"']+)[\"'][^>]*>", html):
        if re.match(r"https?://|//", match.group(1)):
            found.append(("script", match.group(0)))
    for match in re.finditer(r"<link\b[^>]*\bhref\s*=\s*[\"']([^\"']+)[\"'][^>]*>", html):
        # rel="canonical" is metadata for crawlers — the browser never fetches
        # its href, so there are no bytes for an integrity hash to pin.
        if re.search(r"rel\s*=\s*[\"']canonical[\"']", match.group(0)):
            continue
        if re.match(r"https?://|//", match.group(1)):
            found.append(("link", match.group(0)))
    return found


# --------------------------------------------------------------------------
# One file, and only one file
# --------------------------------------------------------------------------


def test_build_writes_a_single_self_contained_file(tmp_path):
    """The build produces one HTML file and nothing beside it."""
    out = tmp_path / "nested" / "index.html"
    result = export.build_site(str(out))

    assert out.exists()
    assert Path(result) == out, "build_site should return the path it wrote"
    assert [p.name for p in out.parent.iterdir()] == ["index.html"], (
        "the arena must not need sidecar files"
    )


def test_no_local_asset_references_survive_the_build(html):
    """Styles and scripts are inlined, not linked — there is nothing to link to.

    A `data:` URI is still self-contained, so it is allowed; a relative path
    is not, because there is nothing next to the file to resolve it against.

    The favicon is the one exception, and only because it is committed next
    to the page: `docs/favicon.svg` ships beside `docs/index.html` the same
    way the recorded match video does, so a relative `rel="icon"` link
    resolves on the static host. Any other relative link still fails here.
    """
    for tag in re.findall(r"<link\b[^>]*>", html):
        href = re.search(r'href\s*=\s*["\']([^"\']+)["\']', tag)
        if href is None or re.match(r"https?://|data:|#", href.group(1)):
            continue
        assert re.search(r'rel\s*=\s*["\']icon["\']', tag), (
            f"a relative link would 404 on a static host: {tag}"
        )
        assert (REPO_ROOT / "docs" / href.group(1)).is_file(), (
            f"the favicon link points at docs/{href.group(1)}, which is not committed"
        )
    assert not re.search(r'<script\b[^>]*src\s*=\s*["\'](?!https?://|data:)', html), (
        "a relative script src would 404 on a static host"
    )


def test_no_unresolved_placeholders_remain(html):
    """Every injection point got filled in.

    The brace check looks for a mustache-shaped placeholder — `{{ name }}` —
    rather than for a bare `{{`. Both braces occur legitimately in the built
    page: `}}` closes any nested JSON object in the baked data, and `{{`
    opens a JSDoc record type like `{{prev: *, dt: number}}`. Matching the
    placeholder shape catches a real leftover without flagging either.
    """
    leftovers = [
        marker
        for marker in ("/*__DATA__*/", "__STYLES__", "__SCRIPTS__", "<!--__")
        if marker in html
    ]
    assert leftovers == [], f"unresolved template markers in the built page: {leftovers}"
    assert not re.search(r"\{\{\s*[\w.]+\s*\}\}", html), "an unrendered {{placeholder}} survived"


def test_page_is_valid_enough_to_open(html):
    """A minimal structural sanity check on the shell."""
    assert html.lstrip().lower().startswith("<!doctype html>")
    assert html.rstrip().lower().endswith("</html>")
    for tag in ("<html", "<head", "<body", "</body>", "</head>"):
        assert tag in html.lower(), f"missing {tag}"
    assert "<title>" in html.lower()
    assert 'name="viewport"' in html, "the arena has to be usable on a phone"


def test_shell_has_the_regions_the_arena_will_fill(html):
    """The placeholders the interactive build hangs itself on are present."""
    for anchor in (
        'id="arena"',
        'id="demoCanvas"',
        'id="how"',
        'id="accuracy"',
        'id="measurements"',
        'id="results"',
        "<header",
    ):
        assert anchor in html, f"missing page region {anchor!r}"


def test_every_in_page_link_has_a_target(html):
    """A `#name` link that points at nothing scrolls nowhere and switches no view.

    The page moves between its game and measurements views by hash, so a
    renamed section would leave a dead link in the navigation without this.
    """
    ids = set(re.findall(r'\sid="([^"]+)"', html))
    targets = set(re.findall(r'href="#([^"]+)"', html))
    assert targets, "expected in-page links"
    missing = sorted(targets - ids)
    assert missing == [], f"links with no target on the page: {missing}"


# --------------------------------------------------------------------------
# Typefaces
# --------------------------------------------------------------------------

FONTS_DIR = REPO_ROOT / "site" / "fonts"


def test_fonts_are_inlined_as_data_uris(html):
    """The page makes no font request of its own: every face rides along inside it."""
    assert not re.search(r"url\(\s*[\"']?fonts/", html), "a font is still linked by path"
    faces = re.findall(r"@font-face\s*{(.*?)}", html, re.S)
    assert len(faces) >= 3, f"expected the three typeface files, found {len(faces)} faces"
    for face in faces:
        assert 'url("data:font/woff2;base64,' in face, f"a face without an inlined file: {face[:80]}"


def test_every_font_file_is_used_and_accounted_for():
    """Each file in site/fonts is named by a stylesheet and in the licence notice."""
    styles = "".join(path.read_text(encoding="utf-8") for path in (REPO_ROOT / "site").glob("*.css"))
    notice = (FONTS_DIR / "NOTICE.md").read_text(encoding="utf-8")
    fonts = sorted(FONTS_DIR.glob("*.woff2"))
    assert fonts, "no typeface files in site/fonts"
    for font in fonts:
        assert f"fonts/{font.name}" in styles, f"{font.name} is shipped but no stylesheet uses it"
        assert font.name in notice, f"{font.name} is missing from site/fonts/NOTICE.md"
        assert font.read_bytes()[:4] == b"wOF2", f"{font.name} is not a WOFF2 file"


def test_a_missing_font_fails_the_build():
    """A face that is not there must stop the build, not fall back silently."""
    with pytest.raises(FileNotFoundError):
        export._inline_fonts('src: url("fonts/not-a-font.woff2")', Path("style.css"))


MAX_DEMO_VIDEO_BYTES = 5 * 1024 * 1024
"""Five megabytes: comfortably over the ~2 MB H.264 encode, still small
enough to serve same-origin next to the page without noticeably slowing it
down."""


def test_recorded_match_video_is_shipped_and_referenced(html):
    """`docs/demo_match.mp4` exists, is small, and the built page points at it.

    The video is not inlined into `RL_DATA` — at several megabytes it would
    blow the page's own size budget — so it ships as a same-origin sibling
    file next to `docs/index.html` instead, and the arena references it by a
    bare relative path (`demo_match.mp4`) rather than an absolute URL, which
    is exactly what `test_no_local_asset_references_survive_the_build` would
    otherwise flag for a `<script>` or `<link>`. A `<video>` tag is neither,
    so that check does not apply here, and this test covers the video's own
    promise instead: the file is committed, it is small, and the page that
    is supposed to play it actually names it.
    """
    video_path = REPO_ROOT / "docs" / "demo_match.mp4"
    assert video_path.exists(), f"{video_path} should be committed alongside docs/index.html"

    size = video_path.stat().st_size
    assert size < MAX_DEMO_VIDEO_BYTES, (
        f"docs/demo_match.mp4 is {size / 1024 / 1024:.1f} MiB, over the "
        f"{MAX_DEMO_VIDEO_BYTES / 1024 / 1024:.0f} MiB budget"
    )

    with open(video_path, "rb") as handle:
        # The ftyp box near the start of an MP4 names its brand; isom/mp42
        # are what libx264 + faststart produce. This is a cheap check that
        # the file is actually a browser-playable H.264 mp4 and not, say, an
        # OpenCV mp4v file that happens to share the extension.
        header = handle.read(64)
    assert b"ftyp" in header, "docs/demo_match.mp4 does not look like a valid mp4 container"

    assert "demo_match.mp4" in html, "the built page never references the recorded match video"


def test_site_fits_in_the_size_budget(built_site):
    size = built_site.stat().st_size
    assert size < MAX_SITE_BYTES, (
        f"built page is {size / 1024:.0f} KiB, over the {MAX_SITE_BYTES / 1024:.0f} KiB budget"
    )
    assert size > 50 * 1024, f"built page is suspiciously small ({size} bytes) — did the data bake?"


# --------------------------------------------------------------------------
# Anything fetched from elsewhere is pinned
# --------------------------------------------------------------------------


def test_every_external_resource_is_integrity_pinned(html):
    """A CDN tag without a hash is an invitation to serve different bytes later."""
    resources = external_resources(html)
    assert resources, "expected at least one pinned CDN resource (the chart library)"

    for kind, tag in resources:
        if any(host in tag for host in SRI_EXEMPT_HOSTS):
            continue
        assert "integrity=" in tag, f"unpinned {kind} tag: {tag}"
        assert "crossorigin" in tag, f"{kind} tag needs crossorigin for SRI to apply: {tag}"
        assert re.search(r'integrity\s*=\s*["\']sha(256|384|512)-', tag), (
            f"integrity must be a sha256/384/512 hash: {tag}"
        )


def test_cdn_urls_are_pinned_to_exact_versions(html):
    """No floating `@latest` or bare major — the hash has to keep matching."""
    for _, tag in external_resources(html):
        url = re.search(r'(?:src|href)\s*=\s*["\']([^"\']+)["\']', tag).group(1)
        assert "@latest" not in url, f"floating version in {url}"
        assert re.search(r"@\d+\.\d+\.\d+", url), f"CDN URL is not pinned to an exact version: {url}"


def test_runtime_loaded_dependency_is_documented(html):
    """The one thing that cannot carry a hash says so, in the page itself.

    MediaPipe resolves its own wasm and model URLs at run time, so there are
    no build-time bytes to hash. That is a real exception to the rule above,
    and it is written down where the next person will find it rather than
    left as an unexplained gap.
    """
    assert "tasks-vision" in html, "the pose runtime should be referenced and explained"
    lowered = html.lower()
    assert "integrity" in lowered and "wasm" in lowered, (
        "the page should explain why the pose runtime carries no integrity hash"
    )


# --------------------------------------------------------------------------
# The data the page runs on
# --------------------------------------------------------------------------


def extract_rl_data(html: str) -> dict:
    """Pull `window.RL_DATA = {...};` back out of the built page."""
    marker = "window.RL_DATA = "
    start = html.index(marker) + len(marker)
    end = html.index("\n", start)
    payload = html[start:end].rstrip().rstrip(";")
    return json.loads(payload)


def test_rl_data_is_present_and_parses(html):
    assert "window.RL_DATA" in html
    data = extract_rl_data(html)
    assert isinstance(data, dict)


def test_rl_data_carries_every_section_the_page_needs(html):
    data = extract_rl_data(html)
    for key in ("config", "chant", "fixtures", "benchmark"):
        assert key in data, f"RL_DATA is missing {key!r}"


def test_rl_data_config_matches_the_python_defaults(html):
    """The browser judge runs on the same constants the engine was tuned with."""
    from redlight import judge
    from redlight.config import GameConfig

    data = extract_rl_data(html)["config"]
    defaults = GameConfig()

    assert data["window"] == judge.WINDOW == 96
    assert data["diffPixelDelta"] == judge.DIFF_PIXEL_DELTA == 25
    assert data["sampleIntervalS"] == judge.SAMPLE_INTERVAL_S
    assert data["timestampToleranceS"] == judge.TIMESTAMP_TOLERANCE_S
    assert data["threshold"] == defaults.threshold
    assert data["diffThreshold"] == defaults.diff_threshold
    assert data["confirmFrames"] == defaults.confirm_frames
    assert data["smoothing"] == defaults.smoothing
    assert data["graceS"] == defaults.grace_s
    assert data["countdownS"] == defaults.countdown_s
    assert data["durationS"] == defaults.duration_s
    assert data["phaseMinS"] == defaults.phase_min_s
    assert data["phaseMaxS"] == defaults.phase_max_s


def test_rl_data_benchmark_subset_carries_the_results_the_page_shows(html):
    benchmark = extract_rl_data(html)["benchmark"]

    for key in (
        "classifiers",
        "chosen_thresholds",
        "resolution_sweep",
        "fps_sweep",
        "background_scenario",
        "runtime_ms",
        "traces",
        "demo_match",
        "meta",
    ):
        assert key in benchmark, f"benchmark subset is missing {key!r}"

    # Both background variants, since the honest one is the occluded run.
    assert set(benchmark["background_scenario"]) >= {"isolated", "occluded"}

    # All eight player traces, each labelled with what it shows.
    traces = benchmark["traces"]["red_phase_player_traces"]
    assert len(traces) == 8
    for trace in traces:
        assert {"track_id", "kind", "t", "score"} <= set(trace)
        assert trace["kind"], "every trace needs a kind label to be readable"
        assert len(trace["t"]) == len(trace["score"])


def test_rl_data_chant_is_playable(html):
    chant = extract_rl_data(html)["chant"]
    assert chant["bpm"] > 0
    assert len(chant["notes"]) > 0
    for midi, beats in chant["notes"]:
        assert 0 <= midi <= 127
        assert beats > 0


def test_rl_data_fixtures_are_the_committed_golden_file(html):
    """The page self-tests against the same fixtures the parity suite uses."""
    data = extract_rl_data(html)["fixtures"]
    committed = json.loads(
        (REPO_ROOT / "tests" / "fixtures" / "judge_fixtures.json").read_text(encoding="utf-8")
    )
    assert data == committed


def test_selftest_hook_is_wired_into_the_page(html):
    """`?selftest=1` has to exist in the shipped file, not just in the source tree."""
    assert "selftest" in html
    assert "SELFTEST PASS" in html
    assert "SELFTEST FAIL" in html
    assert "document.title" in html


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def test_two_builds_are_byte_identical(tmp_path):
    """The committed page must only change when the inputs change."""
    first = tmp_path / "first.html"
    second = tmp_path / "second.html"
    export.build_site(str(first))
    export.build_site(str(second))

    assert first.read_bytes() == second.read_bytes(), (
        "build_site is not deterministic; every build would churn the committed page"
    )


def test_committed_page_is_up_to_date(tmp_path):
    """`docs/index.html` in the repository matches what the build produces now.

    The published arena is a build artefact that is committed on purpose, so
    a visitor gets the current page. If this fails, build it again.
    """
    committed = REPO_ROOT / "docs" / "index.html"
    assert committed.exists(), "docs/index.html should be committed alongside its sources"

    fresh = tmp_path / "index.html"
    export.build_site(str(fresh))
    assert committed.read_bytes() == fresh.read_bytes(), (
        "docs/index.html is stale — run `redlight build-site` and commit the result"
    )


# --------------------------------------------------------------------------
# Fixture generation
# --------------------------------------------------------------------------


def test_write_judge_fixtures_is_seeded_and_reproducible(tmp_path):
    """Same call, same bytes — the golden file is not allowed to wander."""
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    export.write_judge_fixtures(str(first))
    export.write_judge_fixtures(str(second))

    assert first.read_bytes() == second.read_bytes()


def test_write_judge_fixtures_honours_the_case_count(tmp_path):
    path = tmp_path / "fixtures.json"
    export.write_judge_fixtures(str(path), n=30)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert len(data["cases"]) == 30
    assert len({case["name"] for case in data["cases"]}) == 30
    for case in data["cases"]:
        assert len(base64.b64decode(case["prev"])) == 96 * 96
        assert len(base64.b64decode(case["cur"])) == 96 * 96


def test_write_judge_fixtures_refuses_to_drop_the_boundary_cases(tmp_path):
    """The hand-built boundary cases are the point; `n` cannot cut them out."""
    with pytest.raises(ValueError):
        export.write_judge_fixtures(str(tmp_path / "too_few.json"), n=3)


def test_generated_fixtures_span_zero_and_non_zero_scores(tmp_path):
    """A fixture set that scored 0.0 everywhere would pass any broken port."""
    path = tmp_path / "fixtures.json"
    export.write_judge_fixtures(str(path))
    scores = [case["expected"] for case in json.loads(path.read_text(encoding="utf-8"))["cases"]]

    assert any(score == 0.0 for score in scores), "expected some genuinely still pairs"
    assert any(score > 0.0 for score in scores), "expected some genuinely moving pairs"
    assert max(scores) > 1.0, "expected at least one strongly moving pair"
