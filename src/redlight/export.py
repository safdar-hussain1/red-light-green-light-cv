"""Build the browser arena, and the golden fixtures that hold it honest.

Two jobs live here, and they are two halves of one claim.

`write_judge_fixtures` records what the Python judge says about a set of
deliberately awkward 96x96 window pairs — identical frames, sensor-grade
noise, single pixels sitting exactly on the change threshold and one step
past it, block shifts, flat textures. Those recorded numbers are the
contract. `tests/test_js_parity.py` replays the same pairs through
`site/judge.js` and requires bit-identical answers, so "the browser runs the
same scoring kernel" is a tested fact rather than a claim in a README.

What that does and does not cover is worth stating plainly, because the
distinction is easy to lose in a sentence. The fixtures pin the *scoring*
path: given two 96x96 uint8 windows and a dt, both languages produce the
same number. They say nothing about how those windows were produced.
`judge.crop_window` clamps the detection box, resamples with OpenCV's
INTER_AREA and applies a 3x3 Gaussian blur; the browser arrives at its
windows through canvas instead. So the honest claim is identical scoring of
identical windows, not identical verdicts from an identical camera frame.

`build_site` collapses the arena into a single HTML file: the page shell,
every stylesheet and script from `site/`, and one `window.RL_DATA` blob
carrying the engine's constants, the published benchmark results, the chant,
and those same golden fixtures. One file, no build step, no server — drop it
on a static host and it works. The fixtures ride along so the shipped page
can re-run that scoring-kernel comparison in the visitor's own browser via
`?selftest=1`, in whatever engine actually loaded it.

The build is deterministic on purpose. `docs/index.html` is a committed
artefact, and a build that varied run to run would make every commit of it a
diff full of noise hiding the one line that actually changed.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from redlight import judge
from redlight.config import GameConfig
from redlight.judge import WINDOW, diff_score_window
from redlight.synth import load_chant

REPO_ROOT = Path(__file__).resolve().parents[2]
SITE_DIR = REPO_ROOT / "site"
TEMPLATE_PATH = SITE_DIR / "template.html"
BENCHMARK_PATH = REPO_ROOT / "reports" / "benchmark_results.json"
FIXTURES_PATH = REPO_ROOT / "tests" / "fixtures" / "judge_fixtures.json"

DEFAULT_SITE_OUT = "docs/index.html"

JS_ORDER = ("judge.js", "game.js", "doll.js", "chant.js", "arena.js")
"""Load order for the inlined scripts.

Only files that actually exist are inlined, and anything in `site/` that is
not named here is appended afterward in sorted order — so a new script can
be added to the arena without touching this module, and only a script with a
real load-order dependency needs to be listed.
"""

STYLES_MARKER = "<!--__STYLES__-->"
SCRIPTS_MARKER = "<!--__SCRIPTS__-->"
DATA_MARKER = "/*__DATA__*/"

BENCHMARK_KEYS = (
    "meta",
    "classifiers",
    "chosen_thresholds",
    "resolution_sweep",
    "fps_sweep",
    "background_scenario",
    "runtime_ms",
    "traces",
    "demo_match",
)
"""The slice of the benchmark the page actually renders.

The results file carries intermediate working the page has no use for. Baking
only what is shown keeps the single-file page small enough to open on a phone
without dropping any figure the page makes a claim about.
"""

FIXTURE_SEED = 20260813
"""Fixed seed for fixture generation, so the golden file never wanders."""

FIXTURE_DT_SLOW = 0.1
"""The referee's own sampling interval."""

FIXTURE_DT_FAST = 0.033
"""A 30 fps frame gap — the other interval a real capture loop produces."""


# ---------------------------------------------------------------------------
# Golden fixtures
# ---------------------------------------------------------------------------


def _flat(value: int) -> np.ndarray:
    """A uniform window — the simplest possible baseline."""
    return np.full((WINDOW, WINDOW), value, dtype=np.uint8)


def _blocky_texture(rng: np.random.Generator, cells: int = 12) -> np.ndarray:
    """A coarse checker of random blocks.

    Blocks rather than per-pixel noise, because a shifted per-pixel texture
    changes essentially every pixel and stops discriminating between a small
    step and a large one. Eight-pixel blocks make a 1-pixel shift and a
    16-pixel shift score visibly differently, which is the point of having
    both in the set.
    """
    block = WINDOW // cells
    coarse = rng.integers(0, 256, size=(cells, cells)).astype(np.uint8)
    return np.kron(coarse, np.ones((block, block), dtype=np.uint8))


def _with_noise(base: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    """Add Gaussian sensor grain and clip back into byte range.

    Sigma 2 is the benchmark's measured noise level, and it sits well below
    DIFF_PIXEL_DELTA — so a still player reading 0.0 is structural, not luck.
    These fixtures pin that: grain alone must not score.
    """
    noisy = base.astype(np.float64) + rng.normal(0.0, sigma, size=base.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def _poke(base: np.ndarray, value: int, index: int = 0) -> np.ndarray:
    """Copy a window with exactly one pixel changed to `value`."""
    poked = base.copy()
    poked.flat[index] = value
    return poked


def _blob(background: int, level: int, top: int, left: int, size: int) -> np.ndarray:
    """A flat background with one bright rectangle — a stand-in for a body."""
    win = _flat(background)
    win[top : top + size, left : left + size] = level
    return win


def _core_cases() -> list[dict]:
    """The hand-built cases. Every one of these is here for a named reason."""
    rng = np.random.default_rng(FIXTURE_SEED)

    gray = _flat(128)
    texture = _blocky_texture(rng)
    fine = _blocky_texture(rng, cells=24)

    cases: list[dict] = [
        # Nothing moved. The score must be exactly zero, not nearly zero.
        {"name": "identical_flat_gray", "kind": "identical", "dt": FIXTURE_DT_SLOW,
         "prev": gray, "cur": gray},
        {"name": "identical_texture", "kind": "identical", "dt": FIXTURE_DT_FAST,
         "prev": texture, "cur": texture},

        # Sensor grain at the benchmark's measured sigma. Still zero: the
        # deadband is what makes a frozen player read as frozen.
        {"name": "noise_sigma2_flat", "kind": "noise", "dt": FIXTURE_DT_SLOW,
         "prev": gray, "cur": _with_noise(gray, rng, 2.0)},
        {"name": "noise_sigma2_texture", "kind": "noise", "dt": FIXTURE_DT_FAST,
         "prev": texture, "cur": _with_noise(texture, rng, 2.0)},
        # Grain loud enough to cross the delta, so the fixtures do not only
        # assert that noise is ignored — they pin where it stops being.
        {"name": "noise_sigma20_flat", "kind": "noise", "dt": FIXTURE_DT_SLOW,
         "prev": gray, "cur": _with_noise(gray, rng, 20.0)},

        # The threshold itself. The comparison is strict `>`, so a swing of
        # exactly 25 must not count and 26 must. These four are the cases a
        # port is most likely to get subtly wrong.
        {"name": "single_pixel_plus_25", "kind": "boundary", "dt": FIXTURE_DT_SLOW,
         "prev": gray, "cur": _poke(gray, 153)},
        {"name": "single_pixel_plus_26", "kind": "boundary", "dt": FIXTURE_DT_SLOW,
         "prev": gray, "cur": _poke(gray, 154)},
        {"name": "single_pixel_minus_25", "kind": "boundary", "dt": FIXTURE_DT_FAST,
         "prev": gray, "cur": _poke(gray, 103)},
        {"name": "single_pixel_minus_26", "kind": "boundary", "dt": FIXTURE_DT_FAST,
         "prev": gray, "cur": _poke(gray, 102)},

        # Wrap risk. Subtract these as unsigned bytes and one direction comes
        # out as 230 instead of 26 — a pixel that should not score, scoring.
        # Both directions are pinned so a port cannot pass on the easy sign.
        {"name": "wrap_risk_100_to_126", "kind": "boundary", "dt": FIXTURE_DT_SLOW,
         "prev": _flat(100), "cur": _poke(_flat(100), 126)},
        {"name": "wrap_risk_126_to_100", "kind": "boundary", "dt": FIXTURE_DT_SLOW,
         "prev": _flat(126), "cur": _poke(_flat(126), 100)},
        # The full-range version of the same trap.
        {"name": "wrap_risk_0_to_255", "kind": "boundary", "dt": FIXTURE_DT_FAST,
         "prev": _flat(0), "cur": _poke(_flat(0), 255)},
        {"name": "wrap_risk_255_to_0", "kind": "boundary", "dt": FIXTURE_DT_FAST,
         "prev": _flat(255), "cur": _poke(_flat(255), 0)},

        # A whole row on each side of the boundary: 0 pixels, then 96.
        {"name": "row_delta_25", "kind": "boundary", "dt": FIXTURE_DT_SLOW,
         "prev": gray, "cur": np.concatenate(
             [_flat(153)[:1], gray[1:]], axis=0)},
        {"name": "row_delta_26", "kind": "boundary", "dt": FIXTURE_DT_SLOW,
         "prev": gray, "cur": np.concatenate(
             [_flat(154)[:1], gray[1:]], axis=0)},

        # Block shifts: real motion, at sizes that should score differently.
        {"name": "shift_texture_1px", "kind": "shift", "dt": FIXTURE_DT_SLOW,
         "prev": texture, "cur": np.roll(texture, 1, axis=1)},
        {"name": "shift_texture_4px", "kind": "shift", "dt": FIXTURE_DT_SLOW,
         "prev": texture, "cur": np.roll(texture, 4, axis=1)},
        {"name": "shift_texture_16px", "kind": "shift", "dt": FIXTURE_DT_FAST,
         "prev": texture, "cur": np.roll(texture, 16, axis=1)},
        {"name": "shift_fine_vertical_8px", "kind": "shift", "dt": FIXTURE_DT_FAST,
         "prev": fine, "cur": np.roll(fine, 8, axis=0)},

        # A body-sized blob taking a step, with and without grain on top.
        {"name": "blob_step_2px", "kind": "blob", "dt": FIXTURE_DT_SLOW,
         "prev": _blob(40, 200, 20, 20, 40), "cur": _blob(40, 200, 20, 22, 40)},
        {"name": "blob_step_12px", "kind": "blob", "dt": FIXTURE_DT_SLOW,
         "prev": _blob(40, 200, 20, 20, 40), "cur": _blob(40, 200, 20, 32, 40)},
        {"name": "blob_held_with_noise", "kind": "blob", "dt": FIXTURE_DT_FAST,
         "prev": _blob(40, 200, 20, 20, 40),
         "cur": _with_noise(_blob(40, 200, 20, 20, 40), rng, 2.0)},

        # Two unrelated textures: the saturated end of the scale, where
        # almost every pixel changes.
        {"name": "texture_unrelated_pair", "kind": "texture", "dt": FIXTURE_DT_SLOW,
         "prev": _blocky_texture(rng), "cur": _blocky_texture(rng)},
        {"name": "texture_unrelated_fine", "kind": "texture", "dt": FIXTURE_DT_FAST,
         "prev": _blocky_texture(rng, cells=32), "cur": _blocky_texture(rng, cells=32)},
    ]
    return cases


def _extra_cases(count: int) -> list[dict]:
    """Procedural filler, for when a caller asks for more than the core set.

    Seeded from the same generator, so asking for more cases never changes
    the ones already there.
    """
    rng = np.random.default_rng(FIXTURE_SEED + 1)
    cases = []
    for index in range(count):
        base = _blocky_texture(rng, cells=[8, 12, 16, 24][index % 4])
        shift = 1 + (index % 7)
        axis = index % 2
        cases.append(
            {
                "name": f"generated_shift_{index:02d}",
                "kind": "shift" if index % 3 else "texture",
                "dt": FIXTURE_DT_SLOW if index % 2 == 0 else FIXTURE_DT_FAST,
                "prev": base,
                "cur": np.roll(base, shift, axis=axis)
                if index % 3
                else _blocky_texture(rng, cells=16),
            }
        )
    return cases


def judge_fixtures(n: int = 24) -> dict:
    """Build the golden fixture set in memory.

    Args:
        n: How many cases to produce. Must be at least the size of the
            hand-built core set — those boundary cases are the entire point,
            and trimming them would leave a fixture file that a broken port
            could pass.

    Raises:
        ValueError: If `n` is smaller than the core set.
    """
    core = _core_cases()
    if n < len(core):
        raise ValueError(
            f"n must be at least {len(core)} to keep every boundary case, got {n}"
        )

    cases = core + _extra_cases(n - len(core))

    encoded = []
    for case in cases:
        prev, cur = case["prev"], case["cur"]
        encoded.append(
            {
                "name": case["name"],
                "kind": case["kind"],
                "dt": case["dt"],
                "prev": base64.b64encode(prev.tobytes()).decode("ascii"),
                "cur": base64.b64encode(cur.tobytes()).decode("ascii"),
                "expected": diff_score_window(prev, cur, case["dt"]),
            }
        )

    return {
        "window": WINDOW,
        "diff_pixel_delta": judge.DIFF_PIXEL_DELTA,
        "note": (
            "Golden pairs for the browser judge. `expected` is what "
            "redlight.judge.diff_score_window returns for the decoded 96x96 "
            "uint8 windows at `dt`, recorded at full float precision. The "
            "browser port must reproduce these exactly."
        ),
        "cases": encoded,
    }


def write_judge_fixtures(path: str | Path = FIXTURES_PATH, n: int = 24) -> str:
    """Write the golden fixture file, creating parent directories as needed.

    Returns:
        The path written, as a string.
    """
    data = judge_fixtures(n)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return str(out)


# ---------------------------------------------------------------------------
# Site build
# ---------------------------------------------------------------------------


def _read(path: Path, what: str) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at {path}. The site build runs from a source "
            "checkout, not from an installed package."
        )
    return path.read_text(encoding="utf-8")


def _script_paths() -> list[Path]:
    """Every script in `site/`, ordered so dependencies load first."""
    present = {path.name: path for path in sorted(SITE_DIR.glob("*.js"))}
    ordered = [present.pop(name) for name in JS_ORDER if name in present]
    # Anything not spelled out in JS_ORDER has no declared dependency, so it
    # goes last in a stable order — sorted, never filesystem order, or the
    # build would stop being byte-reproducible across machines.
    ordered.extend(present[name] for name in sorted(present))
    return ordered


def _check_no_closing_tag(source: str, path: Path, tag: str) -> None:
    """Refuse to inline a source that would close its own wrapper tag.

    An HTML parser ends a `<script>` block at the first `</script` it sees,
    even inside a string or a comment. A source containing one would truncate
    the page at that point and dump the rest as body text — and the failure
    looks like a rendering bug a long way from its cause. Cheaper to refuse
    the build than to debug the page.
    """
    if f"</{tag}" in source.lower():
        raise ValueError(
            f"{path} contains a literal '</{tag}', which would end the inlined "
            f"<{tag}> block early and truncate the page. Split it (e.g. "
            f"'<\\/{tag}') before inlining."
        )


def _inline_styles() -> str:
    blocks = []
    for path in sorted(SITE_DIR.glob("*.css")):
        source = path.read_text(encoding="utf-8")
        _check_no_closing_tag(source, path, "style")
        blocks.append(f"<style>\n/* {path.name} */\n{source}</style>")
    return "\n".join(blocks)


def _inline_scripts() -> str:
    blocks = []
    for path in _script_paths():
        source = path.read_text(encoding="utf-8")
        _check_no_closing_tag(source, path, "script")
        blocks.append(f"<script>\n// {path.name}\n{source}</script>")
    return "\n".join(blocks)


def _benchmark_subset() -> dict:
    """The published results, trimmed to what the page renders.

    Raises:
        KeyError: If the results file is missing a section the page shows.
            Silently dropping one would leave a chart rendering nothing with
            no indication why.
    """
    results = json.loads(_read(BENCHMARK_PATH, "benchmark results"))
    missing = [key for key in BENCHMARK_KEYS if key not in results]
    if missing:
        raise KeyError(
            f"benchmark results at {BENCHMARK_PATH} are missing {missing}; "
            "re-run `redlight benchmark`"
        )
    return {key: results[key] for key in BENCHMARK_KEYS}


def _site_data() -> dict:
    """Everything baked into `window.RL_DATA`."""
    defaults = GameConfig()

    if not FIXTURES_PATH.exists():
        # Deliberately not generated on the fly. The golden file is the
        # contract the browser is held to, and a build that quietly
        # manufactures its own contract when the real one is missing would
        # ship a page whose selftest proves nothing — it would be checking
        # the port against numbers produced by the same run. Generating it
        # is an explicit act, and the result gets committed and reviewed.
        raise FileNotFoundError(
            f"golden judge fixtures not found at {FIXTURES_PATH}. "
            "Run redlight.export.write_judge_fixtures() first and commit the "
            "result — the site build will not generate its own contract."
        )
    fixtures = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))

    return {
        "config": {
            "window": judge.WINDOW,
            "diffPixelDelta": judge.DIFF_PIXEL_DELTA,
            "sampleIntervalS": judge.SAMPLE_INTERVAL_S,
            "timestampToleranceS": judge.TIMESTAMP_TOLERANCE_S,
            # The browser scores with the diff metric — dense optical flow is
            # not affordable in a tab — so `diffThreshold` is the live cutoff
            # here, while `threshold` is what the Python engine uses. Both
            # ship so the page can show the pair it was tuned against.
            "threshold": defaults.threshold,
            "diffThreshold": defaults.diff_threshold,
            "confirmFrames": defaults.confirm_frames,
            "smoothing": defaults.smoothing,
            "graceS": defaults.grace_s,
            "countdownS": defaults.countdown_s,
            "durationS": defaults.duration_s,
            "phaseMinS": defaults.phase_min_s,
            "phaseMaxS": defaults.phase_max_s,
        },
        "chant": load_chant(),
        "fixtures": fixtures,
        "benchmark": _benchmark_subset(),
    }


def build_site(out: str | Path = DEFAULT_SITE_OUT) -> str:
    """Build the arena into one self-contained HTML file.

    Reads `site/template.html`, inlines every stylesheet and script from
    `site/`, and bakes the engine's constants, the benchmark results, the
    chant, and the golden judge fixtures into `window.RL_DATA`. The result
    needs no server, no build step, and no network beyond the two pinned CDN
    entries the template documents.

    The output is byte-reproducible: given the same sources it produces the
    same file every time, so the committed `docs/index.html` only changes
    when something real changed.

    Args:
        out: Where to write the page.

    Returns:
        The path written, as a string.

    Raises:
        FileNotFoundError: If the template or the benchmark results are
            missing.
        ValueError: If a build marker is absent from the template.
    """
    html = _read(TEMPLATE_PATH, "site template")

    for marker in (STYLES_MARKER, SCRIPTS_MARKER, DATA_MARKER):
        if marker not in html:
            raise ValueError(f"template at {TEMPLATE_PATH} is missing the {marker} marker")

    # Compact separators: the fixtures dominate the payload, and pretty
    # printing them would cost a third of the page size for nothing. Sorting
    # is deliberately off — the benchmark's own key order is meaningful, and
    # Python preserves it, so the output stays stable without reordering.
    data = json.dumps(_site_data(), separators=(",", ":"), ensure_ascii=False)

    # The blob is emitted inside a <script> block, where an HTML parser ends
    # the block at the first `</script` it sees — string literal or not. A
    # benchmark note or a trace label containing `</script>` would therefore
    # truncate the page. `\/` is a valid JSON escape for `/`, so escaping
    # every `</` parses back to exactly the same string while making that
    # sequence impossible to form. Cheap, and it removes a whole class of
    # "the page renders half-way and then shows raw JSON" failure.
    data = data.replace("</", "<\\/")

    html = html.replace(STYLES_MARKER, _inline_styles())
    html = html.replace(SCRIPTS_MARKER, _inline_scripts())
    html = html.replace(DATA_MARKER, data)

    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    return str(destination)
