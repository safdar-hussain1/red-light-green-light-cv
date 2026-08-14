#!/usr/bin/env python3
"""Drive the published arena in a real browser and report what it did.

The unit tests hold the judge, the game and the build. None of them opens
the page. This does: it runs `docs/index.html` in headless Chrome three
different ways, and every one of them ends in an assertion rather than a
screenshot somebody is supposed to squint at.

**The fixture selftest.** `?selftest=1` replays every golden 96x96 window
pair through the shipped page and writes the verdict into `document.title`,
so `--dump-dom` can read it. This is the browser-side half of the parity
claim: CI proves it under node, this proves it in a browser engine.

**The play path.** A camera cannot be granted headlessly on macOS, so the
page is served from a local HTTP server with a small script injected ahead
of it that swaps `navigator.mediaDevices.getUserMedia` for a
`canvas.captureStream(30)` of animated synthetic content, and fills in
`window.RL_POSE_FACTORY` — the seam `site/pose.js` documents — so no wasm
blob has to be fetched. Everything downstream of those two substitutions is
the real thing: the real crop path, the real sampler, the real judge, the
real game machine. The page records the phases it passed through, and every
call it made, in its own title under `?probe=1` — the one piece of state a
`--dump-dom` run can read back.

The synthetic footage draws two people: one who walks and one who does not
move a pixel, both with speckled skin and both with a seeded plus or minus
2 px of jitter on the box the fake pose runtime reports. That jitter is the
point. Landmarks wobble on a real camera, and a crop window cut from a
wobbling box slides across a stationary body, which reads as movement. The
run asserts both halves of the referee: the walker is called out, and the
still one survives two whole red lights.

Getting a `--dump-dom` to happen *late* takes a trick, because the dump
fires on the load event and `--virtual-time-budget` — the usual way to wait
— freezes media, which is exactly what this run needs running. So the
injected markup asks for an image the server holds open for a few seconds.
The load event waits for it, the match plays out underneath, and the dump
lands after the arena has reached a green light.

**Screenshots.** Both themes, desktop and phone, plus the arena mid-match
and each section on its own. Headless Chrome renders dark by default, so
light is stamped explicitly rather than assumed — `?theme=` is honoured by
the page's own head script.

One limit worth knowing: headless Chrome on macOS will not lay out a
viewport narrower than 500 CSS pixels, whatever `--window-size` asks for. A
narrower request is rendered at 500 and then cropped, which looks like a
broken page rather than a small one. The phone captures are therefore taken
at 500 — still inside the page's narrow breakpoint, so the phone layout is
the one being photographed.

Usage:
    python scripts/verify_site.py            # everything
    python scripts/verify_site.py --shots-only
"""

from __future__ import annotations

import argparse
import functools
import html
import http.server
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE = REPO_ROOT / "docs" / "index.html"
SHOT_DIR = REPO_ROOT / "reports" / "site"

CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
)

BASE_FLAGS = (
    "--headless",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--disable-gpu",
    "--hide-scrollbars",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-extensions",
    "--allow-file-access-from-files",
)

PHONE_WIDTH = 500
"""The narrowest viewport headless Chrome will actually lay out on macOS.

Asking for 390 does not produce a 390-pixel layout; it produces a 500-pixel
layout cropped to 390, which photographs as a broken page. 500 is inside the
page's narrow breakpoint, so this is still the phone layout.
"""

HOLD_MS = 13000
"""How long the server holds the injected image open, delaying the load event.

The run has to cover registration (1 s), the countdown (3 s) and then two
complete red lights, because one red light cannot tell the difference
between a player who is holding still and a player who has not been armed
against yet. The stub pins both ends of the phase length at 1.5 s, so the
second red light closes at about 10.3 s and this leaves the margin.
"""

# A 1x1 transparent GIF. The response body has to be a real image or Chrome
# gives up on it early and the load event stops waiting.
PIXEL_GIF = bytes.fromhex(
    "47494638396101000100800000000000ffffff21f90401000000002c000000000100"
    "010000020144003b"
)

INJECTION = """
<script>
/*
 * Verification stubs. Injected by scripts/verify_site.py, never shipped.
 *
 * Three substitutions, all at the edge of the system: the camera, the pose
 * runtime, and the length of a light. Everything between them and the
 * verdict is the real code — the real crop path, the real sampler, the real
 * judge, the real rules.
 *
 * The footage is built to answer the one question the unit tests cannot ask:
 * does somebody who is genuinely holding still survive a red light in a
 * browser? So it draws two people. The left one walks. The right one does
 * not move a single pixel, ever. Both carry a fixed speckled texture,
 * because a flat rectangle cannot reveal a crop window that slipped
 * sideways — and sliding sideways is exactly what the crop window used to
 * do, because both figures' detection boxes are handed a seeded plus or
 * minus 2 px of jitter on every frame. That is what pose landmarks actually
 * do on a real camera, and scoring a statue through a box that moves is how
 * a statue used to get called for moving.
 */
(function () {
  var FRAME_W = 640;
  var FRAME_H = 480;

  /* Landmark wobble, in frame pixels, applied to every reported box. */
  var JITTER_PX = 2;

  var canvas = document.createElement("canvas");
  canvas.width = FRAME_W;
  canvas.height = FRAME_H;
  var ctx = canvas.getContext("2d");
  var frame = 0;

  // Seeded, both of them. A verification that fails one run in five proves
  // nothing at all, and this is the same generator the game draws its light
  // schedule from.
  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  var texture = mulberry32(1789);
  var wobble = mulberry32(9041);

  // Two bodies. `sway` is how far the figure actually walks: the second one
  // is pinned at zero and must survive every red light in the run.
  var FIGURES = [
    { home: 90, top: 150, w: 110, h: 240, head: 40, skin: "#d9cfbe", sway: 40, dots: [] },
    { home: 400, top: 160, w: 100, h: 230, head: 38, skin: "#c9d6cf", sway: 0, dots: [] },
  ];

  // Markings are drawn once and then belong to the body. A figure that
  // re-randomises its own pixels every frame is not a person standing
  // still, and no amount of crop stabilisation could rescue one.
  FIGURES.forEach(function (fig) {
    for (var i = 0; i < 260; i += 1) {
      fig.dots.push({
        dx: Math.round(texture() * (fig.w - 4)),
        dy: Math.round(texture() * (fig.h - 4)),
        shade: i % 2 ? "#2b2b2b" : "#f2f2f2",
      });
    }
  });

  function offsetOf(fig) {
    return fig.sway ? Math.round(Math.sin(frame / 11) * fig.sway) : 0;
  }

  function draw() {
    frame += 1;
    ctx.fillStyle = "#1d2b33";
    ctx.fillRect(0, 0, FRAME_W, FRAME_H);
    ctx.fillStyle = "#33454f";
    ctx.fillRect(0, 360, FRAME_W, 120);

    FIGURES.forEach(function (fig) {
      var x = fig.home + offsetOf(fig);
      ctx.fillStyle = fig.skin;
      ctx.fillRect(x, fig.top, fig.w, fig.h);
      ctx.beginPath();
      ctx.arc(x + fig.w / 2, fig.top - 22, fig.head, 0, 6.3);
      ctx.fill();
      fig.dots.forEach(function (dot) {
        ctx.fillStyle = dot.shade;
        ctx.fillRect(x + dot.dx, fig.top + dot.dy, 4, 4);
      });
    });

    requestAnimationFrame(draw);
  }
  draw();

  var stream = canvas.captureStream(30);
  if (!navigator.mediaDevices) {
    Object.defineProperty(navigator, "mediaDevices", { value: {}, configurable: true });
  }
  navigator.mediaDevices.getUserMedia = function () {
    return Promise.resolve(stream);
  };

  // The seam site/pose.js documents: stand in for the wasm pose runtime so
  // the play path can be exercised with no network at all. The box follows
  // the body it belongs to, and then wobbles, which is the whole point.
  window.RL_POSE_FACTORY = function () {
    return Promise.resolve({
      detect: function () {
        return FIGURES.map(function (fig) {
          var x = fig.home + offsetOf(fig);
          var jx = (wobble() * 2 - 1) * JITTER_PX;
          var jy = (wobble() * 2 - 1) * JITTER_PX;
          return {
            x1: (x + jx) / FRAME_W,
            y1: (fig.top - fig.head - 22 + jy) / FRAME_H,
            x2: (x + fig.w + jx) / FRAME_W,
            y2: (fig.top + fig.h + jy) / FRAME_H,
          };
        });
      },
      close: function () {},
    });
  };

  // The click is deferred rather than fired from this listener directly.
  // Listeners run in registration order, and this script is injected ahead
  // of the page's own — clicking here would land before the arena has bound
  // its handler, and nothing would happen.
  document.addEventListener("DOMContentLoaded", function () {
    document.documentElement.style.scrollBehavior = "auto";

    // The third substitution. Shipped phase lengths are drawn from 2-5 s,
    // and two whole red lights have to fit inside the window the held image
    // keeps the load event open for. Pinning both ends makes the schedule
    // short and repeatable; not one rule changes.
    if (window.RL_DATA && window.RL_DATA.config) {
      window.RL_DATA.config.phaseMinS = 1.5;
      window.RL_DATA.config.phaseMaxS = 1.5;
    }

    var action = new URLSearchParams(window.location.search).get("action");
    window.setTimeout(function () {
      var target =
        action === "replay"
          ? document.querySelector("[data-replay]")
          : document.getElementById("btnPlay");
      if (target) target.click();
    }, 250);
    // Deliberately no scrolling. `--screenshot` rasterises the viewport and
    // places it at the document origin, so a scrolled page photographs as a
    // blank band where the hero should be. The arena capture uses a tall
    // window instead and lets the whole page fit above the fold.
  });
})();
</script>
"""

HOLD_MARKUP = '<img src="/hold?ms={ms}" alt="" width="1" height="1" />'


# --------------------------------------------------------------------------
# browser plumbing
# --------------------------------------------------------------------------


def chrome_binary() -> str:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise SystemExit(
        "No Chrome or Chromium found. Looked in:\n  " + "\n  ".join(CHROME_CANDIDATES)
    )


def run_chrome(args: list[str], timeout: float = 120.0) -> subprocess.CompletedProcess:
    """Run one headless Chrome and return what it printed.

    Deliberately no `--user-data-dir`. Headless Chrome makes its own
    throwaway profile, and handing it an explicit one on macOS while a
    desktop Chrome is already running makes the launch block indefinitely —
    which shows up here as every check timing out for no visible reason.
    """
    return subprocess.run(
        [chrome_binary(), *BASE_FLAGS, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def dump_page(url: str, extra: tuple[str, ...] = ()) -> tuple[str, str]:
    """Load a URL and return its final document title and serialised DOM."""
    result = run_chrome(["--dump-dom", *extra, url])
    match = re.search(r"<title>(.*?)</title>", result.stdout, re.S)
    # `--dump-dom` serialises the DOM, so the title comes back HTML-escaped:
    # the probe writes `A>B>C` and the dump reads `A&gt;B&gt;C`.
    title = html.unescape(match.group(1).strip()) if match else ""
    return title, html.unescape(result.stdout)


def dump_title(url: str, extra: tuple[str, ...] = ()) -> str:
    """Load a URL and return the document title it ended up with."""
    return dump_page(url, extra)[0]


def screenshot(url: str, out: Path, size: tuple[int, int], settle_ms: int | None) -> None:
    """Capture one screenshot.

    `settle_ms` uses `--virtual-time-budget`, which is the reliable way to
    let charts and fonts settle — but it freezes media, so the interactive
    captures pass None and rely on the server holding the load event open
    instead.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    flags = [f"--screenshot={out}", f"--window-size={size[0]},{size[1]}"]
    if settle_ms is not None:
        flags.append(f"--virtual-time-budget={settle_ms}")
    run_chrome([*flags, url])


# --------------------------------------------------------------------------
# a server that can hold a request open
# --------------------------------------------------------------------------


class HoldingHandler(http.server.SimpleHTTPRequestHandler):
    """Serves a directory, plus one endpoint that answers slowly on purpose.

    `--dump-dom` and `--screenshot` both fire on the load event. An image the
    server sits on for a few seconds is what keeps that event pending while
    the match underneath it plays out.
    """

    def do_GET(self):  # noqa: N802 - the base class names it
        if self.path.startswith("/hold"):
            match = re.search(r"ms=(\d+)", self.path)
            time.sleep(int(match.group(1)) / 1000 if match else 1.0)
            self.send_response(200)
            self.send_header("Content-Type", "image/gif")
            self.send_header("Content-Length", str(len(PIXEL_GIF)))
            self.end_headers()
            self.wfile.write(PIXEL_GIF)
            return
        super().do_GET()

    def log_message(self, *args):
        pass


def serve(directory: Path):
    """Start a threaded server on a free port and return it with its port."""
    handler = functools.partial(HoldingHandler, directory=str(directory))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, httpd.server_address[1]


SIDECAR_ASSETS = ("demo_match.mp4", "demo_poster.jpg")
"""Files `docs/index.html` references by a same-origin relative path.

Everything else on the page is inlined at build time, but the recorded
match video is not — it ships as a sibling file next to `docs/index.html`
(see `redlight.export`'s docstring for why). A stubbed copy of the page
served from a bare temp directory would 404 on it, so the recorded-match
screenshot would silently show a broken video with no assertion to catch
it. Copying these alongside the stub keeps that path honest.
"""


def stubbed_page(destination: Path) -> None:
    """Write a copy of the built page with the verification stubs injected.

    The stubs go into the head, ahead of everything the arena defines, and
    the held image goes at the end of the body where it delays the load
    event without delaying anything the page does.

    Also copies the page's sidecar assets (the recorded-match video and its
    poster) next to the stub, so a screenshot of the replay actually shows
    what a visitor would see rather than a 404'd `<video>`.
    """
    html = PAGE.read_text(encoding="utf-8")
    html = html.replace("</head>", INJECTION + "</head>", 1)
    html = html.replace("</body>", HOLD_MARKUP.format(ms=HOLD_MS) + "</body>", 1)
    destination.write_text(html, encoding="utf-8")
    for name in SIDECAR_ASSETS:
        source = PAGE.parent / name
        if source.exists():
            shutil.copy(source, destination.parent / name)


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------


def check_selftest(report: list) -> bool:
    """The shipped page re-runs the golden fixtures in a real browser."""
    title = dump_title(f"file://{PAGE}?selftest=1")
    ok = title.startswith("SELFTEST PASS")
    report.append(("fixture selftest", ok, title or "no title"))
    return ok


def calibration_line(dom: str) -> str:
    """The calibration sentence the arena wrote under the meters, if any."""
    match = re.search(r'id="calibrationNote"[^>]*>(.*?)</p>', dom, re.S)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def check_play_path(report: list, shots: bool) -> bool:
    """Play, with a fake camera and a fake pose source, referees both players.

    The stub's two figures are the whole assertion. Player 1 walks and has to
    be called out — a referee that never calls anybody is not a referee.
    Player 2 stands perfectly still through two armed red lights while its
    detection box wobbles by a couple of pixels a frame, and has to survive
    both — a referee that calls a statue is not one either.
    """
    with tempfile.TemporaryDirectory(prefix="rl-site-") as tmp:
        root = Path(tmp)
        stubbed_page(root / "index.html")
        httpd, port = serve(root)
        try:
            base = f"http://127.0.0.1:{port}/index.html?probe=1"
            title, dom = dump_page(f"{base}&theme=dark")
            phases = title.replace("PROBE ", "").split(">")

            reached_countdown = "COUNTDOWN" in phases
            reached_green = "GREEN" in phases
            two_reds = phases.count("RED") >= 2
            mover_called = "out:1" in phases
            still_survived = "out:2" not in phases
            calibrated = "Calibrated to your camera" in dom

            report.append(("play path reaches a countdown", reached_countdown, title))
            report.append(("play path reaches a green light", reached_green, title))
            report.append(("play path covers two red lights", two_reds, title))
            report.append(("a walking player is called out", mover_called, title))
            report.append(
                (
                    "a still player survives both red lights",
                    still_survived,
                    title,
                )
            )
            report.append(
                (
                    "the cutoff is calibrated on the countdown",
                    calibrated,
                    calibration_line(dom) or "no calibration line in the page",
                )
            )

            # The no-camera path has to work too, and it shares the phase
            # machinery — so it is worth its own run rather than an assumption.
            replay_title = dump_title(f"{base}&action=replay&theme=dark")
            replay_phases = replay_title.replace("PROBE ", "").split(">")
            replayed = "replay" in replay_phases and "RED" in replay_phases
            report.append(("replay runs without a camera", replayed, replay_title))

            ok = (
                reached_countdown
                and reached_green
                and two_reds
                and mover_called
                and still_survived
                and calibrated
                and replayed
            )

            if shots:
                for theme in ("light", "dark"):
                    screenshot(
                        f"{base}&theme={theme}",
                        SHOT_DIR / f"arena-playing-{theme}.png",
                        (1400, 1800),
                        None,
                    )
                    screenshot(
                        f"{base}&action=replay&theme={theme}",
                        SHOT_DIR / f"arena-replay-{theme}.png",
                        (1400, 1800),
                        None,
                    )
            return ok
        finally:
            httpd.shutdown()
            httpd.server_close()


SECTIONS = ("arena", "pipeline", "lab", "naive", "results", "engine")

SOLO_STYLE = """
<style>
/* Verification only: show one section at the top of the document.

   Scrolling is not an option — `--screenshot` rasterises the viewport and
   then places it at the document origin, so a scrolled page photographs as a
   blank band where the content above should be. Hiding everything else puts
   the section under review at the top with nothing to scroll past. */
.brandbar, .masthead, footer, main > section:not(#SECTION) { display: none !important; }
main > section { padding-top: 1.2rem !important; }
</style>
"""


def solo_page(section: str, destination: Path) -> None:
    """Write a copy of the page showing only one section."""
    html_text = PAGE.read_text(encoding="utf-8")
    style = SOLO_STYLE.replace("SECTION", section)
    destination.write_text(html_text.replace("</head>", style + "</head>", 1), encoding="utf-8")


def take_screenshots(report: list) -> bool:
    """Every theme at both widths, plus each section on its own.

    Both themes every time. Headless Chrome renders dark by default, so a
    run that only ever looks at the default has never looked at the light
    theme at all.
    """
    written = []
    for name, theme, size in (
        ("desktop-light", "light", (1400, 2400)),
        ("desktop-dark", "dark", (1400, 2400)),
        ("mobile-light", "light", (PHONE_WIDTH, 1600)),
        ("mobile-dark", "dark", (PHONE_WIDTH, 1600)),
    ):
        out = SHOT_DIR / f"{name}.png"
        screenshot(f"file://{PAGE}?theme={theme}", out, size, 6000)
        written.append((name, out.exists() and out.stat().st_size > 5000))

    with tempfile.TemporaryDirectory(prefix="rl-solo-") as tmp:
        for section in SECTIONS:
            page = Path(tmp) / f"{section}.html"
            solo_page(section, page)
            for theme in ("light", "dark"):
                name = f"section-{section}-{theme}"
                out = SHOT_DIR / f"{name}.png"
                screenshot(f"file://{page}?theme={theme}", out, (1400, 1250), 6000)
                written.append((name, out.exists() and out.stat().st_size > 5000))

    ok = all(good for _, good in written)
    report.append(
        (
            "screenshots written",
            ok,
            f"{len(written)} files"
            + ("" if ok else ": missing " + ", ".join(n for n, good in written if not good)),
        )
    )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shots-only", action="store_true", help="skip the assertions, just capture"
    )
    parser.add_argument(
        "--no-shots", action="store_true", help="run the assertions without capturing"
    )
    args = parser.parse_args()

    if not PAGE.exists():
        raise SystemExit(f"{PAGE} does not exist. Run `redlight build-site` first.")

    report: list[tuple[str, bool, str]] = []
    ok = True

    if not args.shots_only:
        ok &= check_selftest(report)
        ok &= check_play_path(report, shots=not args.no_shots)
    if not args.no_shots:
        ok &= take_screenshots(report)
        if args.shots_only:
            check_play_path(report, shots=True)

    width = max((len(name) for name, _, _ in report), default=0)
    for name, good, detail in report:
        print(f"{'PASS' if good else 'FAIL'}  {name.ljust(width)}  {detail}")
    print(f"\nScreenshots in {SHOT_DIR.relative_to(REPO_ROOT)}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
