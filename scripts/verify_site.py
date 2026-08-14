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
real game machine. The page records the phases it passed through in its own
title under `?probe=1`, which is the one piece of state a `--dump-dom` run
can read back.

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

HOLD_MS = 9000
"""How long the server holds the injected image open, delaying the load event.

Long enough for registration (1 s), the countdown (3 s) and a green light to
arrive with room to spare — the phase lengths are drawn from a PRNG, so the
margin is deliberate.
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
 * Two substitutions, both at the edge of the system: the camera, and the
 * pose runtime. Everything between them and the verdict is the real code.
 */
(function () {
  var canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 480;
  var ctx = canvas.getContext("2d");
  var frame = 0;

  // Synthetic footage: a textured figure that actually moves, so the diff
  // score is a real number rather than a flat zero.
  function draw() {
    frame += 1;
    ctx.fillStyle = "#1d2b33";
    ctx.fillRect(0, 0, 640, 480);
    ctx.fillStyle = "#33454f";
    ctx.fillRect(0, 360, 640, 120);

    var x = 90 + Math.sin(frame / 11) * 40;
    ctx.fillStyle = "#d9cfbe";
    ctx.fillRect(x, 150, 110, 240);
    ctx.beginPath();
    ctx.arc(x + 55, 128, 40, 0, 6.3);
    ctx.fill();
    for (var i = 0; i < 260; i += 1) {
      ctx.fillStyle = i % 2 ? "#2b2b2b" : "#f2f2f2";
      ctx.fillRect(x + Math.random() * 110, 150 + Math.random() * 240, 4, 4);
    }

    ctx.fillStyle = "#c9d6cf";
    ctx.fillRect(400, 160, 100, 230);
    ctx.beginPath();
    ctx.arc(450, 138, 38, 0, 6.3);
    ctx.fill();

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
  // the play path can be exercised with no network at all.
  window.RL_POSE_FACTORY = function () {
    return Promise.resolve({
      detect: function (video, timestampMs) {
        var s = timestampMs / 1000;
        var wobble = Math.sin(s * 5) * 0.045;
        return [
          { x1: 0.11 + wobble, y1: 0.22, x2: 0.36 + wobble, y2: 0.86 },
          { x1: 0.60, y1: 0.24, x2: 0.82, y2: 0.84 },
        ];
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


def dump_title(url: str, extra: tuple[str, ...] = ()) -> str:
    """Load a URL and return the document title it ended up with."""
    result = run_chrome(["--dump-dom", *extra, url])
    match = re.search(r"<title>(.*?)</title>", result.stdout, re.S)
    # `--dump-dom` serialises the DOM, so the title comes back HTML-escaped:
    # the probe writes `A>B>C` and the dump reads `A&gt;B&gt;C`.
    return html.unescape(match.group(1).strip()) if match else ""


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


def stubbed_page(destination: Path) -> None:
    """Write a copy of the built page with the verification stubs injected.

    The stubs go into the head, ahead of everything the arena defines, and
    the held image goes at the end of the body where it delays the load
    event without delaying anything the page does.
    """
    html = PAGE.read_text(encoding="utf-8")
    html = html.replace("</head>", INJECTION + "</head>", 1)
    html = html.replace("</body>", HOLD_MARKUP.format(ms=HOLD_MS) + "</body>", 1)
    destination.write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------


def check_selftest(report: list) -> bool:
    """The shipped page re-runs the golden fixtures in a real browser."""
    title = dump_title(f"file://{PAGE}?selftest=1")
    ok = title.startswith("SELFTEST PASS")
    report.append(("fixture selftest", ok, title or "no title"))
    return ok


def check_play_path(report: list, shots: bool) -> bool:
    """Play, with a fake camera and a fake pose source, reaches a green light."""
    with tempfile.TemporaryDirectory(prefix="rl-site-") as tmp:
        root = Path(tmp)
        stubbed_page(root / "index.html")
        httpd, port = serve(root)
        try:
            base = f"http://127.0.0.1:{port}/index.html?probe=1"
            title = dump_title(f"{base}&theme=dark")
            phases = title.replace("PROBE ", "").split(">")

            reached_countdown = "COUNTDOWN" in phases
            reached_green = "GREEN" in phases
            report.append(("play path reaches a countdown", reached_countdown, title))
            report.append(("play path reaches a green light", reached_green, title))

            # The no-camera path has to work too, and it shares the phase
            # machinery — so it is worth its own run rather than an assumption.
            replay_title = dump_title(f"{base}&action=replay&theme=dark")
            replay_phases = replay_title.replace("PROBE ", "").split(">")
            replayed = "replay" in replay_phases and "RED" in replay_phases
            report.append(("replay runs without a camera", replayed, replay_title))

            ok = reached_countdown and reached_green and replayed

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
