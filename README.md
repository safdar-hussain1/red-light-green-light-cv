# Red Light, Green Light — a computer-vision referee

Stand in front of a camera and hold still. A referee finds everyone in
frame, tracks them, measures how much of each body changes every tenth of a
second, and calls out anyone who moves while the light is red. It plays in a
browser tab on your own machine, and the same rules run as a Python engine
over a webcam or any video file — the playground game made world-famous by
Squid Game.

[![tests](https://github.com/safdar-hussain1/red-light-green-light-cv/actions/workflows/tests.yml/badge.svg)](https://github.com/safdar-hussain1/red-light-green-light-cv/actions/workflows/tests.yml)
[![Python 3.12 | 3.13](https://img.shields.io/badge/python-3.12%20%7C%203.13-3776ab)](.github/workflows/tests.yml)
[![MIT licence](https://img.shields.io/badge/licence-MIT-2ea44f)](LICENSE)

**[Play it in your browser](https://safdar-hussain1.github.io/red-light-green-light-cv/)** —
no install, no upload, nothing recorded.

![Red light on the benchmark footage: every player detected and tracked, a
crossed box and an OUT label marking the players already
called](reports/figures/hud_red_light.png)

*Red light on the ground-truth footage. Every tracked player is boxed;
crossed boxes labelled OUT are players who already moved.*

---

## What is here

- **A browser arena that plays.** The page opens on a demo match between
  drawn players, called by the same referee that then plays you on your
  webcam: pose detection, tracking, the judge and the game rules all run in
  the tab. Every frame stays on your device: nothing is uploaded, nothing
  is recorded, nothing survives the tab closing.
- **A Python engine and CLI.** `redlight play` referees a webcam or a video
  file with YOLO11n or weights-free HOG detection, draws a HUD, and reports
  how the match ended.
- **A judge that was measured, not guessed.** Movement is scored in
  body-fractions per second on a fixed 0.1 s sampling clock, so a player at
  the back of the room and a player at the front are judged the same, and a
  laptop at 60 fps and a phone at 24 return the same verdicts.
- **The naive designs, built and benchmarked.** Three referees anyone would
  reach for first are implemented properly and scored on the same samples —
  including where they win.
- **Browser/Python parity, bit for bit.** The scoring kernel in the tab and
  the one in Python produce identical numbers on shared golden 96 × 96
  window fixtures. Exact equality, not a tolerance.
- **291 tests** (289 without the two slow ones), and six load-bearing
  claims checked by mutating the mechanism behind each one: **6 of 6
  mutations were caught** by tests that already existed.

---

## Every command

The commands below were run on macOS (Apple silicon) with Python 3.13 while
this README was written, all except the webcam line, and the outputs shown
are real (trimmed where marked). CI runs the same test suite on Linux with
Python 3.12 and 3.13, then builds the page again and fails if `docs/` changes.

### Set up

```bash
git clone https://github.com/safdar-hussain1/red-light-green-light-cv.git
cd red-light-green-light-cv

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[audio,analysis,dev]"

redlight --help                    # lists play, benchmark, make-audio, build-site
```

`python -m redlight` works the same as `redlight`. The extras: `audio` is
pygame for the chant and buzzer (the referee runs silently without it),
`analysis` is scikit-learn, matplotlib and Jupyter for `redlight benchmark`,
the figures and the notebook, and `dev` is pytest. Node.js 18 or newer is
needed only for the browser/Python parity tests, which skip without it.

The YOLO11n weights are committed at `models/yolo11n.pt`, so there is
nothing to download. The fetch script checks them, and downloads them from
the pinned Ultralytics release only if the file is missing:

```bash
python scripts/fetch_weights.py
# already present: .../models/yolo11n.pt (5.6 MB, sha256 0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1)
```

Run every command from the repository root. The weights path,
`models/yolo11n.pt`, is relative to the folder you run in; from anywhere
else Ultralytics first downloads its own copy of `yolo11n.pt` into
`./models/` (the same 5.6 MB file, but it needs a network connection).

> **If `redlight` fails with `ModuleNotFoundError: No module named 'redlight'`
> after the install worked,** Python skipped the editable install's `.pth`
> file. Current Python releases (3.12 and 3.13 both behave this way) skip
> `.pth` files that carry macOS's hidden flag, and inside a folder that
> iCloud Drive syncs (Desktop or Documents) that flag can appear on the
> virtual environment's files a few minutes after the install.
> Clone somewhere iCloud does not sync, or clear the flag with
> `chflags nohidden .venv/lib/python3.*/site-packages/*.pth`, or run from the
> repository root with `export PYTHONPATH=src`. A non-editable
> `pip install .` is not a fix: `redlight build-site` only works from a
> source checkout.

### Test

```bash
pytest                            # 291 passed, one to two minutes
pytest -m "not slow"              # 289 passed, 2 deselected: skips the two YOLO passes over the footage
pytest tests/test_js_parity.py    # the browser/Python parity suite under Node: 17 passed
```

CI runs the whole suite with nothing skipped: it needs no camera, no GPU
and no private data, because the weights and the footage are committed. The
one skip is local: in a copy with no git history (a ZIP download),
`tests/test_framing.py` skips, because it scans `git ls-files`, and `pytest`
reports `275 passed, 1 skipped`.

### Play a match

```bash
redlight play                     # webcam 0, with a window: press S to start, Q to quit
```

That one needs a camera and a desktop session, so it is the one command
here that was not run for this README; `--source 1` picks a second camera.
Everything else runs on the committed footage.

Referee a video file with no window. `--headless` needs `--auto-start N`,
which starts the match once N frames in a row have seen a player:

```bash
redlight play --source data/vtest.avi --headless --auto-start 10
# victory - 5 player(s), 4 survived, 1 eliminated
```

Without `--seed` the light schedule is random, so that line changes from
run to run. The seeded demo match the benchmark publishes always ends the
same way:

```bash
redlight play --source data/vtest.avi --headless --mute \
  --seed 3 --skip 200 --auto-start 20 --max-seconds 40 \
  --countdown 1 --duration 12 --phase-min 1.5 --phase-max 3 --grace 0.6 \
  --metric flow --detector yolo --conf 0.35 --confirm-frames 3
# victory - 9 player(s), 2 survived, 7 eliminated
```

Record the annotated match, or swap in the weights-free detector and the
metric the browser runs:

```bash
redlight play --source data/vtest.avi --headless --auto-start 10 --seed 3 --duration 10 --record match.mp4
# victory - 5 player(s), 4 survived, 1 eliminated
redlight play --source data/vtest.avi --headless --auto-start 10 --seed 3 --duration 10 --detector hog --metric diff
# victory - 5 player(s), 4 survived, 1 eliminated
```

`--record` writes with OpenCV's mp4v codec (`match.mp4` above is 140
frames, 768 × 576, 10 fps) wherever you point it; left in the repository
root it shows up in `git status` as untracked. The HOG run takes about 30
seconds against about 10 for YOLO.

Every `play` flag, with the values the code accepts (`redlight play --help`
prints the same list):

| Flag | Default | Accepts |
|---|---|---|
| `--source` | `0` | a webcam index, a video file path, or a stream URL |
| `--seed` | none | any integer; leave it out for a new light schedule every run |
| `--countdown S` | `3.0` | seconds, 0 or more |
| `--duration S` | `60.0` | seconds of match, 0 or more |
| `--phase-min S`, `--phase-max S` | `2.0`, `5.0` | seconds per light, above 0, min no larger than max |
| `--grace S` | `0.6` | seconds after red turns on before anyone can be called, 0 or more |
| `--metric` | `flow` | `flow` (dense optical flow) or `diff` (changed pixels, the browser's metric) |
| `--threshold` | `0.0767` | the flow cutoff in body-fractions per second, above 0 |
| `--diff-threshold` | `0.1455` | the diff cutoff in body-fractions per second, above 0 |
| `--confirm-frames` | `3` | samples in a row over the cutoff before a call, 1 or more |
| `--smoothing` | `0.5` | weight on the newest sample, above 0 and at most 1 |
| `--detector` | `yolo` | `yolo` (YOLO11n) or `hog` (OpenCV's people detector, no weights) |
| `--conf` | `0.35` | detection cutoff: a 0–1 confidence for YOLO, the raw SVM score for HOG |
| `--max-misses` | `15` | frames a player can go undetected before they leave the arena |
| `--headless` | off | no window; needs `--auto-start` |
| `--record PATH` | none | write the annotated match to an mp4 |
| `--skip N` | `0` | frames to discard before the match clock starts |
| `--auto-start N` | none | start once N frames in a row see a player |
| `--max-seconds S` | none | stop after S seconds of footage and report `aborted` |
| `--mute` | off | no chant, no buzzer |

A value outside those ranges stops the run with exit status 2 and says
which one:

```bash
redlight play --smoothing 0
# error: Configuration errors:
#   - smoothing must be in (0, 1], got 0.0
```

### Reproduce the numbers

```bash
redlight benchmark --fast --out /tmp/bench-fast.json   # 60 frames, about 15 s
redlight benchmark --out /tmp/bench.json               # the full harness, about 2 minutes
```

The benchmark prints nothing and writes one JSON file. Every figure in the
Results section below is read from `reports/benchmark_results.json`.

That file and the committed figures were produced with torch 2.13.0,
torchvision 0.28.0, ultralytics 8.4.92, opencv-python 4.13.0.92, NumPy
2.5.1, scikit-learn 1.9.0 and matplotlib 3.11.0. On those versions a rerun
matches the file byte for byte apart from the `runtime_ms` timings:

```bash
pip install "torch==2.13.0" "torchvision==0.28.0" "ultralytics==8.4.92" \
            "opencv-python==4.13.0.92" "numpy==2.5.1" "scikit-learn==1.9.0" \
            "matplotlib==3.11.0"
redlight benchmark --out /tmp/bench.json
```

A plain install resolves newer releases, and they change a handful of YOLO
detections. With the versions pip picked on 23 September 2026 (torch 2.14.0,
ultralytics 8.4.160, opencv-python 4.14.0.94, NumPy 2.5.3), the harness kept
4407 paired samples instead of 4410, and its midpoint cutoffs came out at
0.0779 (flow) and 0.1541 (diff) instead of 0.0767 and 0.1455. Every flag
rate in the tables below moved by less than 0.06 percentage points and every
AUC by less than 0.0002; the 100.00% frame-rate agreement, the
background-scenario results and the demo match did not move at all.

> **Before a demo:** `redlight benchmark` without `--out` writes over the
> committed results. The timings always change, so
> `test_committed_page_is_up_to_date` fails until you run
> `redlight build-site` again, and on newer package versions
> `tests/test_threshold_sync.py` fails as well, because the shipped cutoffs
> no longer match the file. To put both back:
> `git checkout -- reports/benchmark_results.json docs/index.html`.

### Build and view the page

```bash
redlight build-site                               # wrote docs/index.html
redlight build-site --out /tmp/rl-page/index.html # the same page, somewhere else
```

The build inlines `site/*.css` and `site/*.js` into `site/template.html` and
bakes in the benchmark results, the chant and the golden judge fixtures. It
is deterministic: on an unchanged checkout it rewrites `docs/index.html`
byte for byte and `git status` stays clean. Edit the template and the
`site/` sources, never the built file.

The page is one self-contained file, so it needs no server:

```bash
open docs/index.html              # macOS; xdg-open on Linux
```

A local server is closer to GitHub Pages, and works the same:

```bash
python3 -m http.server 8310 --bind 127.0.0.1 --directory docs
open http://localhost:8310/
```

Add `?selftest=1` to either address and the page replays every golden
fixture and a scripted match through its own JavaScript, then writes the
verdict into the tab title: `SELFTEST PASS n=59`. `?theme=light` or
`?theme=dark` forces a theme.

With Google Chrome or Chromium installed, one script drives the built page
the way a visitor would — the selftest, the demo match the page opens on
(only the two drawn players written to move may be called), a whole match
against a fake camera (a walker who must be called out and a statue who
must survive two red lights), and the recorded replay — then saves
screenshots of every section in both themes to `reports/site/`, which is
gitignored:

```bash
python scripts/verify_site.py
# PASS  fixture selftest                           SELFTEST PASS n=59
# PASS  demo match calls the two movers only       PROBE LOBBY>demo>COUNTDOWN>GREEN>RED>out:2>GREEN>RED>out:4>GREEN>RED>GREEN>VICTORY>end:VICTORY
# PASS  play path reaches a countdown              PROBE LOBBY>camera>COUNTDOWN>GREEN>RED>out:1>GREEN>RED>GREEN>RED
# PASS  play path reaches a green light            (same probe)
# PASS  play path covers two red lights            (same probe)
# PASS  a walking player is called out             (same probe)
# PASS  a still player survives both red lights    (same probe)
# PASS  the cutoff is calibrated on the countdown  Calibrated to your camera: its noise sits well under the line.
# PASS  replay runs without a camera               PROBE LOBBY>replay>COUNTDOWN>GREEN>RED>GREEN>RED>GREEN
# PASS  screenshots written                        18 files

python scripts/verify_site.py --no-shots     # the nine checks only, about two and a half minutes
python scripts/verify_site.py --shots-only   # the screenshots, then the play path again to capture it
```

### Figures, chant and notebook

```bash
python scripts/make_figures.py    # about 90 s; rewrites everything in reports/figures/
redlight make-audio               # wrote assets/doll_song.wav (5.38s)
redlight make-audio --out /tmp/rl-page/chant.wav   # the same bytes, somewhere else
jupyter nbconvert --to notebook --execute notebooks/judge_design.ipynb --output-dir /tmp/nb
```

On the pinned versions above, `make_figures.py` reproduces every committed
figure byte for byte. On newer versions the PNG bytes change even where the
pixels do not, and the ROC curves and HUD stills move slightly with the
detections, so check `git status` afterwards and
`git checkout -- reports/figures` if you did not mean to update them. The
chant is deterministic on any version.

---

## How the judge works

**Movement is measured in body-fractions per second.** A player two metres
from the camera moves more pixels per step than the same player ten metres
away, and a 4K sensor sees more of them than a 480p one. So every player's
box is cropped out of the grayscale frame and resampled to a fixed 96 × 96
window before anything is scored. Distance and resolution divide out, and
the score becomes the fraction of *that player* that changed, per second.

Two metrics run on those windows:

- `flow_score` — dense Farneback flow across the window; the 95th percentile
  of the displacement magnitudes, minus a 0.5 px noise floor, divided by the
  window size and by the elapsed time.
- `diff_score` — the fraction of the window whose brightness moved by more
  than 25 levels, divided by the elapsed time. Plain integer arithmetic, no
  blur, no floats in the threshold, which is what makes bit-exact agreement
  with the browser a reasonable thing to demand.

**Frame rate is handled by a clock, not by arithmetic.** Displacement really
is proportional to the gap between frames, so dividing flow by `dt` genuinely
turns it into a rate — within a design-time operating band of roughly 2.2 to
7 pixels of window displacement, documented on `flow_score` in `judge.py`.
Below that band the fixed sub-pixel deadband takes a disproportionate bite out
of the smaller step; above it the flow estimator saturates and over-reads. The
band's edges are a design-time choice rather than a swept measurement; the
tests exercise interior points of it (2.4 to 6.0 px of window displacement).

A changed-pixel count is different: it grows *faster* than linearly with
displacement, because a bigger step does not merely move more pixels, it
pushes more of them past the change threshold. Dividing by `dt` does not
rescue it. On the synthetic scene the tests use, doubling both the step and
the interval leaves the same real velocity scoring about 72% higher — so the
diff metric is made fair a different way: `FrameSampler` releases a pair only
once 0.1 s of real time has passed, and the interval is simply held fixed.
Fast cameras get more frames, not different verdicts.

**Two guards sit in front of every call.** Scores flicker — a flow estimate
spikes on a lighting change, a detection box wobbles. So each player's score
goes through an exponential moving average weighting the newest sample 0.5,
and the smoothed score has to stay above the cutoff for 3 consecutive
samples before anyone is called out. One sample back at or below the cutoff
resets the streak. On top of that the game gives a 0.6 s grace window after
the light turns red, because a player mid-stride cannot stop in zero time.
The judge and the sampler are both reset on every phase change, so no
baseline survives from one light into the next.

**The cutoffs were measured.** On the benchmark's 4410 paired samples, the
frozen class's 99th percentile is 0.0 on both metrics and the moving class's
1st percentile is 0.1534 (flow) and 0.2910 (diff). The shipped cutoff is the
midpoint of those two edges — the point furthest from both classes:
**0.0767** for flow, **0.1455** for diff. Applied to the samples they were
derived from, they flag 0.00% of frozen samples and 99.59% (flow) / 99.52%
(diff) of moving ones.

---

## Results

Everything below comes from `reports/benchmark_results.json`, produced by
`redlight benchmark` on `data/vtest.avi` — 795 frames of pedestrians
crossing a courtyard at 10 fps, YOLO11n detection at conf 0.35, seed 1729.
The sampler released 794 pairs. Every walker the tracker held becomes a
*moving* sample (4410 of them); each is paired with a *frozen* twin — the
same frame held in place with seeded gaussian sensor grain, sigma 2, on top
(4410). On the package versions listed under
[Reproduce the numbers](#reproduce-the-numbers), reruns are byte-identical
apart from the timings.

### Five designs, same samples

![ROC curves for the five referee designs](reports/figures/roc_curves.png)

| Design | Scope | AUC | Cutoff used | Frozen flagged | Moving flagged |
|---|---|---|---|---|---|
| `brightness_count` | per box | 0.4998 | 7000 px | 0.50% | 0.50% |
| `frame_diff_area` | **whole frame** | 1.0000 | 7000 px | 0.00% | 35.31% |
| `raw_flow_max` | per box | 0.9996 | 2.0 px | 0.05% | 96.15% |
| `flow_score` | per box | 0.9995 | **0.0767** | **0.00%** | **99.59%** |
| `diff_score` | per box | 0.9995 | **0.1455** | **0.00%** | **99.52%** |

Read that table with three things in mind. `brightness_count` sits at chance
because it never compares two frames — its frozen and moving rates are
identical to three decimals, which is the clearest possible statement that
it has no temporal term. `frame_diff_area` takes no box at all, so every
player in a pair gets the same number; its row is a statement about the
footage, not about a player, and its failure shows up in the background
scenario below. And the "cutoff used" column is not decoration: each naive
design is judged at the round number somebody reaching for it would reach
for first, while the last two rows are judged at the cutoffs the engine
actually ships — the benchmark-selected 0.0767 and 0.1455 — so those two rows
are what the shipped referee does to these samples, not what some other
setting would have done.

![Moving vs. frozen score distributions](reports/figures/score_distributions.png)

Both normalized metrics score exactly 0.000 on all 4410 frozen samples
(frozen max = 0.0).

### The same players, three apparent sizes

![Flag rates for each design at 0.5x, 1x and 1.667x](reports/figures/resolution_sweep.png)

Moving flagged / frozen flagged, at fixed cutoffs, over the 4368 samples
scoreable at all three scales:

| Design | 0.5x | 1x | 1.667x |
|---|---|---|---|
| `brightness_count` | 0.00% / 0.00% | 0.50% / 0.50% | 24.22% / 24.24% |
| `raw_flow_max` | 75.23% / 0.09% | 96.15% / 0.05% | 99.20% / 0.02% |
| `flow_score` | 98.88% / 0.00% | 99.59% / 0.00% | 99.68% / 0.00% |
| `diff_score` | 99.15% / 0.00% | 99.52% / 0.00% | 99.40% / 0.00% |

`raw_flow_max` misses 21 points of walkers at half size that it catches at
native resolution: the same stride, fewer raw pixels, below the cutoff.
Both normalized metrics hold within about a point across a 3.3x span of
apparent size — 11x in pixel area — and never flag a held frame at any scale.

### Frame rate changes nothing

The same footage delivered on a 10 fps clock and on a 30 fps clock (each
frame offered three times), compared over the content pairs both arms
released:

| Metric | Pairs compared | dt mismatch | Decisions compared | Decisions agree |
|---|---|---|---|---|
| `flow_score` | 794 | 0.00% | 4410 | **100.00%** |
| `diff_score` | 794 | 0.00% | 4410 | **100.00%** |

That is the payoff for the fixed sampling clock. Both arms release all 794
pairs at an identical 0.1 s, and every one of 4410 decisions agrees.

### A still player in a busy scene

One player crop (82 × 204) is frozen at a fixed box and composited into 230
frames of live courtyard traffic, with fresh sensor grain every frame. Two
variants:

| | Traffic passes by | Traffic walks through |
|---|---|---|
| Frames where a walker crossed the box | 40 | 177 |
| Whole-frame `frame_diff_area` flags the still player | **51.09%** of samples, first at **0.8 s** | 69.43%, first at 0.5 s |
| Per-box `flow_score` / `diff_score` | **0.00% / 0.00%** | 69.87% / 73.80% |
| Per-box, counting only crossing samples | **0.00% / 0.00%** | **89.77% / 94.89%** |

The left column is the headline: whole-frame differencing charges a
motionless player for everybody else in the shot, crossing its cutoff within
a second and staying there for half the run, while the same player scored
inside their own box reads exactly zero.

The right column is the limitation, published rather than buried. Scoring a
box means scoring whatever is inside it, and a body walking **through** your
box puts moving pixels inside it: 89.77% of the crossing samples flag a
motionless player on flow, and 94.89% on diff. See
[Limitations](#limitations) for the two caveats that make those a ceiling
rather than precise figures.

### Runtime

![Per-call runtime, median over warm calls](reports/figures/runtime.png)

Median per call, Apple Silicon, single-threaded, 768 × 576 frames:

| Stage | ms |
|---|---|
| YOLO11n detection, per frame | 32.7 |
| HOG detection, per frame | 114.9 |
| `flow_score`, per player | 0.85 |
| `diff_score`, per player | 0.036 |

HOG is the **weights-free classical option** — no model file to fetch — and
on this footage at these settings it is about 3.5x *slower* than YOLO11n, not
faster. Judging costs almost nothing next to finding people, which is why
the browser can afford the diff metric on four players at 10 samples a
second.

### The demo match

![The seeded demo match on the ground-truth footage](reports/figures/demo_match.gif)

One seeded run of the same `redlight play` pipeline, starting 200 frames in
so the courtyard is busy: 9 players registered, 1 lost from the arena at
0.8 s, 5 eliminated for moving at 2.7 s and 1 more at 3.3 s, 2 survivors,
outcome **victory**. The starting offset and the seed were the only things
chosen; the clustering is real, because a dozen people walking across a
courtyard are all moving when the light turns.

---

## Limitations

Every one of these is measured or reproducible, not hypothetical.

- **Somebody crossing your box can put you out.** The per-box judge protects
  a still player from traffic elsewhere in the shot (0.00%), but not from a
  body passing in front of them: 89.77% of crossing samples flag a
  motionless player on flow, and 94.89% on diff. Two caveats make those a
  ceiling rather than exact figures — the occluder is the walker's bounding
  box rather than a silhouette, so it drags some background motion across
  with it, and the measurement is raw per-sample scoring without the
  smoothing and 3-in-a-row confirmation a brief pass-through may never
  clear. The direction is not in doubt, only the magnitude.
- **Flow saturates on fast motion.** `flow_score`'s operating band is
  roughly 2.2 to 7 pixels of window displacement — a design-time band
  documented on the function, not a swept measurement; the tests exercise
  interior points of it (2.4 to 6.0 px). Beyond it Farneback over-reads, so
  a fast step scores higher than its true displacement. Over-reading errs
  toward calling movement, which is the safe direction for a referee, but it
  means the number stops being a literal body-fraction at the fast end.
- **The frozen class is held frames plus sigma-2 grain.** Sigma 2 sits an
  order of magnitude below the diff metric's 25-level change threshold, so
  `diff_score` reading exactly 0.0 on frozen samples is *structural, not
  evidence*. The frozen result worth trusting is flow's, whose deadband is
  sub-pixel and was not chosen against this noise level.
- **The moving class is people walking continuously.** Its 1st percentile is
  not the faintest motion a real player can make — someone shifting their
  weight scores far below anyone in this footage. The measured cutoffs
  therefore err strict, and they were taken on stable crops rather than on
  anyone standing in a living room. The browser treats the measured number
  as its strictest setting for that reason: ruthless is 1.0x, standard —
  the default — is 2.0x, forgiving is 4.0x, and the countdown lifts the
  cutoff further if the player's own camera turns out to be noisier than
  the footage was.
- **A wobbling box used to be scored as a moving player.** Pose landmarks
  shift a couple of pixels every frame even on somebody holding perfectly
  still, and a crop window cut from that box shifts with it. The browser now
  smooths each player's scoring rectangle and pins it in place through the
  countdown and every red light, so a still player's window does not move.
  `docs/DESIGN.md` § 7 has the mechanism and the numbers.
- **Identity in the browser is overlap matching, not re-identification.**
  Two players who swap places while crossing will swap ids, and the referee
  would then judge them as each other. In a game where everyone is standing
  still that situation does not arise, but it is a real bound on the
  tracker.
- **The pose runtime is loaded without a subresource-integrity hash.** It is
  the one documented exception on the page: MediaPipe's loader resolves its
  own wasm binary and model URLs at run time, so there is no fixed set of
  bytes to hash. It is pinned to an exact version, fetched only after you
  press play and grant a camera, and nothing else on the page depends on it
  — every published number, the referee lab, the replay and the selftest run
  with it never loading.
- **The recorded replay is only as fine as its frames.** Under *Watch a
  recorded match*, the light, doll, countdown and roster beside the video
  are worked out from the video's own playback position and the match's
  recorded light schedule, so they follow a slow start, a pause or a scrub.
  The recording is 10 fps and the lights changed between frames, so within
  a tenth of a second of a change the sidebar can show the new light one
  frame before the picture does.
- **The benchmark is one scene.** `data/vtest.avi` is a single outdoor
  courtyard at 10 fps. Every number above describes the judge on that
  footage. Nothing here has been measured across lighting conditions,
  indoor scenes, or phone cameras.

---

## Privacy and ethics

Every frame is processed on the device it was captured on. The browser
arena opens the camera in the tab, runs pose detection, scoring and the game
rules locally, and never sends a frame anywhere — nothing is uploaded,
nothing is recorded, and nothing is kept after the tab closes. The Python
engine writes video only when you explicitly pass `--record`.

This is a referee for a party game. It measures how much of a bounding box
changed between two frames; it does not identify anybody, does not store
anything about anybody, and is not a surveillance tool. Please do not point
it at people who have not chosen to play.

---

## Repository structure

```
src/redlight/        the engine: detection, tracking, judge, game, HUD, CLI,
                     audio, the benchmark harness, and the site build
site/                the browser arena: judge, game, pose, arena, the demo
                     match, lab, charts, chant, doll, confetti, the page's
                     two views, styles, typefaces, page template
docs/index.html      the built page, served by GitHub Pages
docs/                beside it: favicon.svg, og-image.png (the share image),
                     sitemap.xml, and the recorded match (demo_match.mp4 and
                     its poster)
docs/DESIGN.md       the design card: pipeline contracts, judge math,
                     threshold selection, claims and their guarding tests
notebooks/           judge_design.ipynb — how the judge was arrived at
reports/             benchmark_results.json and the committed figures
scripts/             figure rendering, weight fetching, site verification
tests/               291 tests, including browser/Python parity, framing and
                     the page's metadata
data/, models/       ground-truth footage and the YOLO11n weights
assets/              the synthesised chant, doll_song.wav
.github/workflows/   CI: the test suite on Python 3.12 and 3.13
```

## Built with

Python (tested on 3.12 and 3.13; the package declares 3.10+), NumPy,
OpenCV, Ultralytics YOLO11n, scikit-learn and matplotlib for the benchmark
and figures, pygame for optional sound. The
browser side is plain JavaScript with no build step — MediaPipe Tasks Vision
for pose and Chart.js for the result charts, both pinned — set in Black Han
Sans and IBM Plex Sans KR. Tests are pytest, with Node driving the parity
suite.

## Licence

This code is MIT licensed — see [LICENSE](LICENSE).

Three things in the repository carry other terms:

- `data/vtest.avi` is an OpenCV sample video, BSD licensed.
- The typefaces in `site/fonts/`, Black Han Sans and IBM Plex Sans KR, are
  under the SIL Open Font License 1.1; `site/fonts/NOTICE.md` lists each
  file and its copyright.
- `models/yolo11n.pt` is an Ultralytics YOLO11 model. Ultralytics is an
  AGPL-3.0 runtime dependency; if you redistribute or host something built
  on those weights, read their licence and comply with it.

## Read more

- **[docs/DESIGN.md](docs/DESIGN.md)** — the design card: every pipeline
  stage's contract, the judge's formulas, how the thresholds were chosen,
  the invariance evidence, the six claims and the tests that guard them.
- **[notebooks/judge_design.ipynb](notebooks/judge_design.ipynb)** —
  designing a motion judge you cannot fool, walked through against the
  measured results.
</content>
