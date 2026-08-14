# Design card — the motion judge

How the referee decides that somebody moved, why it is built this way, what
was measured, and where it breaks. Every number here comes from
`reports/benchmark_results.json` (produced by `redlight benchmark` on
`data/vtest.avi`) or from a named test.

The product overview lives in [../README.md](../README.md); the narrative
walkthrough lives in
[../notebooks/judge_design.ipynb](../notebooks/judge_design.ipynb). This
page is the engineering detail underneath both.

---

## 1. Pipeline

```
                     Python engine (redlight play)          Browser arena (docs/index.html)
                     ────────────────────────────────       ─────────────────────────────────
  01  frames         FrameSource: camera index or file      <video> from getUserMedia
                     BGR frame + timestamp (s)              canvas frame + performance.now()
                              │                                        │
  02  find people    YoloDetector / HogDetector             MediaPipe PoseLandmarker
                     Detection(x, y, w, h, conf)            up to 4 people, landmarks
                              │                                        │
  03  identity       Tracker: greedy IoU match,             BoxTracker: greedy overlap match,
                     stable ids, max_misses=15              landmark box grown 15%
                              │                                        │
  04  pace           FrameSampler.offer(gray, ts)  ──────── identical port, same 0.1 s clock
                     releases (prev, cur, dt) or None
                              │
  05  score          crop_window: clamp box → resize to     canvas drawImage → 96 x 96 grey
                     96 x 96 (INTER_AREA) → 3x3 blur
                     flow_score / diff_score                diffScoreWindow  ← pinned by fixtures
                              │                                        │
  06  decide         MotionJudge: EMA 0.5, then 3           identical port
                     consecutive samples over cutoff
                              │                                        │
  07  rules          Game: phases, 0.6 s grace,             identical port (game.js)
                     registration closed at countdown
                              │                                        │
                     HUD overlay + MatchReport              DOM overlay + end card
```

### Stage contracts

| # | Stage | Input | Output | Contract |
|---|---|---|---|---|
| 01 | `sources.FrameSource` | camera index / path / URL | `(BGR frame, timestamp_s)` or `None` | Raises `SourceError` if the source will not open. Timestamps are the source's own clock, monotonic. |
| 02 | `detection.make_detector` | BGR frame | `list[Detection]` | Person class only, filtered by `conf`. HOG's raw SVM decision value is thresholded on its own scale, not on a probability. |
| 03 | `tracking.Tracker` | detections for one frame | `list[Track]` with stable ids | Greedy IoU matching, best pair first. A track survives `max_misses` frames without a match, then is dropped and its player leaves the arena. Ids never repeat. |
| 04 | `judge.FrameSampler` | `(gray, ts)` every frame | `(prev_gray, cur_gray, dt)` or `None` | Releases only when `ts - last_released_ts >= 0.1 s - 1e-9`. Always compares against the last *released* frame, never the last offered one. |
| 05 | `judge.crop_window` | gray frame + box | `96 x 96 uint8` or `None` | Box clamped to the frame; `None` when the visible part is under 8 px on either side. INTER_AREA resample, then a 3x3 Gaussian blur. |
| 05 | `judge.flow_score` / `diff_score` | two grays, box, dt | body-fractions per second, or `None` | `None` on an unusable crop or a non-positive dt. `diff_score_window` *raises* on a bad dt or a wrong-shaped window instead — it is the function the browser port is held against, so a wrong input is a bug to surface. |
| 06 | `judge.MotionJudge.update` | `(track_id, score)` | `bool` | True only after the smoothed score has been **strictly** above the cutoff for `confirm_frames` samples in a row. One sample at or below resets the streak. |
| 07 | `game.Game.update` | `now`, moving ids, lost ids | `list[Event]` | Pure logic: no video, no wall clock, no sleeping. Only players registered when the countdown opened can ever be eliminated or win. A late `update` replays every phase boundary it skipped, in order. |

`app.run` wires 01–07 together. Every frame is detected and tracked (the
tracker needs to see all of them to keep ids stable), but a pair is scored
only when the sampler releases one *and* the current red light is armed.
Both the judge and the sampler are reset on every phase change, so no
baseline survives from one light into the next — without that, the first
sample of a new red light compares against a frame from the old one and
manufactures a spike.

---

## 2. The judge's math

Let `B` be a player's detection box in frame `t`, `W = 96` the analysis
window, `dt` the seconds since the last released frame.

**Window.** `w_t = blur3x3(resize_INTER_AREA(gray_t[B], W x W))`.

Resampling to a fixed window is the whole trick: the score becomes a
fraction of *that player's* apparent size, so distance from the camera and
sensor resolution divide out. The blur takes the edge off single-pixel
grain before anything is scored.

**Flow.**

```
flow      = Farneback(w_{t-1}, w_t)            pyr_scale 0.5, levels 3,
mag       = |flow|                             winsize 15, iterations 3,
p95       = percentile(mag, 95)                poly_n 5, poly_sigma 1.2

flow_score = max(p95 - 0.5 px, 0) / 96 / dt
```

The 95th percentile tracks the fastest-moving part of a player — a hand, a
shifting foot — while ignoring the handful of outliers any flow estimator
produces. `FLOW_NOISE_FLOOR_PX = 0.5` is a sub-pixel deadband: grain jitters
flow by a fraction of a pixel every frame, and dividing that by a small `dt`
would amplify exactly that jitter into an elimination.

**Diff.**

```
changed    = count(|w_{t-1} - w_t| > 25)       DIFF_PIXEL_DELTA = 25
diff_score = changed / 9216 / dt
```

Deliberately plain integer arithmetic on two uint8 windows — no blur inside
the kernel, no resize, no floating-point threshold — because this is the
function the browser has to reproduce bit for bit.

**Units.** Both are body-fractions per second: what fraction of the player's
own apparent size changed, per second of real time.

### What `/dt` does and does not buy

| | Divides out with `/dt`? | Why |
|---|---|---|
| `flow_score` | Yes, within a band | Displacement is proportional to elapsed time, so displacement / time is a rate. Honest across roughly **2.2–7 px** of window displacement. |
| `diff_score` | **No** | A changed-pixel count is superlinear in displacement: a bigger step does not merely move more pixels, it pushes more of them past the 25-level delta. |

Outside the flow band, both edges are documented rather than hidden: below
~2.2 px the fixed sub-pixel deadband takes a disproportionate bite out of
the smaller step; above ~7 px Farneback saturates and over-reads, so a fast
step scores higher than its true displacement. Over-reading errs toward
calling movement — the safe direction for a referee — but the score stops
being a literal body-fraction at the fast end.

The diff metric's non-invariance is pinned as a characterisation test rather
than left implicit (`tests/test_judge.py::test_diff_score_is_not_dt_invariant`):
on the synthetic scene, 20 px over 0.2 s and 10 px over 0.1 s are the same
real velocity, and the first scores about 72% higher than the second. Nobody
should be able to read the `/dt` in the formula as a promise of frame-rate
independence. The fix is the sampler, not arithmetic.

---

## 3. The sampling clock

```python
if dt < interval_s - TIMESTAMP_TOLERANCE_S:   # 1e-9
    return None
```

`FrameSampler` holds frames back and releases a pair only once at least
0.1 s has passed since the pair it last released. That is what makes the
diff metric fair across frame rates: rather than trying to rescale a count
that does not scale, the interval it is measured over is held fixed.

**Measured `dt` is not exactly 0.1 s, and the design accounts for it.** The
sampler releases the *first frame at or past* the boundary, so the released
interval lands in `[interval, interval + one frame period)` — up to 0.133 s
on a 30 fps camera, up to 0.117 s at 60 fps. Both metrics divide by the
`dt` they were actually handed, so that jitter is measured rather than
assumed away; the residual it leaves is what `fps_sweep`'s `dt_mismatch_pct`
reports (0.00% on the benchmark footage, where both arms land on identical
0.1 s intervals).

**The tolerance is load-bearing, not cosmetic.** A camera running at a clean
multiple of the interval — 10 fps against a 0.1 s clock is the obvious case
— produces timestamps like `i / 10.0`, and in binary floating point the gap
between two of those lands a fraction of an ulp under 0.1 about half the
time. Compared exactly, the sampler rejects every other frame that is
precisely on the boundary and paces itself at half the rate it was asked
for, with the interval alternating between one and two frame periods. That
is not hypothetical: it is what this harness did before the tolerance was
added, and fixing it moved the released-pair count from 543 to 794 of 795
frames and the frame-rate agreement from 99.74% to 100.00%. A nanosecond is
orders of magnitude below any real frame period, so the tolerance cannot
admit a genuinely early frame.

---

## 4. Choosing the cutoffs

One rule, always:

```
threshold = (frozen_p99 + moving_p1) / 2      if frozen_p99 < moving_p1
            none                              otherwise
```

The frozen class's 99th percentile and the moving class's 1st percentile are
the two edges that matter: a cutoff between them clears the noisiest held
frame and still catches all but the faintest walker. Their midpoint is the
point furthest from both, which is the whole point of choosing one — every
other position sits nearer to one class than the other and buys nothing for
it. A geometric midpoint was considered and removed: it collapses to zero
whenever the frozen edge is zero, which is exactly what a working deadband
produces, so it would be a rule that stops working precisely when the judge
is working.

If the two distributions overlap, the harness reports `separated: false` and
picks nothing, rather than publishing a number that separates nothing.

### The corridor, as measured

| Metric | Frozen p99 | Frozen max | Moving p1 | Separated | Chosen | Shipped |
|---|---|---|---|---|---|---|
| `flow_norm` | 0.0 | 0.0 | 0.153404 | yes | **0.076702** | 0.0767 |
| `diff_norm` | 0.0 | 0.0 | 0.290994 | yes | **0.145497** | 0.1455 |

Applied to the 4410 + 4410 samples they were derived from: **0.00%** of
frozen samples flagged; **99.59%** (flow) and **99.52%** (diff) of moving
ones. `tests/test_threshold_sync.py` reads the JSON and asserts
`GameConfig`'s defaults still equal those values rounded to 4dp, so a rerun
on new footage that nobody propagated fails loudly.

**The moving class is people walking continuously.** Its 1st percentile is
not the faintest real motion a player can make — someone shifting their
weight scores far below anyone in this footage — so these cutoffs err
strict. The browser offers `forgiving` (2x) and `ruthless` (0.6x)
multipliers on top of the measured one for that reason.

### Hysteresis

| Constant | Value | Why |
|---|---|---|
| `smoothing` | 0.5 | Weight on the newest sample in the per-player EMA. The first score a player ever gets seeds their average — starting from zero would hand everyone a free sample of movement. |
| `confirm_frames` | 3 | Consecutive samples strictly above the cutoff before a call. One sample at or below resets the streak to zero. At a 0.1 s clock that is 0.3 s of sustained motion. |
| `grace_s` | 0.6 | Red light is unarmed for this long after it turns: a player mid-stride cannot stop in zero time. |
| `max_misses` | 15 | Frames a registered player may go undetected before their track is dropped and they leave the arena. |

`MotionJudge.update` compares `ema > threshold`, strictly — a score sitting
exactly on the cutoff is not a call — and the benchmark's `_flagged_pct`
uses the same strict comparison so the published rates match what the judge
would actually do.

---

## 5. Invariance evidence

| Claim | Evidence | Result |
|---|---|---|
| Distance / resolution does not change the verdict | `resolution_sweep`, 4368 samples at 0.5x / 1x / 1.667x | `flow_norm` 98.40 / 99.27 / 99.31% moving flagged, 0.00% frozen at every scale; `diff_norm` 96.98 / 97.39 / 97.25%, 0.00% frozen |
| ... and the naive designs do not hold it | same sweep | `brightness_count` 0.00% → 24.22% moving as the footage grows, flagging frozen at the same rate; `raw_flow_max` drops 21 points of walkers at half size |
| ... on a synthetic scene with an exact displacement | `test_score_is_resolution_invariant` | Scores at 0.5x / 1x / 2x within a factor of 1.35, each at least 10x the still-player baseline |
| Frame rate does not change the verdict | `fps_sweep`, 10 fps vs 30 fps through `FrameSampler` | 794 pairs compared, dt mismatch 0.00%, **4410 of 4410 decisions agree (100.00%)** on both metrics |
| ... for flow, by arithmetic | `test_score_is_framerate_invariant` | Half the step over half the interval lands within 30% of the full step |
| ... for diff, by the clock only | `test_diff_score_is_not_dt_invariant`, `test_diff_verdicts_framerate_invariant_via_sampler` | The raw score moves by more than 30% (the characterisation); the verdicts through the sampler do not move at all |
| Sensor grain is not movement | `classifiers`, `test_sensor_noise_stays_under_deadband` | Both normalized metrics score exactly 0.000 on all 4410 frozen samples (frozen max 0.0); on the synthetic scene flow < 0.02 and diff == 0.0 against a sigma-2 noisy frame |
| A spectator elsewhere in the shot cannot put you out | `background_scenario.isolated`, 229 samples with 40 crossing frames | Whole-frame differencing flags the still player in **51.09%** of samples, first at **0.8 s**; per-box flow and diff flag **0.00%**, including 0.00% over the 39 crossing samples |
| The browser scores identically | `tests/test_js_parity.py`, golden 96 x 96 fixtures under node | Bit-identical, exact equality rather than a tolerance; re-runnable in any visitor's browser via `?selftest=1` |

---

## 6. Claims, guarding tests, and the mutations that were run

Six claims carry the product. Each was verified by breaking the mechanism
behind it and confirming that tests which already existed caught the break —
the mutation was applied, the suite run, then reverted and the tree
confirmed clean. Baseline and final state both: `234 passed, 2 deselected`.

| # | Claim | Mutation applied | Targeted failing tests | Suite result |
|---|---|---|---|---|
| 1 | Resolution / distance invariance | `crop_window` skips resampling to the fixed `WINDOW` and resizes the crop to its own size | `test_score_is_resolution_invariant`, `test_crop_window_shape_and_dtype`, `test_frozen_scores_zero`, `test_sensor_noise_stays_under_deadband`, `test_diff_score_is_not_dt_invariant`, `test_diff_verdicts_framerate_invariant_via_sampler`, `test_raw_flow_max_grows_with_scale_while_flow_score_holds` | 9 failed, 190 passed, 35 errors |
| 2a | Frame-rate invariance (flow) | `flow_score` drops the `/ dt` division | `test_score_is_resolution_invariant`, `test_score_is_framerate_invariant` | 2 failed, 232 passed |
| 2b | Frame-rate invariance (clock) | `FrameSampler.offer` releases on every offer, ignoring `interval_s` | `test_diff_verdicts_framerate_invariant_via_sampler`, `test_frame_sampler_holds_the_interval`, `test_frame_sampler_still_holds_back_a_genuinely_early_frame`, `test_js_frame_sampler_matches_python` | 4 failed, 230 passed |
| 3 | Noise deadband | `FLOW_NOISE_FLOOR_PX = 0.0` and `DIFF_PIXEL_DELTA = 0` | `test_frozen_scores_zero`, `test_sensor_noise_stays_under_deadband`, `test_chosen_thresholds_clear_every_held_frame`, `test_a_frozen_player_is_never_flagged_when_nothing_crosses_them`, `test_fixtures_reproduce_python_exactly`, and 6 more | 11 failed, 223 passed |
| 4 | No phantom spike at a phase change | `judge.reset()` and `sampler.reset()` removed from the `PHASE_CHANGED` handler | `test_reset_call_count_matches_phase_changed_event_count` | 1 failed, 233 passed |
| 5 | Per-box protection (the spectator claim) | `flow_score` and `diff_score` ignore `box` and score the whole frame | `test_degenerate_box_returns_none`, `test_chosen_thresholds_clear_every_held_frame`, `test_a_frozen_player_is_never_flagged_when_nothing_crosses_them`, `test_the_isolated_variant_had_walkers_it_painted_over`, and 3 more | 7 failed, 227 passed |
| 6 | Browser parity | `DIFF_PIXEL_DELTA` changed 25 → 24 in `site/judge.js` | `test_js_diff_score_matches_python_on_every_fixture`, `test_committed_page_is_up_to_date` | 2 failed, 232 passed |

**Survivors: 0 of 6.** Every mutation was caught on its first attempt by at
least one existing test; no missing test had to be written, and nothing was
committed.

---

## 7. Failure modes and limitations

### Occlusion — the honest ceiling

`background_scenario.occluded` moves the same frozen player to the busiest
spot in the frame and redraws any tracked box crossing theirs from the live
frame, so walkers pass **in front of** them:

| | Traffic passes by (`isolated`) | Traffic walks through (`occluded`) |
|---|---|---|
| Box | (181, 349) 82 x 204 | (592, 200) 82 x 204 |
| Frames / samples | 230 / 229 | 230 / 229 |
| Crossing frames | 40 | 177 |
| Whole-frame `frame_diff_area` | 51.09%, first at 0.8 s | 69.43%, first at 0.5 s |
| Per-box flow / diff, all samples | 0.00% / 0.00% | 65.94% / 52.84% |
| Per-box flow / diff, crossing samples only | 0.00% / 0.00% | **84.66% / 67.61%** |

Scoring a box means scoring whatever is inside it. Two caveats make 84.66%
a ceiling rather than a precise figure: the occluder is the walker's
*bounding box*, not a silhouette, so it drags some background motion across
the frozen box with it; and this is raw per-sample scoring, without the EMA
and 3-in-a-row confirmation a brief pass-through may never clear. The
direction is not in doubt, only the magnitude. The registration rule still
holds separately: an unregistered spectator cannot be eliminated at all,
whatever they walk in front of.

### The exclusion that flatters flow, and by how much

A walker whose box centre moved less than 0.5 px between the two frames of a
pair is not honest evidence of motion, so those samples are dropped from
both classes (the classes are paired sample-for-sample, so dropping one
drops its twin). On this footage that is **67 of 4618** candidates, 1.5%.
Their median `flow_norm` is **0.306**, and **62.69%** of them would have
been flagged at the shipped cutoff anyway.

So the exclusion does flatter the metric slightly: including those samples
would put `flow_norm`'s moving flag rate at roughly **98.7%** instead of the
published **99.27%**. They are still dropped, because a box that did not
measurably move is not evidence that the player did — but the size of the
effect is published rather than left implicit. `no_prior_box` (141) and
`unusable_crop` (0) are reported the same way.

### The frozen class is structurally easy for diff

The frozen class is a held frame plus seeded gaussian grain at sigma 2, and
sigma 2 sits an order of magnitude below `DIFF_PIXEL_DELTA = 25`. So
`diff_norm` reading exactly 0.0 on every frozen sample is **arithmetic, not
evidence** — the JSON says so in `meta.noise_sigma_note`. The frozen result
worth trusting is `flow_norm`'s, whose deadband is sub-pixel and was not
chosen against this noise level.

### Everything else

- **Farneback saturation.** Over ~7 px of window displacement the estimator
  over-reads; the flow band is documented on `flow_score` itself.
- **One scene.** `data/vtest.avi` is a single outdoor courtyard at 10 fps,
  768 x 576. Nothing here has been measured across lighting conditions,
  indoor scenes, or phone cameras.
- **Detection is upstream of everything.** A missed detection is a player
  the judge never scores; a box that wobbles is motion the judge sees. The
  15-frame miss budget is what stands between a brief occlusion and being
  removed from the arena.
- **Browser identity.** Overlap matching, greedy, best pair first. Enough
  for four players standing in a row; it is not re-identification. Two
  players who swap places while crossing swap ids, and the referee would
  judge them as each other.
- **Pose-box quality on phones.** The browser box comes from pose landmarks
  grown 15% — landmarks sit on the skeleton, so a tight box would cut
  through hair, shoulders and the outside of a swinging arm, and since the
  score is a fraction of the window that would make holding still easier
  than it should be. A phone that drops the pose model to a lower frame rate
  produces coarser boxes, and a coarser box is a noisier score.
- **The pose runtime carries no SRI hash.** The one documented exception on
  the page: MediaPipe's loader resolves its own wasm binary and model URLs
  at run time, so there is no fixed set of bytes known at build time to hash.
  It is pinned to an exact version, fetched only after an explicit opt-in,
  and nothing else on the page depends on it. Chart.js, which *can* be
  hashed, is pinned with an integrity hash.

---

## 8. Browser parity — exactly what is proven

**Proven.** Given two 96 x 96 uint8 windows and a `dt`, `site/judge.js` and
`src/redlight/judge.py` produce the *same double*. `tests/test_js_parity.py`
replays golden fixtures — identical frames, sensor-grade noise, single
pixels sitting exactly on the change threshold and one step past it, block
shifts, flat textures — through node and requires exact equality, not a
tolerance. That is a fair demand: the score is an integer count divided by
9216 and then by `dt`, and IEEE-754 division is exactly rounded in both
languages. `MotionJudge`, `FrameSampler` and the game rules are pinned the
same way, with `phase_min_s == phase_max_s` so the two PRNGs drop out of the
comparison. The shipped page can re-run the whole fixture set in the
visitor's own browser via `?selftest=1`.

**Not proven, by construction.** How a camera frame *becomes* a 96 x 96
window. Python clamps the box, resamples with OpenCV's INTER_AREA and
applies a 3x3 Gaussian blur; the browser gets there through a canvas draw.
So the honest claim is identical scoring of identical windows — not
identical verdicts from an identical camera frame. Anything that changes how
a window is produced sits outside what these fixtures can vouch for, and no
copy anywhere should claim otherwise.

The two sides also differ upstream on purpose: Python detects people with
YOLO11n or HOG, the browser uses pose landmarks, because a tab cannot afford
a detector per frame and pose gives a better box for a person facing the
camera.

---

## 9. Ethics and privacy

- **On-device, always.** The browser arena opens the camera in the tab and
  runs detection, scoring and the rules locally. No frame is uploaded, none
  is recorded, and nothing survives the tab closing. The Python engine
  writes video only when `--record` is passed explicitly.
- **Registration is consent.** Only players in frame when the countdown
  opens are registered, and only registered players can be eliminated. A
  passer-by cannot be put out by a game they did not join.
- **This is a game referee.** It measures how much of a bounding box changed
  between two frames. It does not identify anyone, does not store anything
  about anyone, and is not a surveillance tool. Pointing it at people who
  have not chosen to play is not a use this is built for.
- **Published limits, not just published wins.** The occlusion result, the
  exclusion sensitivity and the structural zero in the frozen diff class are
  in this document because a reader deciding whether to trust the referee
  needs them more than the referee needs to look good.
</content>
