"""The measurement harness: every number this product publishes.

The referee makes three claims — the verdict does not change when a player
stands further from the camera, it does not change when the footage arrives
faster, and a spectator walking past cannot put a player out. Claims like
those are cheap to write and easy to believe about your own code, so this
module measures them instead, on real footage, against the naive designs in
`redlight.baselines` that a reasonable person would try first.

**Ground truth.** `data/vtest.avi` is 795 frames of pedestrians crossing a
courtyard at 10 fps. Every walker YOLO finds and the tracker holds becomes a
*moving* sample. For the *frozen* class, the same frame is held in place as
its own successor with seeded gaussian sensor grain (sigma 2) added on top —
a player standing perfectly still in front of a real camera. Both classes
are scored on the same boxes, on the same frames, through the same sampler,
so nothing distinguishes them except whether the scene moved.

A walker whose box did not shift at all between the two frames of a pair is
not honest evidence of motion, so those samples are dropped and counted; the
count and the flow score they would have contributed are both reported.
Because the two classes are paired sample-for-sample, an exclusion drops the
frozen twin too — otherwise the classes could differ in box sizes rather
than in movement, and the AUC would be measuring the wrong thing.

**What is deliberately not flattering.** `brightness_count` reads a single
frame, so its score for a held frame is its score for the moving one by
construction, and its AUC lands at chance. `frame_diff_area` separates the
two classes almost perfectly here and its failure only shows up in the
composited background scenario, where it calls a motionless player out
because somebody else walked past. `raw_flow_max` also scores well at native
resolution; its failure is the resolution sweep. Each is reported as
measured, including where it wins.

**Reproducibility.** Seeds are fixed, OpenCV and torch are pinned to one
thread for the duration, and the JSON carries no wall-clock stamp, so two
runs of this harness on the same footage produce byte-identical output —
with one honest exception. `runtime_ms` measures how fast the machine is and
genuinely varies between runs; stabilising it would mean publishing a number
that had not been measured.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from redlight import __version__, baselines
from redlight.app import MatchReport
from redlight.config import GameConfig
from redlight.detection import Detection, make_detector
from redlight.judge import (
    SAMPLE_INTERVAL_S,
    FrameSampler,
    diff_score,
    flow_score,
)
from redlight.tracking import Tracker

VIDEO = "data/vtest.avi"
"""The ground-truth footage: pedestrians crossing a courtyard at 10 fps."""

SEED = 1729
"""One seed for every random draw the harness makes, so reruns match."""

FAST_FRAMES = 60
"""Frame cap for `--fast`: enough footage to exercise every code path."""

NOISE_SIGMA = 2.0
"""Sensor grain added to a held frame, in brightness levels."""

MIN_CENTER_DISPLACEMENT_PX = 0.5
"""A walker whose box centre moved less than this is not evidence of motion."""

DETECTOR = "yolo"
DETECTOR_CONF = 0.35

THRESHOLDS: dict[str, float] = {
    "brightness": 7000.0,
    "frame_diff_area": 7000.0,
    "raw_flow": 2.0,
    "flow_norm": GameConfig().threshold,
    "diff_norm": GameConfig().diff_threshold,
}
"""Each design judged at the cutoff it actually ships with.

The two normalized metrics use the shipped `GameConfig` thresholds. The
three naive designs have no shipped cutoff — nobody ships them — so each
gets the round number someone reaching for that design would reach for
first: 7000 pixels for the two pixel counts, 2 pixels of displacement for
raw flow.
"""

SCALES: tuple[float, ...] = (0.5, 1.0, 1.667)
"""Resolution sweep: half size, native, and the jump to a 1280-wide sensor."""

SCALE_KEYS = {0.5: "0.5x", 1.0: "1x", 1.667: "1.667x"}

SWEEP_METRICS = ("brightness", "raw_flow", "flow_norm", "diff_norm")

TRACE_SAMPLES_FULL = 30
TRACE_SAMPLES_FAST = 8
"""Length of the armed segment the referee-lab traces are cut from, in samples."""

RUNTIME_FRAMES_FULL = 30
RUNTIME_FRAMES_FAST = 5

BACKGROUND_MIN_FRAMES = 20
"""Shortest composited run worth reporting a crossing time from."""

DEMO_CONFIG = GameConfig(
    seed=7,
    countdown_s=1.0,
    duration_s=12.0,
    phase_min_s=1.5,
    phase_max_s=3.0,
    grace_s=0.6,
    metric="flow",
    confirm_frames=3,
    detector="yolo",
    conf=DETECTOR_CONF,
    max_misses=15,
)
"""The seeded demo match: long enough for several red lights on this footage."""

DEMO_SKIP = 150
"""Start the demo once the courtyard is busy: a fuller arena to referee."""

DEMO_AUTO_START = 20
"""Register after twenty frames of company, so the match has a crowd in it."""

DEMO_MAX_SECONDS = 40.0
DEMO_FAST_DURATION_S = 2.0
DEMO_FAST_MAX_SECONDS = 5.0


# --------------------------------------------------------------------------
# frames, boxes, and the two classes
# --------------------------------------------------------------------------


def _read_frames(max_frames: int | None) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (index, timestamp, BGR frame) from the ground-truth footage."""
    from redlight.sources import FrameSource

    with FrameSource(VIDEO) as source:
        index = 0
        while True:
            result = source.read()
            if result is None:
                return
            if max_frames is not None and index >= max_frames:
                return
            frame, timestamp = result
            yield index, timestamp, frame
            index += 1


def _video_fps() -> float:
    from redlight.sources import FrameSource

    with FrameSource(VIDEO) as source:
        return float(source.fps or 30.0)


def _track_pass(max_frames: int | None) -> tuple[list[dict[int, Detection]], int]:
    """Detect and track once, and keep the boxes for every later pass.

    Detection is by far the most expensive thing here, and every measurement
    below wants the same boxes, so it happens exactly once. Only tracks
    matched on the frame itself are kept: a coasting track is the tracker's
    guess about where somebody probably still is, which is not something to
    build a ground-truth label on.
    """
    detector = make_detector(DETECTOR, DETECTOR_CONF)
    tracker = Tracker(max_misses=GameConfig().max_misses)

    boxes_by_frame: list[dict[int, Detection]] = []
    for _index, _timestamp, frame in _read_frames(max_frames):
        tracks = tracker.update(detector.detect(frame))
        boxes_by_frame.append({t.track_id: t.box for t in tracks if t.misses == 0})
    return boxes_by_frame, len(boxes_by_frame)


def _sampled_pairs(
    max_frames: int | None,
) -> Iterator[tuple[int, int, float, np.ndarray, np.ndarray, float]]:
    """Yield the frame pairs the referee would actually score.

    Pairs come out of `FrameSampler` rather than off consecutive frames, so
    the harness measures exactly what ships: comparisons made on a fixed
    real-time clock. Each yield is (current index, previous index, current
    timestamp, previous gray, current gray, dt).
    """
    sampler = FrameSampler()
    held_index: int | None = None

    for index, timestamp, frame in _read_frames(max_frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sample = sampler.offer(gray, timestamp)
        if sample is None:
            if held_index is None:
                held_index = index
            continue
        prev_gray, cur_gray, dt = sample
        yield index, held_index, timestamp, prev_gray, cur_gray, dt
        held_index = index


def _held_frame(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A frame held perfectly still, as a real camera would record it.

    Nothing in the scene moves; only the sensor grain redraws itself. That
    is the frozen class: not a duplicated buffer (which any metric can score
    at exactly zero) but the same scene photographed twice.
    """
    noise = rng.normal(0.0, NOISE_SIGMA, gray.shape)
    return np.clip(gray.astype(np.int16) + np.rint(noise).astype(np.int16), 0, 255).astype(
        np.uint8
    )


def _center_shift(earlier: Detection, later: Detection) -> float:
    """How far a tracked box's centre travelled, in pixels."""
    return float(
        np.hypot(
            (later.x + later.w / 2) - (earlier.x + earlier.w / 2),
            (later.y + later.h / 2) - (earlier.y + earlier.h / 2),
        )
    )


def _scaled_box(box: Detection, scale: float) -> Detection:
    return Detection(
        x=int(round(box.x * scale)),
        y=int(round(box.y * scale)),
        w=int(round(box.w * scale)),
        h=int(round(box.h * scale)),
        conf=box.conf,
    )


def _rescale(gray: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return gray
    height, width = gray.shape[:2]
    size = (int(round(width * scale)), int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(gray, size, interpolation=interpolation)


def _score_all(
    prev_gray: np.ndarray, cur_gray: np.ndarray, box: Detection, dt: float, whole_frame: float
) -> dict[str, float] | None:
    """Every metric's verdict on one box over one pair, or None if unusable.

    A box too small to carry texture is scored by nobody, so the sample is
    dropped from every metric at once rather than leaving the classifier
    table comparing different sample sets per row.
    """
    flow = flow_score(prev_gray, cur_gray, box, dt)
    if flow is None:
        return None
    diff = diff_score(prev_gray, cur_gray, box, dt)
    raw_flow = baselines.raw_flow_max(prev_gray, cur_gray, box)
    if diff is None or raw_flow is None:
        return None
    return {
        "brightness": baselines.brightness_count(cur_gray, box),
        "frame_diff_area": whole_frame,
        "raw_flow": raw_flow,
        "flow_norm": flow,
        "diff_norm": diff,
    }


# --------------------------------------------------------------------------
# the classifier table
# --------------------------------------------------------------------------


class _Samples:
    """Paired moving/frozen scores for every metric, plus the traces."""

    def __init__(self) -> None:
        self.moving: dict[str, list[float]] = {name: [] for name in THRESHOLDS}
        self.frozen: dict[str, list[float]] = {name: [] for name in THRESHOLDS}
        self.traces: dict[int, list[tuple[int, float, float]]] = {}
        self.exclusions = {"no_prior_box": 0, "static_box": 0, "unusable_crop": 0}
        self.static_box_flow: list[float] = []
        self.pairs = 0


def _collect_samples(max_frames: int | None, boxes_by_frame: list[dict[int, Detection]]) -> _Samples:
    """Score both classes over the footage, one pass, one seeded noise stream."""
    rng = np.random.default_rng(SEED)
    samples = _Samples()
    sample_index = 0

    for index, prev_index, timestamp, prev_gray, cur_gray, dt in _sampled_pairs(max_frames):
        frozen_gray = _held_frame(cur_gray, rng)
        moving_whole = baselines.frame_diff_area(prev_gray, cur_gray)
        frozen_whole = baselines.frame_diff_area(cur_gray, frozen_gray)

        prior_boxes = boxes_by_frame[prev_index] if prev_index is not None else {}
        for track_id, box in sorted(boxes_by_frame[index].items()):
            prior = prior_boxes.get(track_id)
            if prior is None:
                samples.exclusions["no_prior_box"] += 1
                continue

            moving = _score_all(prev_gray, cur_gray, box, dt, moving_whole)
            frozen = _score_all(cur_gray, frozen_gray, box, dt, frozen_whole)
            if moving is None or frozen is None:
                samples.exclusions["unusable_crop"] += 1
                continue

            if _center_shift(prior, box) < MIN_CENTER_DISPLACEMENT_PX:
                samples.exclusions["static_box"] += 1
                samples.static_box_flow.append(moving["flow_norm"])
                continue

            for name in THRESHOLDS:
                samples.moving[name].append(moving[name])
                samples.frozen[name].append(frozen[name])
            samples.traces.setdefault(track_id, []).append(
                (sample_index, timestamp, moving["flow_norm"])
            )

        sample_index += 1

    samples.pairs = sample_index
    return samples


def _flagged_pct(scores: list[float], threshold: float) -> float:
    """Share of scores a judge at this threshold would call movement.

    Strictly above, matching `MotionJudge`, which eliminates on `ema >
    threshold` — a score sitting exactly on the cutoff is not a call.
    """
    if not scores:
        return 0.0
    values = np.asarray(scores, dtype=float)
    return 100.0 * float(np.count_nonzero(values > threshold)) / len(values)


def _classifier_table(samples: _Samples) -> dict[str, dict[str, float]]:
    from sklearn.metrics import roc_auc_score

    table: dict[str, dict[str, float]] = {}
    for name, threshold in THRESHOLDS.items():
        moving = np.asarray(samples.moving[name], dtype=float)
        frozen = np.asarray(samples.frozen[name], dtype=float)
        labels = np.concatenate([np.ones(len(moving)), np.zeros(len(frozen))])
        scores = np.concatenate([moving, frozen])
        table[name] = {
            "auc": float(roc_auc_score(labels, scores)),
            "threshold": threshold,
            "frozen_flagged": _flagged_pct(samples.frozen[name], threshold),
            "moving_flagged": _flagged_pct(samples.moving[name], threshold),
            "frozen_median": float(np.median(frozen)),
            "moving_median": float(np.median(moving)),
        }
    return table


def _choose_threshold(samples: _Samples, metric: str) -> tuple[float | None, dict]:
    """The margin-maximizing cutoff between the two classes, if one exists.

    The frozen class's 99th percentile and the moving class's 1st are the
    two edges that matter: a cutoff between them clears the noisiest held
    frame and still catches all but the faintest walker. The geometric
    midpoint sits equidistant in the ratio sense, which is the right sense
    for a score that spans orders of magnitude — but it collapses to zero
    when the frozen edge is exactly zero, which is what a working deadband
    produces, so the arithmetic midpoint takes over there.

    If the two distributions overlap there is no such cutoff, and this says
    so rather than picking a number that separates nothing.
    """
    frozen = np.asarray(samples.frozen[metric], dtype=float)
    moving = np.asarray(samples.moving[metric], dtype=float)
    frozen_p99 = float(np.percentile(frozen, 99))
    moving_p1 = float(np.percentile(moving, 1))

    separated = frozen_p99 < moving_p1
    if not separated:
        chosen = None
        rule = "none"
    elif frozen_p99 > 0.0:
        chosen = float(np.sqrt(frozen_p99 * moving_p1))
        rule = "geometric_midpoint"
    else:
        chosen = moving_p1 / 2.0
        rule = "midpoint"

    separation = {
        "frozen_p99": frozen_p99,
        "frozen_max": float(np.max(frozen)),
        "moving_p1": moving_p1,
        "separated": separated,
        "rule": rule,
        "applied": {
            "frozen_flagged": _flagged_pct(samples.frozen[metric], chosen)
            if chosen is not None
            else None,
            "moving_flagged": _flagged_pct(samples.moving[metric], chosen)
            if chosen is not None
            else None,
        },
    }
    return chosen, separation


# --------------------------------------------------------------------------
# the sweeps
# --------------------------------------------------------------------------


def _resolution_sweep(
    max_frames: int | None, boxes_by_frame: list[dict[int, Detection]]
) -> dict[str, dict]:
    """Rescale the footage and see which metrics keep their verdict.

    The frame is rescaled and the boxes with it, and the grain is added
    *after* rescaling — sensor noise belongs to the sensor, so a higher
    resolution means more noisy pixels, not the same noise smoothed out.

    A sample only counts if every metric can score it at all three scales:
    a box 15 pixels wide is 7 at half size, below the minimum crop, and
    letting scales quietly measure different samples would turn a sample
    set difference into an apparent invariance result.
    """
    rng = np.random.default_rng(SEED + 1)
    flagged = {
        metric: {SCALE_KEYS[s]: {"moving": [], "frozen": []} for s in SCALES}
        for metric in SWEEP_METRICS
    }
    used = 0

    for index, prev_index, _timestamp, prev_gray, cur_gray, dt in _sampled_pairs(max_frames):
        scaled: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for scale in SCALES:
            prev_s = _rescale(prev_gray, scale)
            cur_s = _rescale(cur_gray, scale)
            scaled[scale] = (prev_s, cur_s, _held_frame(cur_s, rng))

        prior_boxes = boxes_by_frame[prev_index] if prev_index is not None else {}
        for track_id, box in sorted(boxes_by_frame[index].items()):
            prior = prior_boxes.get(track_id)
            if prior is None or _center_shift(prior, box) < MIN_CENTER_DISPLACEMENT_PX:
                continue

            per_scale: dict[float, tuple[dict[str, float], dict[str, float]]] = {}
            for scale in SCALES:
                prev_s, cur_s, frozen_s = scaled[scale]
                scaled_box = _scaled_box(box, scale)
                moving = _score_all(prev_s, cur_s, scaled_box, dt, 0.0)
                frozen = _score_all(cur_s, frozen_s, scaled_box, dt, 0.0)
                if moving is None or frozen is None:
                    per_scale = {}
                    break
                per_scale[scale] = (moving, frozen)
            if not per_scale:
                continue

            used += 1
            for scale, (moving, frozen) in per_scale.items():
                key = SCALE_KEYS[scale]
                for metric in SWEEP_METRICS:
                    flagged[metric][key]["moving"].append(moving[metric])
                    flagged[metric][key]["frozen"].append(frozen[metric])

    sweep: dict[str, dict] = {"samples": used}
    for metric in SWEEP_METRICS:
        sweep[metric] = {
            SCALE_KEYS[scale]: {
                "frozen_flagged": _flagged_pct(
                    flagged[metric][SCALE_KEYS[scale]]["frozen"], THRESHOLDS[metric]
                ),
                "moving_flagged": _flagged_pct(
                    flagged[metric][SCALE_KEYS[scale]]["moving"], THRESHOLDS[metric]
                ),
            }
            for scale in SCALES
        }
    return sweep


def _fps_arm(
    max_frames: int | None,
    boxes_by_frame: list[dict[int, Detection]],
    repeat: int,
    tick_fps: float,
) -> dict[tuple[int, int], tuple[float, dict[int, tuple[bool, bool]]]]:
    """Run the footage past the sampler at `tick_fps` and collect verdicts.

    The footage is 10 fps, so a 30 fps camera pointed at the same scene is
    modelled by offering each frame three times on a 30 fps clock: the same
    real motion, arriving three times as often. What the sampler does with
    that is the thing being measured — it is supposed to make the extra
    ticks irrelevant, and scoring every tick instead (what the naive loop
    does) would compare a frame against itself and read zero motion.

    Verdicts are keyed by the *content* the sampler released — the pair of
    frame indices it actually compared — not by position in the output.
    The two arms do not release the same number of pairs, so lining them up
    by position would compare one arm's verdict on one stretch of footage
    against the other arm's verdict on a different stretch, and call the
    disagreement a frame-rate effect.

    The interval the sampler measured each pair over is kept alongside the
    verdicts, because the two arms do not always agree on it either.
    """
    sampler = FrameSampler()
    decisions: dict[tuple[int, int], tuple[float, dict[int, tuple[bool, bool]]]] = {}
    held_index: int | None = None
    tick = 0

    for index, _timestamp, frame in _read_frames(max_frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for _ in range(repeat):
            timestamp = tick / tick_fps
            tick += 1
            sample = sampler.offer(gray, timestamp)
            if sample is None:
                if held_index is None:
                    held_index = index
                continue
            prev_gray, cur_gray, dt = sample
            prior_boxes = boxes_by_frame[held_index] if held_index is not None else {}
            verdicts: dict[int, tuple[bool, bool]] = {}
            for track_id, box in sorted(boxes_by_frame[index].items()):
                prior = prior_boxes.get(track_id)
                if prior is None or _center_shift(prior, box) < MIN_CENTER_DISPLACEMENT_PX:
                    continue
                flow = flow_score(prev_gray, cur_gray, box, dt)
                diff = diff_score(prev_gray, cur_gray, box, dt)
                if flow is None or diff is None:
                    continue
                verdicts[track_id] = (
                    flow > THRESHOLDS["flow_norm"],
                    diff > THRESHOLDS["diff_norm"],
                )
            decisions[(held_index, index)] = (dt, verdicts)
            held_index = index

    return decisions


def _fps_sweep(max_frames: int | None, boxes_by_frame: list[dict[int, Detection]]) -> dict:
    """Do 10 fps and 30 fps ticks produce the same eliminations?

    Only the stretches of footage both arms actually compared can answer
    that, so agreement is counted over the content pairs the two arms have
    in common — the comparable sample sequence — and how many decisions that
    came to is published alongside the percentage.

    `dt_mismatch_pct` is the reason the answer is not a flat 100%. The
    sampler releases on a wall-clock interval, so a pair of frames can be
    measured over 0.1 s on one clock and 0.133 s on the other: the same
    motion, divided by a third more time. That residual is what any
    disagreement here is made of, so it is reported next to it rather than
    left for a reader to wonder about.
    """
    native = _fps_arm(max_frames, boxes_by_frame, repeat=1, tick_fps=10.0)
    tripled = _fps_arm(max_frames, boxes_by_frame, repeat=3, tick_fps=30.0)
    shared = sorted(set(native) & set(tripled))
    mismatched = sum(1 for key in shared if abs(native[key][0] - tripled[key][0]) > 1e-12)
    dt_mismatch_pct = 100.0 * mismatched / len(shared) if shared else 0.0

    sweep: dict[str, dict] = {}
    for position, metric in enumerate(("flow_norm", "diff_norm")):
        agreements = 0
        compared = 0
        for key in shared:
            left, right = native[key][1], tripled[key][1]
            for track_id in sorted(set(left) & set(right)):
                compared += 1
                agreements += left[track_id][position] == right[track_id][position]
        sweep[metric] = {
            "10fps": {
                "pairs_released": len(native),
                "samples": sum(len(v) for _dt, v in native.values()),
                "flagged_pct": _decision_rate(native, position),
            },
            "30fps": {
                "pairs_released": len(tripled),
                "samples": sum(len(v) for _dt, v in tripled.values()),
                "flagged_pct": _decision_rate(tripled, position),
            },
            "pairs_compared": len(shared),
            "dt_mismatch_pct": dt_mismatch_pct,
            "decisions_compared": compared,
            "decisions_agree_pct": 100.0 * agreements / compared if compared else 0.0,
        }
    return sweep


def _decision_rate(
    decisions: dict[tuple[int, int], tuple[float, dict[int, tuple[bool, bool]]]], position: int
) -> float:
    total = sum(len(v) for _dt, v in decisions.values())
    if not total:
        return 0.0
    flagged = sum(
        1 for _dt, frame in decisions.values() for verdict in frame.values() if verdict[position]
    )
    return 100.0 * flagged / total


# --------------------------------------------------------------------------
# the spectator scenario
# --------------------------------------------------------------------------


def _pick_frozen_player(
    boxes_by_frame: list[dict[int, Detection]],
) -> tuple[int, int, Detection] | None:
    """The largest box early enough to leave a run of footage behind it.

    Largest because a bigger crop makes the composite obvious rather than
    marginal: if whole-frame differencing still cannot see this player
    standing still, it never could.
    """
    latest_start = len(boxes_by_frame) - BACKGROUND_MIN_FRAMES
    best: tuple[int, int, Detection] | None = None
    best_area = 0
    for index in range(max(latest_start, 0)):
        for track_id, box in sorted(boxes_by_frame[index].items()):
            area = box.w * box.h
            if area > best_area:
                best_area = area
                best = (index, track_id, box)
    return best


def _background_scenario(
    max_frames: int | None, boxes_by_frame: list[dict[int, Detection]]
) -> dict:
    """Paste a motionless player into live traffic and see who calls them out.

    The player is frozen: the same crop, at the same coordinates, in every
    frame, with fresh sensor grain each time. Everything around them keeps
    moving, because everything around them is the real footage. A referee
    that scores the whole frame cannot tell those two facts apart.
    """
    picked = _pick_frozen_player(boxes_by_frame)
    if picked is None:
        return {
            "frames": 0,
            "samples": 0,
            "frozen_box": {"x": 0, "y": 0, "w": 0, "h": 0},
            "frame_diff_area_flagged_pct": 0.0,
            "first_crossing_s": None,
            "per_box_flagged_pct": {"flow_norm": 0.0, "diff_norm": 0.0},
        }

    start_index, _track_id, box = picked
    rng = np.random.default_rng(SEED + 2)
    sampler = FrameSampler()

    patch: np.ndarray | None = None
    region: tuple[int, int, int, int] | None = None
    start_ts: float | None = None
    whole_flags: list[bool] = []
    flow_flags: list[bool] = []
    diff_flags: list[bool] = []
    first_crossing: float | None = None
    frames = 0

    for index, timestamp, frame in _read_frames(max_frames):
        if index < start_index:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if patch is None:
            height, width = gray.shape[:2]
            x1, y1 = max(0, box.x), max(0, box.y)
            x2, y2 = min(width, box.x + box.w), min(height, box.y + box.h)
            region = (x1, y1, x2, y2)
            patch = gray[y1:y2, x1:x2].copy()
            start_ts = timestamp

        composited = gray.copy()
        x1, y1, x2, y2 = region
        composited[y1:y2, x1:x2] = _held_frame(patch, rng)
        frames += 1

        sample = sampler.offer(composited, timestamp)
        if sample is None:
            continue
        prev_gray, cur_gray, dt = sample
        whole = baselines.frame_diff_area(prev_gray, cur_gray)
        flagged = whole > THRESHOLDS["frame_diff_area"]
        if flagged and first_crossing is None:
            first_crossing = timestamp - start_ts
        whole_flags.append(flagged)

        flow = flow_score(prev_gray, cur_gray, box, dt)
        diff = diff_score(prev_gray, cur_gray, box, dt)
        if flow is not None:
            flow_flags.append(flow > THRESHOLDS["flow_norm"])
        if diff is not None:
            diff_flags.append(diff > THRESHOLDS["diff_norm"])

    def rate(flags: list[bool]) -> float:
        return 100.0 * sum(flags) / len(flags) if flags else 0.0

    return {
        "frames": frames,
        "samples": len(whole_flags),
        "frozen_box": {"x": box.x, "y": box.y, "w": box.w, "h": box.h},
        "frame_diff_area_flagged_pct": rate(whole_flags),
        "first_crossing_s": first_crossing,
        "per_box_flagged_pct": {"flow_norm": rate(flow_flags), "diff_norm": rate(diff_flags)},
    }


# --------------------------------------------------------------------------
# runtime, traces, demo match
# --------------------------------------------------------------------------


def _runtime_ms(fast: bool, boxes_by_frame: list[dict[int, Detection]]) -> dict[str, float]:
    """Median cost per frame of each detector, and per player of each metric.

    Median rather than mean: the first call through any of these warms a
    model, allocates a buffer, or touches cold memory, and publishing a
    warm-up as if it were the steady-state cost would overstate what the
    pipeline actually costs to run.
    """
    frames_wanted = RUNTIME_FRAMES_FAST if fast else RUNTIME_FRAMES_FULL
    frames = [frame for _index, _ts, frame in _read_frames(frames_wanted)]

    timings: dict[str, list[float]] = {
        "yolo11n": [],
        "hog": [],
        "flow_norm_per_player": [],
        "diff_norm_per_player": [],
    }

    for name, detector_name in (("yolo11n", "yolo"), ("hog", "hog")):
        detector = make_detector(detector_name, DETECTOR_CONF)
        detector.detect(frames[0])  # warm up outside the measurement
        for frame in frames:
            started = time.perf_counter()
            detector.detect(frame)
            timings[name].append(1000.0 * (time.perf_counter() - started))

    grays = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    dt = SAMPLE_INTERVAL_S
    for index in range(1, len(grays)):
        for _track_id, box in sorted(boxes_by_frame[index].items()):
            started = time.perf_counter()
            flow_score(grays[index - 1], grays[index], box, dt)
            timings["flow_norm_per_player"].append(1000.0 * (time.perf_counter() - started))

            started = time.perf_counter()
            diff_score(grays[index - 1], grays[index], box, dt)
            timings["diff_norm_per_player"].append(1000.0 * (time.perf_counter() - started))

    # An empty list would median to NaN, which is not valid JSON — 0.0 keeps
    # the file readable and fails the "every timing is positive" test loudly
    # rather than shipping a number nobody can parse.
    return {
        name: float(np.median(values)) if values else 0.0 for name, values in timings.items()
    }


def _traces(samples: _Samples, fast: bool) -> dict:
    """Per-player score curves from one armed segment, for the referee lab.

    The site lets a visitor drag the threshold across these curves and watch
    the calls change, so the segment picked is the one carrying the most
    players all the way through — a lab bench with one trace on it teaches
    nothing about where the line should go.
    """
    window = TRACE_SAMPLES_FAST if fast else TRACE_SAMPLES_FULL
    if not samples.traces:
        return {
            "metric": "flow_norm",
            "threshold": THRESHOLDS["flow_norm"],
            "sample_interval_s": SAMPLE_INTERVAL_S,
            "window_samples": window,
            "red_phase_player_traces": [],
        }

    last_index = max(point[0] for points in samples.traces.values() for point in points)
    window = min(window, last_index + 1)
    seen = {
        track_id: {point[0] for point in points}
        for track_id, points in sorted(samples.traces.items())
    }

    best_start = 0
    best_ids: list[int] = []
    for start in range(0, last_index - window + 2):
        wanted = set(range(start, start + window))
        ids = [track_id for track_id, indices in seen.items() if indices >= wanted]
        if len(ids) > len(best_ids):
            best_start, best_ids = start, ids

    if not best_ids:
        # Nobody is held for a whole window on this footage, so fall back to
        # the longest single run rather than shipping an empty lab.
        track_id = max(samples.traces, key=lambda k: (len(samples.traces[k]), -k))
        best_ids = [track_id]
        best_start = samples.traces[track_id][0][0]

    end = best_start + window - 1
    traces = []
    for track_id in best_ids:
        points = [
            point for point in samples.traces[track_id] if best_start <= point[0] <= end
        ]
        traces.append(
            {
                "track_id": track_id,
                "t": [round(point[1] - points[0][1], 6) for point in points],
                "score": [point[2] for point in points],
            }
        )

    return {
        "metric": "flow_norm",
        "threshold": THRESHOLDS["flow_norm"],
        "sample_interval_s": SAMPLE_INTERVAL_S,
        "window_samples": window,
        "red_phase_player_traces": traces,
    }


def _demo_match(fast: bool) -> dict:
    """One seeded match on the ground-truth footage, played for real.

    Nothing here is staged: the same `app.run` the CLI calls, on the same
    footage, with a fixed seed so the match that gets published is the match
    anyone rerunning this gets.
    """
    from redlight import app

    config = DEMO_CONFIG
    max_seconds = DEMO_MAX_SECONDS
    if fast:
        config = GameConfig(**{**config.__dict__, "duration_s": DEMO_FAST_DURATION_S})
        max_seconds = DEMO_FAST_MAX_SECONDS

    report: MatchReport = app.run(
        config,
        VIDEO,
        headless=True,
        auto_start_frames=DEMO_AUTO_START,
        skip=DEMO_SKIP,
        max_seconds=max_seconds,
    )
    return {
        "outcome": report.outcome,
        "players": report.players,
        "survivors": report.survivors,
        "eliminations": [list(row) for row in report.eliminations],
        "config": {
            "seed": config.seed,
            "detector": config.detector,
            "metric": config.metric,
            "threshold": config.threshold,
            "confirm_frames": config.confirm_frames,
            "countdown_s": config.countdown_s,
            "duration_s": config.duration_s,
            "phase_min_s": config.phase_min_s,
            "phase_max_s": config.phase_max_s,
            "grace_s": config.grace_s,
            "skip": DEMO_SKIP,
            "auto_start_frames": DEMO_AUTO_START,
            "max_seconds": max_seconds,
        },
    }


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def _rounded(value, digits: int = 6):
    """Round every float in a structure, so reruns compare byte for byte.

    Float arithmetic on the same input is reproducible, but the last bit or
    two of a percentile is not worth publishing either way; rounding keeps
    the JSON stable and readable at once. Negative zero is folded into zero
    so a metric that measured nothing does not print as `-0.0`.
    """
    if isinstance(value, bool) or isinstance(value, int) or value is None:
        return value
    if isinstance(value, float):
        result = round(value, digits)
        return 0.0 if result == 0 else result
    if isinstance(value, dict):
        return {key: _rounded(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(item, digits) for item in value]
    return value


def run_benchmark(out_path: str, fast: bool = False) -> dict:
    """Measure the referee and the naive designs, and write the results JSON.

    Args:
        out_path: Where to write the results; parent directories are created.
        fast: Cap the footage at `FAST_FRAMES` frames. Same schema, same
            method, small enough to run in a test.

    Returns:
        The same payload that was written.
    """
    previous_cv_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    previous_torch_threads = _set_torch_threads(1)

    try:
        max_frames = FAST_FRAMES if fast else None
        boxes_by_frame, frames_used = _track_pass(max_frames)
        samples = _collect_samples(max_frames, boxes_by_frame)

        flow_threshold, flow_separation = _choose_threshold(samples, "flow_norm")
        diff_threshold, diff_separation = _choose_threshold(samples, "diff_norm")

        payload = {
            "meta": {
                "video": VIDEO,
                "fps": _video_fps(),
                "frames_used": frames_used,
                # Fewer than one pair per frame is expected, not a bug: the
                # sampler releases on a 0.1 s clock and this footage arrives
                # at exactly 10 fps, so a frame whose timestamp lands a hair
                # under the interval waits for the next one.
                "pairs_sampled": samples.pairs,
                "fast": fast,
                "seed": SEED,
                "sample_interval_s": SAMPLE_INTERVAL_S,
                "detector": DETECTOR,
                "detector_conf": DETECTOR_CONF,
                "noise_sigma": NOISE_SIGMA,
                "n_moving": len(samples.moving["flow_norm"]),
                "n_frozen": len(samples.frozen["flow_norm"]),
                "exclusions": {
                    **samples.exclusions,
                    "min_center_displacement_px": MIN_CENTER_DISPLACEMENT_PX,
                    "static_box_median_flow_norm": float(np.median(samples.static_box_flow))
                    if samples.static_box_flow
                    else None,
                },
                "thresholds": dict(THRESHOLDS),
                "resolution_scales": list(SCALES),
                "package_version": __version__,
            },
            "classifiers": _classifier_table(samples),
            "chosen_thresholds": {"flow": flow_threshold, "diff": diff_threshold},
            "threshold_separation": {"flow": flow_separation, "diff": diff_separation},
            "resolution_sweep": _resolution_sweep(max_frames, boxes_by_frame),
            "fps_sweep": _fps_sweep(max_frames, boxes_by_frame),
            "background_scenario": _background_scenario(max_frames, boxes_by_frame),
            "runtime_ms": _runtime_ms(fast, boxes_by_frame),
            "traces": _traces(samples, fast),
            "demo_match": _demo_match(fast),
        }
    finally:
        cv2.setNumThreads(previous_cv_threads)
        _set_torch_threads(previous_torch_threads)

    payload = _rounded(payload)
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _set_torch_threads(count: int | None) -> int | None:
    """Pin torch to `count` threads and report what it was, if torch is here.

    Detection runs through torch, and its reduction order across worker
    threads is not fixed, which is enough to move a confidence score in the
    last few decimals and occasionally change a box by a pixel. Since every
    number below is derived from those boxes, one thread is the price of a
    reproducible measurement.
    """
    if count is None:
        return None
    try:
        import torch
    except ImportError:  # pragma: no cover - torch ships with ultralytics
        return None
    previous = torch.get_num_threads()
    torch.set_num_threads(count)
    return previous
