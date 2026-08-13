"""Tests for the naive referee designs — pinning down exactly where each one breaks.

These are measured against `flow_score` (see `tests/test_judge.py`) on the
same deterministic synthetic scene, so the gap between "the obvious way to
build this" and the referee's actual design shows up as a number.
"""

import cv2
import numpy as np

from redlight.baselines import brightness_count, frame_diff_area, raw_flow_max
from redlight.detection import Detection
from redlight.judge import flow_score

# Same "world" construction as tests/test_judge.py: a canvas twice the size
# of the 1x frame, so 0.5x / 1x / 2x renders are all downsamples of one
# source of truth rather than three separately drawn scenes.
WORLD_W, WORLD_H = 1280, 960
BASE_W, BASE_H = 640, 480
BASE_BOX = (160, 80, 160, 320)  # x, y, w, h at 1x


def _world(seed: int = 7) -> np.ndarray:
    """A deterministic textured canvas, fine-grained enough to stay textured
    at a full 2x raw crop — unlike `flow_score`, `raw_flow_max` never
    resamples to a fixed window, so it needs real local detail at whatever
    raw resolution the ROI happens to be, not just after a 96x96 resize."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, size=(60, 80)).astype(np.uint8)
    return cv2.resize(coarse, (WORLD_W, WORLD_H), interpolation=cv2.INTER_CUBIC)


def _render(world: np.ndarray, scale: float) -> np.ndarray:
    """Render the world as a camera at `scale` times the reference resolution."""
    size = (int(BASE_W * scale), int(BASE_H * scale))
    return cv2.resize(world, size, interpolation=cv2.INTER_AREA)


def _box(scale: float) -> Detection:
    """The player's box at `scale`, framing the same patch of the world."""
    x, y, w, h = BASE_BOX
    return Detection(int(x * scale), int(y * scale), int(w * scale), int(h * scale))


def _shift(world: np.ndarray, dx_world: int) -> np.ndarray:
    """Slide the whole canvas sideways by an exact number of world pixels."""
    return np.roll(world, dx_world, axis=1)


def _pair(scale: float, dx_world: int, seed: int = 7):
    """A (before, after, box) triple: the same scene, displaced, at one scale."""
    world = _world(seed)
    prev = _render(world, scale)
    cur = _render(_shift(world, dx_world), scale)
    return prev, cur, _box(scale)


# --- brightness_count -------------------------------------------------------


def test_brightness_count_counts_pixels_over_threshold():
    """Direct correctness check, independent of the failure-mode story."""
    gray = np.zeros((20, 20), dtype=np.uint8)
    box = Detection(0, 0, 20, 20)
    gray[0, :5] = 100
    assert brightness_count(gray, box, threshold_value=30) == 5.0

    at_threshold = np.zeros((20, 20), dtype=np.uint8)
    at_threshold[0, :5] = 30
    assert brightness_count(at_threshold, box, threshold_value=30) == 0.0


def test_brightness_count_same_for_frozen_and_moving_pair():
    """No temporal term: the same held frame scores identically whether it
    is about to sit still or about to move.

    `_pair` builds a (prev, cur, box) triple from one `prev` frame; `cur` is
    `prev` with an exact displacement applied, or `prev` itself unmoved.
    `prev` is identical in both constructions — dx only ever touches `cur` —
    so it is the frame every scorer would call "the one to check". Every
    other metric here also looks at `cur` and would tell the two pairs
    apart. `brightness_count` never receives `cur` at all: it takes one
    frame and reports on it, so scoring the shared, held frame gives the
    same answer regardless of which pair it was drawn from. That is the
    asymmetry: nothing distinguishes a player about to hold still from a
    player about to move, because nothing about the future is in the input.
    """
    frozen_prev, _, box = _pair(1.0, 0)
    moving_prev, _, _ = _pair(1.0, 20)

    assert brightness_count(frozen_prev, box) == brightness_count(moving_prev, box)


# --- frame_diff_area ---------------------------------------------------------


def test_frame_diff_area_counts_pixels_over_delta():
    """Direct correctness check, independent of the failure-mode story."""
    prev = np.full((10, 10), 100, dtype=np.uint8)
    cur = prev.copy()
    cur[0, :4] = 100 + 25 + 1
    assert frame_diff_area(prev, cur, pixel_delta=25) == 4.0

    at_delta = prev.copy()
    at_delta[0, :4] = 100 + 25
    assert frame_diff_area(prev, at_delta, pixel_delta=25) == 0.0


def test_frame_diff_area_fires_on_background_motion():
    """Whole-frame differencing can't tell the player moving from anything
    else moving in the shot.

    Two synthetic frames are composited: the player's box is bit-for-bit
    identical in both, and a patch entirely outside it changes. A referee
    that only cares whether the player moved should read zero. This one has
    no box argument to check against, so it can't.
    """
    box = Detection(50, 50, 100, 200)

    prev = np.zeros((300, 300), dtype=np.uint8)
    cur = prev.copy()
    cur[10:20, 250:260] = 200  # well outside the player's box

    player_prev = prev[box.y : box.y + box.h, box.x : box.x + box.w]
    player_cur = cur[box.y : box.y + box.h, box.x : box.x + box.w]
    assert np.array_equal(player_prev, player_cur)  # the player never moved

    assert frame_diff_area(prev, cur) > 0.0


# --- raw_flow_max -------------------------------------------------------------


def test_raw_flow_max_reads_near_zero_on_a_frozen_pair():
    """No deadband: unlike `flow_score`, this design never subtracts a noise
    floor, so a frozen pair reads small but not exactly zero."""
    frame, _, box = _pair(1.0, 0)
    assert 0.0 <= raw_flow_max(frame, frame, box) < 1.0


def test_raw_flow_max_degenerate_box_returns_none():
    frame, cur, _ = _pair(1.0, 16)
    sliver = Detection(10, 10, 4, 200)
    assert raw_flow_max(frame, cur, sliver) is None


def test_raw_flow_max_grows_with_scale_while_flow_score_holds():
    """Head to head: raw_flow_max reads camera resolution as movement; the
    referee's own metric doesn't.

    Same normalized displacement (20 world px, 6.0 window px — inside
    `flow_score`'s honest 2.2-7 px operating band) rendered at 1x and 2x.
    Doubling the render resolution roughly doubles the raw pixel
    displacement inside an unresampled box, so a naive Farneback-on-the-raw-
    ROI design reports a much bigger number for identical motion. `flow_score`
    resamples the box first, so it barely moves.
    """
    dt = 0.1
    dx_world = 20

    prev1, cur1, box1 = _pair(1.0, dx_world)
    prev2, cur2, box2 = _pair(2.0, dx_world)

    raw1 = raw_flow_max(prev1, cur1, box1)
    raw2 = raw_flow_max(prev2, cur2, box2)
    assert raw1 > 0.0, raw1
    assert raw2 / raw1 > 1.4, (raw1, raw2)

    flow1 = flow_score(prev1, cur1, box1, dt)
    flow2 = flow_score(prev2, cur2, box2, dt)
    assert flow1 > 0.0, flow1
    assert max(flow1, flow2) / min(flow1, flow2) <= 1.35, (flow1, flow2)
