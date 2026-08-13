"""Tests for the motion judge — the claims the referee's fairness rests on.

The judge has to give the same verdict for the same movement whether the
player is close to the camera or far from it, and whether the machine is
managing 10 frames a second or 60. These tests pin those two claims down
with a synthetic scene we can move by an exact number of pixels, so a
regression shows up as a number rather than as an argument about a replay.
"""

import cv2
import numpy as np

from redlight.detection import Detection
from redlight.judge import (
    DIFF_PIXEL_DELTA,
    WINDOW,
    MotionJudge,
    crop_window,
    diff_score,
    diff_score_window,
    flow_score,
)

# The synthetic scene lives in "world" pixels: a canvas twice the size of the
# 1x frame, so 0.5x / 1x / 2x renders are all downsamples of one source of
# truth rather than three separately drawn scenes.
WORLD_W, WORLD_H = 1280, 960
BASE_W, BASE_H = 640, 480
BASE_BOX = (160, 80, 160, 320)  # x, y, w, h at 1x


def _world(seed: int = 7) -> np.ndarray:
    """A deterministic textured canvas with features big enough to survive 0.5x."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, size=(15, 20)).astype(np.uint8)
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


def test_frozen_scores_zero():
    """A player holding still scores exactly zero on both metrics."""
    frame, _, box = _pair(1.0, 0)

    assert flow_score(frame, frame, box, 0.1) == 0.0
    assert diff_score(frame, frame, box, 0.1) == 0.0


def test_sensor_noise_stays_under_deadband():
    """Camera grain alone must not read as movement, at any frame rate."""
    frame, _, box = _pair(1.0, 0)
    rng = np.random.default_rng(1234)
    noisy = np.clip(
        frame.astype(np.int16) + rng.normal(0.0, 2.0, frame.shape).round(), 0, 255
    ).astype(np.uint8)

    assert flow_score(frame, noisy, box, 0.1) < 0.02
    assert diff_score(frame, noisy, box, 0.1) == 0.0


def test_score_is_resolution_invariant():
    """The same movement scores the same whether the player fills the frame or not.

    Cropping at the box and resampling to a fixed window is what buys this:
    a stride is a stride, measured against the player's own apparent size.
    """
    dx_world = 16
    scores = {}
    frozen = {}
    for scale in (0.5, 1.0, 2.0):
        prev, cur, box = _pair(scale, dx_world)
        scores[scale] = flow_score(prev, cur, box, 0.1)
        frozen[scale] = flow_score(prev, prev, box, 0.1)

    values = list(scores.values())
    assert min(values) > 0.0, scores
    assert max(values) / min(values) <= 1.35, scores

    # Decisive, not merely different: every moving score clears its own
    # frozen baseline by more than 10x.
    for scale, score in scores.items():
        assert score > 10 * max(frozen[scale], 0.001), (scale, score, frozen[scale])


def test_score_is_framerate_invariant():
    """Halving the step and halving the interval leaves the score unchanged.

    A player crossing the same distance is judged the same whether the
    machine saw it in one frame or two.
    """
    full = flow_score(*_pair(1.0, 16), 0.2)
    half = flow_score(*_pair(1.0, 8), 0.1)

    assert full > 0.0 and half > 0.0
    assert abs(half - full) / full <= 0.30, (half, full)

    # And a still player reads as still at any frame rate.
    frame, _, box = _pair(1.0, 0)
    assert flow_score(frame, frame, box, 0.033) == 0.0
    assert flow_score(frame, frame, box, 0.1) == 0.0


def test_degenerate_box_returns_none():
    """Boxes too small or off-frame yield no score rather than a garbage one."""
    frame, cur, _ = _pair(1.0, 16)

    sliver = Detection(10, 10, 4, 200)
    offscreen = Detection(BASE_W + 50, BASE_H + 50, 100, 100)

    for box in (sliver, offscreen):
        assert crop_window(frame, box) is None
        assert flow_score(frame, cur, box, 0.1) is None
        assert diff_score(frame, cur, box, 0.1) is None


def test_nonpositive_dt_returns_none():
    """Without a positive time step there is no per-second rate to report."""
    prev, cur, box = _pair(1.0, 16)

    for dt in (0.0, -0.1):
        assert flow_score(prev, cur, box, dt) is None
        assert diff_score(prev, cur, box, dt) is None


def test_judge_requires_n_consecutive():
    """One twitchy frame is not an elimination; three in a row is."""
    judge = MotionJudge(threshold=0.5, confirm_frames=3, smoothing=0.5)
    assert [judge.update(1, s) for s in (1.0, 1.0)] == [False, False]

    judge.reset()
    assert [judge.update(1, s) for s in (1.0, 1.0, 1.0)] == [False, False, True]

    judge.reset()
    fired = [judge.update(1, s) for s in (1.0, 0.0, 1.0, 1.0, 1.0)]
    assert fired == [False, False, False, False, True]


def test_judge_reset_clears_state():
    """A new round starts from a clean slate, not the last round's baseline."""
    judge = MotionJudge(threshold=0.5, confirm_frames=3, smoothing=0.5)
    for _ in range(5):
        judge.update(1, 1.0)
    assert judge.smoothed(1) > 0.9

    judge.reset()
    assert judge.smoothed(1) == 0.0
    assert judge.update(1, 1.0) is False  # streak gone too, not just the EMA
    assert judge.smoothed(1) == 1.0  # first score after reset seeds the EMA


def test_judge_smoothed_unknown_track_is_zero():
    """A player the judge has never scored reads as perfectly still."""
    assert MotionJudge(threshold=0.5).smoothed(99) == 0.0


def test_diff_window_matches_manual():
    """The diff metric is integer-exact, so a port can be checked pixel for pixel."""
    prev = np.full((WINDOW, WINDOW), 100, dtype=np.uint8)
    cur = prev.copy()
    cur[0, :9] = 100 + DIFF_PIXEL_DELTA + 1  # exactly 9 pixels over the delta

    dt = 0.1
    assert diff_score_window(prev, cur, dt) == 9 / (WINDOW * WINDOW) / dt

    # A change of exactly DIFF_PIXEL_DELTA is under the bar, not over it.
    edge = prev.copy()
    edge[0, :9] = 100 + DIFF_PIXEL_DELTA
    assert diff_score_window(prev, edge, dt) == 0.0


def test_crop_window_shape_and_dtype():
    """Every crop reaches the judge as the same fixed-size grayscale window."""
    frame, _, box = _pair(1.0, 0)
    win = crop_window(frame, box)
    assert win is not None
    assert win.shape == (WINDOW, WINDOW)
    assert win.dtype == np.uint8


def test_crop_window_clamps_to_frame():
    """A box hanging off the edge is trimmed, not allowed to read past it."""
    frame, _, _ = _pair(1.0, 0)
    overhang = Detection(BASE_W - 40, BASE_H - 60, 200, 200)
    assert crop_window(frame, overhang) is not None
