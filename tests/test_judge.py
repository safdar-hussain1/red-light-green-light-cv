"""Tests for the motion judge — the claims the referee's fairness rests on.

The judge has to give the same verdict for the same movement whether the
player is close to the camera or far from it, and whether the machine is
managing 10 frames a second or 60. These tests pin those two claims down
with a synthetic scene we can move by an exact number of pixels, so a
regression shows up as a number rather than as an argument about a replay.
"""

import cv2
import numpy as np
import pytest

from redlight.detection import Detection
from redlight.judge import (
    DIFF_PIXEL_DELTA,
    FLOW_NOISE_FLOOR_PX,
    SAMPLE_INTERVAL_S,
    TIMESTAMP_TOLERANCE_S,
    WINDOW,
    FrameSampler,
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


def _grainy(frame: np.ndarray, sigma: float = 2.0, seed: int = 1234) -> np.ndarray:
    """The same frame as seen through a noisy sensor."""
    rng = np.random.default_rng(seed)
    return np.clip(
        frame.astype(np.int16) + rng.normal(0.0, sigma, frame.shape).round(), 0, 255
    ).astype(np.uint8)


def test_frozen_scores_zero():
    """A player holding still scores exactly zero on both metrics."""
    frame, _, box = _pair(1.0, 0)

    assert flow_score(frame, frame, box, 0.1) == 0.0
    assert diff_score(frame, frame, box, 0.1) == 0.0


def test_sensor_noise_stays_under_deadband():
    """Camera grain alone must not read as movement, at any frame rate."""
    frame, _, box = _pair(1.0, 0)
    noisy = _grainy(frame)

    assert flow_score(frame, noisy, box, 0.1) < 0.02
    assert diff_score(frame, noisy, box, 0.1) == 0.0


def test_score_is_resolution_invariant():
    """The same movement scores the same whether the player fills the frame or not.

    Cropping at the box and resampling to a fixed window is what buys this:
    a stride is a stride, measured against the player's own apparent size.
    """
    dt = 0.1
    dx_world = 20  # 6.0 px of window displacement, inside flow's honest band
    scores = {}
    still = {}
    for scale in (0.5, 1.0, 2.0):
        prev, cur, box = _pair(scale, dx_world)
        scores[scale] = flow_score(prev, cur, box, dt)
        still[scale] = flow_score(prev, _grainy(prev), box, dt)

    values = list(scores.values())
    assert min(values) > 0.0, scores
    assert max(values) / min(values) <= 1.35, scores

    # Decisive, not merely different. The baseline is a still player in
    # front of a noisy sensor, which is what the judge actually has to tell
    # movement apart from. That reads as exactly 0.0 because the deadband
    # clamps it, so the margin is measured against the smallest score the
    # metric can express at all — one deadband's worth — which keeps this a
    # real bar rather than a comparison against zero.
    floor = FLOW_NOISE_FLOOR_PX / WINDOW / dt
    for scale, score in scores.items():
        assert still[scale] < floor, (scale, still[scale])
        assert score > 10 * max(still[scale], floor), (scale, score, still[scale])


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


def test_diff_score_is_not_dt_invariant():
    """Characterisation: the diff metric does NOT survive a change of frame rate.

    Halving the step and halving the interval leaves flow where it was, but
    it does not do the same for a changed-pixel count. Twice the
    displacement pushes far more than twice as many pixels past the
    brightness delta, so the score is superlinear in the step and dividing
    by dt does not undo that.

    This is pinned deliberately rather than left implicit: the browser port
    inherits the same behaviour, and nobody should be able to read the /dt
    in the formula as a promise of frame-rate independence. The fix is
    FrameSampler, not arithmetic — see the test below.
    """
    prev, _, box = _pair(1.0, 0)
    world = _world()
    full = diff_score(prev, _render(_shift(world, 20), 1.0), box, 0.2)
    half = diff_score(prev, _render(_shift(world, 10), 1.0), box, 0.1)

    assert full > 0.0 and half > 0.0
    # Measurably lower, not marginally: it misses flow's 30% budget outright.
    assert half < full
    assert abs(half - full) / full > 0.30, (half, full)


def _diff_verdicts(fps: float, v_world: float, duration: float = 1.5) -> list[bool]:
    """Play a constant-velocity scene at `fps` and return the judge's calls.

    Frames arrive at the camera's rate; FrameSampler decides which pairs are
    actually scored, so the diff metric only ever sees a roughly fixed
    interval no matter how fast they come in.
    """
    world = _world()
    box = _box(1.0)
    sampler = FrameSampler()
    judge = MotionJudge(threshold=1.0, confirm_frames=3, smoothing=0.5)

    verdicts = []
    for i in range(int(round(duration * fps)) + 1):
        ts = i / fps
        frame = _render(_shift(world, int(round(v_world * ts))), 1.0)
        sampled = sampler.offer(frame, ts)
        if sampled is None:
            continue
        prev_frame, cur_frame, dt = sampled
        # The interval is honoured to within the float-boundary tolerance:
        # a 30 fps clock lands exactly on 0.1 and may render as a hair under.
        assert SAMPLE_INTERVAL_S - TIMESTAMP_TOLERANCE_S <= dt < 2 * SAMPLE_INTERVAL_S, dt
        score = diff_score_window(
            crop_window(prev_frame, box), crop_window(cur_frame, box), dt
        )
        verdicts.append(judge.update(1, score))
    return verdicts


def test_diff_verdicts_framerate_invariant_via_sampler():
    """The same real motion gets the same calls at 30fps and at 60fps.

    The diff score itself is not frame-rate independent, so the fixed
    sampling clock is what carries the claim: both machines compare frames
    about a tenth of a second apart, so both see the same real movement and
    reach the same verdict after the same number of scored samples.
    """
    moving_30 = _diff_verdicts(30, v_world=160)
    moving_60 = _diff_verdicts(60, v_world=160)
    still_30 = _diff_verdicts(30, v_world=0)
    still_60 = _diff_verdicts(60, v_world=0)

    # A faster camera yields more frames but not more scored samples; the
    # two streams may differ by one at the tail, so compare what overlaps.
    overlap = min(len(moving_30), len(moving_60))
    assert overlap >= 10
    assert moving_30[:overlap] == moving_60[:overlap]
    assert moving_30.index(True) == moving_60.index(True) == 2  # confirm_frames

    still_overlap = min(len(still_30), len(still_60))
    assert not any(still_30) and not any(still_60)
    assert still_30[:still_overlap] == still_60[:still_overlap]


def test_frame_sampler_holds_the_interval():
    """The sampler primes on the first frame and paces on its own clock."""
    frame = _render(_world(), 1.0)
    sampler = FrameSampler()

    assert sampler.offer(frame, 0.0) is None  # first frame only primes
    assert sampler.offer(frame, 0.05) is None  # too soon

    sampled = sampler.offer(frame, 0.12)
    assert sampled is not None
    _, _, dt = sampled
    # Measured against the last released frame, not the last one offered.
    assert dt == pytest.approx(0.12)

    assert sampler.offer(frame, 0.15) is None  # clock restarted from 0.12
    sampler.reset()
    assert sampler.offer(frame, 9.0) is None  # reset re-primes


def test_frame_sampler_releases_every_boundary_on_a_clean_frame_clock():
    """A 10 fps camera against a 0.1 s clock must release every frame.

    `i / 10.0` timestamps are the ordinary case, not a corner case, and in
    binary floating point about half of those gaps land a hair under 0.1.
    Compared exactly, the sampler would drop every other frame and pace
    itself at 5 fps with the interval flapping between 0.1 and 0.2 — so this
    pins the whole run, not just one boundary.
    """
    frame = _render(_world(), 1.0)
    sampler = FrameSampler()

    released = [sampler.offer(frame, i / 10.0) for i in range(200)]
    dts = [sampled[2] for sampled in released if sampled is not None]

    assert len(dts) == 199  # every frame but the one that primes the sampler
    assert all(dt == pytest.approx(0.1) for dt in dts)


def test_frame_sampler_still_holds_back_a_genuinely_early_frame():
    """The tolerance is for float dust, not for frames that are actually early."""
    frame = _render(_world(), 1.0)
    sampler = FrameSampler()

    assert sampler.offer(frame, 0.0) is None
    # A microsecond short is a thousand times the tolerance: still too soon.
    assert sampler.offer(frame, SAMPLE_INTERVAL_S - 1e-6) is None
    assert sampler.offer(frame, SAMPLE_INTERVAL_S) is not None


def test_diff_score_window_rejects_bad_windows():
    """The port-contract function refuses to score anything it cannot vouch for."""
    good = np.zeros((WINDOW, WINDOW), dtype=np.uint8)

    with pytest.raises(ValueError):
        diff_score_window(good, np.zeros((WINDOW, WINDOW), dtype=np.int16), 0.1)
    with pytest.raises(ValueError):
        diff_score_window(np.zeros((32, 32), dtype=np.uint8), good, 0.1)
    with pytest.raises(ValueError):
        diff_score_window(good, good, 0.0)


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

    # The load-bearing direction: a box whose origin is off the top-left.
    # Left unclamped, negative indices would silently wrap round and crop
    # the opposite corner of the frame, scoring a patch of background as if
    # it were the player. The crop must match the clamped region exactly.
    off_origin = Detection(-30, -20, 80, 90)
    clamped = Detection(0, 0, 80 - 30, 90 - 20)
    win = crop_window(frame, off_origin)
    assert win is not None
    assert np.array_equal(win, crop_window(frame, clamped))
