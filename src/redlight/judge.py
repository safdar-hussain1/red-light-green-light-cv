"""Motion scoring and the elimination decision.

The referee has one hard problem: deciding that a player moved, in a way
that means the same thing for everybody. A player standing at the back of
the room covers fewer pixels per step than one standing at the front, and
a laptop pushing 60 frames a second sees smaller steps between frames than
one struggling at 12. Score raw pixels and both of those turn into unfair
calls.

So motion here is measured in **body-fractions per second**: how far a
player moved in one second, relative to their own apparent size on camera.

Cropping at the player's box and resampling to a fixed window is what makes
a step mean the same thing whether the player is near or far, and whether
the camera is 480p or 4K. Distance and resolution divide out, for both
metrics.

Frame rate is handled differently by the two metrics, and it is worth being
precise about which is which:

* `flow_score` measures displacement, which really is proportional to the
  time between frames, so dividing by dt genuinely turns it into a rate.
  That holds within the operating band documented on the function.
* `diff_score` counts changed pixels, and that count grows faster than
  linearly with displacement — a bigger step does not just move more
  pixels, it moves them past the change threshold. Dividing by dt does not
  rescue it. The diff path is made frame-rate independent a different way:
  `FrameSampler` scores on a fixed clock, so the same real motion produces
  the same comparisons no matter how fast frames arrive.

A sub-pixel deadband finishes the job. Sensor grain jitters flow estimates
by a fraction of a pixel every frame, and dividing by a small dt would
amplify exactly that jitter into an elimination. Anything under the floor
is treated as stillness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from redlight.detection import Detection

WINDOW = 96
"""Side length of the fixed analysis window every player crop is resampled to."""

FLOW_NOISE_FLOOR_PX = 0.5
"""Sub-pixel deadband: flow below this is sensor grain, not a player moving."""

DIFF_PIXEL_DELTA = 25
"""How far a pixel's brightness must swing before it counts as changed."""

MIN_CROP_PX = 8
"""Crops thinner than this carry no usable texture, so they get no score."""

SAMPLE_INTERVAL_S = 0.1
"""How often the referee actually scores, regardless of how fast frames arrive."""

TIMESTAMP_TOLERANCE_S = 1e-9
"""Slack on the sampling interval, so a float division can't hide a frame.

A nanosecond is orders of magnitude below any real frame period, so this
never lets an early frame through — it only keeps a timestamp that is
exactly on the boundary in arithmetic from landing just under it in binary.
"""


def crop_window(gray: np.ndarray, box: Detection) -> np.ndarray | None:
    """Cut the player out of a grayscale frame and normalise them to one size.

    The box is clamped to the frame, so a player half out of shot is judged
    on the half that is visible rather than on out-of-bounds memory. What
    comes back is always a WINDOW x WINDOW patch: this resampling is what
    makes a near player and a far player comparable, since both end up
    measured against their own body rather than against the frame.

    A light blur takes the edge off single-pixel sensor grain before any
    scoring happens.

    Returns:
        The normalised window, or None if the visible part of the box is
        too small to say anything about.
    """
    frame_h, frame_w = gray.shape[:2]
    x1 = max(0, int(box.x))
    y1 = max(0, int(box.y))
    x2 = min(frame_w, int(box.x) + int(box.w))
    y2 = min(frame_h, int(box.y) + int(box.h))

    if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
        return None

    win = cv2.resize(gray[y1:y2, x1:x2], (WINDOW, WINDOW), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(win, (3, 3), 0)


def flow_score(
    prev_gray: np.ndarray, cur_gray: np.ndarray, box: Detection, dt: float
) -> float | None:
    """Score a player's movement between two frames, in body-fractions per second.

    Dense optical flow across the player's window gives a displacement for
    every pixel in it. The 95th percentile of those magnitudes is the
    headline number: it tracks the fastest-moving part of the player — a
    hand, a shifting foot — while ignoring the handful of outliers a flow
    estimator always produces. The noise floor is subtracted before the
    result is divided by the window size and the elapsed time.

    The score is frame-rate independent across steps of roughly 2.2 to 7
    pixels of window displacement. Below that band the fixed sub-pixel
    deadband takes a disproportionate bite out of the smaller step; above
    it the flow estimator saturates and over-reads, so a fast step scores
    higher than its true displacement. Over-reading errs toward calling
    movement, which is the safe direction for a referee, but it does mean
    the threshold is not a literal body-fraction figure at the fast end.

    Returns:
        Movement in body-fractions per second, or None if the box is
        unusable or dt is not positive.
    """
    if dt <= 0:
        return None

    prev_win = crop_window(prev_gray, box)
    cur_win = crop_window(cur_gray, box)
    if prev_win is None or cur_win is None:
        return None

    flow = cv2.calcOpticalFlowFarneback(
        prev_win,
        cur_win,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    p95 = float(np.percentile(mag, 95))
    return max(p95 - FLOW_NOISE_FLOOR_PX, 0.0) / WINDOW / dt


def diff_score_window(prev_win: np.ndarray, cur_win: np.ndarray, dt: float) -> float:
    """Fraction of the window that changed brightness, per second.

    The cheap metric, for machines that cannot afford dense flow: count the
    pixels whose brightness moved by more than DIFF_PIXEL_DELTA, express
    that as a fraction of the window, and divide by the elapsed time.

    Dividing by dt puts the result on a per-second footing but does **not**
    make it frame-rate independent the way it does for flow. A changed-pixel
    count grows faster than linearly with displacement, so halving the step
    and halving the interval does not leave the score where it was. Callers
    that need the diff metric to give the same verdict at any frame rate
    must feed it through `FrameSampler`, which fixes the interval instead.

    This function is deliberately plain integer arithmetic on two uint8
    windows — no blurring, no resizing, no floating-point thresholds — so
    that the browser build can run the identical calculation. It is designed
    to be verified against shared fixtures for bit-exact agreement, so any
    change here has to be mirrored there exactly.

    Args:
        prev_win: Earlier WINDOW x WINDOW uint8 window.
        cur_win: Later WINDOW x WINDOW uint8 window.
        dt: Seconds between the two windows; must be positive.

    Raises:
        ValueError: If dt is not positive, or either window is not a
            WINDOW x WINDOW uint8 array. This is the function the port is
            held against, so a wrong-shaped input is a bug to surface, not
            something to quietly score.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    expected = (WINDOW, WINDOW)
    for name, win in (("prev_win", prev_win), ("cur_win", cur_win)):
        if win.shape != expected or win.dtype != np.uint8:
            raise ValueError(
                f"{name} must be a {expected} uint8 array, "
                f"got shape {win.shape} dtype {win.dtype}"
            )

    delta = np.abs(prev_win.astype(np.int16) - cur_win.astype(np.int16))
    changed = int(np.count_nonzero(delta > DIFF_PIXEL_DELTA))
    return changed / (WINDOW * WINDOW) / dt


def diff_score(
    prev_gray: np.ndarray, cur_gray: np.ndarray, box: Detection, dt: float
) -> float | None:
    """Frame-difference movement score for a player, in body-fractions per second.

    Normalises both frames through the same crop-and-resample path as the
    flow metric so the two are on comparable footing, then counts changed
    pixels.

    Returns:
        Changed fraction per second, or None if the box is unusable or dt
        is not positive.
    """
    if dt <= 0:
        return None

    prev_win = crop_window(prev_gray, box)
    cur_win = crop_window(cur_gray, box)
    if prev_win is None or cur_win is None:
        return None

    return diff_score_window(prev_win, cur_win, dt)


class FrameSampler:
    """Paces scoring on a fixed clock instead of on whatever the camera manages.

    A capture loop hands over every frame it grabs. This holds them back and
    releases a pair only once at least `interval_s` has passed since the last
    pair it released, so comparisons are always made across roughly the same
    slice of real time. A machine running at 60 frames a second and one
    running at 30 end up judging the same motion over the same intervals.

    That is what makes the diff metric fair across frame rates: rather than
    trying to rescale a changed-pixel count that does not scale linearly, the
    interval it is measured over is simply held fixed.

    The comparison is always against the last *released* frame, never the last
    frame offered, so nothing is silently measured over a shorter gap.

    The interval is compared with a small tolerance, and that tolerance is
    load-bearing rather than cosmetic. A camera running at a clean multiple
    of the interval — 10 fps against a 0.1 s clock is the obvious case —
    produces timestamps like `i / 10.0`, and in binary floating point the gap
    between two of those lands a fraction of an ulp under 0.1 about half the
    time. Compared exactly, the sampler would reject every other frame that
    is precisely on the boundary and pace itself at half the rate it was
    asked for, with the interval alternating between one and two frame
    periods. `TIMESTAMP_TOLERANCE_S` is far below any real frame period, so
    it cannot admit a frame that is genuinely early; it only stops the last
    bit of a float division from being read as earliness.
    """

    def __init__(self, interval_s: float = SAMPLE_INTERVAL_S):
        self._interval_s = interval_s
        self._last_gray: np.ndarray | None = None
        self._last_ts: float | None = None

    def offer(
        self, gray: np.ndarray, ts: float
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Hand the sampler a frame and its timestamp in seconds.

        Returns:
            (previous_sampled_frame, this_frame, seconds_between_them) when
            enough time has passed to be worth scoring, otherwise None. The
            very first frame only primes the sampler and returns None.
        """
        if self._last_ts is None:
            self._last_gray = gray
            self._last_ts = ts
            return None

        dt = ts - self._last_ts
        if dt < self._interval_s - TIMESTAMP_TOLERANCE_S:
            return None

        previous = self._last_gray
        self._last_gray = gray
        self._last_ts = ts
        return previous, gray, dt

    def reset(self) -> None:
        """Drop the held frame so the next offer primes a fresh interval."""
        self._last_gray = None
        self._last_ts = None


@dataclass
class MotionJudge:
    """Turns a stream of per-frame scores into eliminate / don't-eliminate calls.

    Raw scores flicker: a flow estimate spikes on a lighting change, a
    detection box wobbles by a few pixels. Calling someone out on a single
    spiky frame is the fastest way to make the referee feel unfair, so two
    guards sit in front of the decision.

    An exponential moving average per player smooths the score, weighting
    the newest frame by `smoothing`. On top of that, the smoothed score has
    to stay above the threshold for `confirm_frames` frames in a row before
    anyone is called out; a single frame back at or below the threshold
    puts the streak back to zero. The player has to actually be moving, not
    have flickered once.

    Attributes:
        threshold: Movement, in body-fractions per second, that counts as moving.
        confirm_frames: Consecutive frames above threshold before a call is made.
        smoothing: Weight on the newest score in the moving average, in (0, 1].
    """

    threshold: float
    confirm_frames: int = 3
    smoothing: float = 0.5
    _ema: dict[int, float] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _streak: dict[int, int] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def update(self, track_id: int, score: float) -> bool:
        """Feed one frame's score for one player and get the verdict.

        Returns:
            True if this player has now been above the threshold for
            `confirm_frames` frames running.
        """
        previous = self._ema.get(track_id)
        # The first score a player is ever given seeds their average — there
        # is no earlier frame to blend it with, and starting from zero would
        # hand everyone a free frame of movement.
        ema = score if previous is None else self.smoothing * score + (1 - self.smoothing) * previous
        self._ema[track_id] = ema

        if ema > self.threshold:
            self._streak[track_id] = self._streak.get(track_id, 0) + 1
        else:
            self._streak[track_id] = 0

        return self._streak[track_id] >= self.confirm_frames

    def smoothed(self, track_id: int) -> float:
        """The player's current smoothed score, or 0.0 if they have none yet."""
        return self._ema.get(track_id, 0.0)

    def reset(self) -> None:
        """Forget every player's history, ready for a fresh round.

        Rounds must not inherit a stale baseline: a player who was sprinting
        when the last round ended starts the next one as still as everyone else.
        """
        self._ema.clear()
        self._streak.clear()
