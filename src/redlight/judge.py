"""Motion scoring and the elimination decision.

The referee has one hard problem: deciding that a player moved, in a way
that means the same thing for everybody. A player standing at the back of
the room covers fewer pixels per step than one standing at the front, and
a laptop pushing 60 frames a second sees smaller steps between frames than
one struggling at 12. Score raw pixels and both of those turn into unfair
calls.

So motion here is measured in **body-fractions per second**: how far a
player moved in one second, relative to their own apparent size on camera.
Two things buy that:

* Cropping at the player's box and resampling to a fixed window makes a
  step mean the same thing whether the player is near or far, and whether
  the camera is 480p or 4K. Distance and resolution divide out.
* Dividing by the elapsed time between the two frames makes the score a
  rate rather than a per-frame increment, so frame rate divides out too.

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

    This function is deliberately plain integer arithmetic on two uint8
    windows — no blurring, no resizing, no floating-point thresholds. The
    browser build runs the identical calculation, and the two are held to
    bit-exact agreement against shared fixtures, so any change here has to
    be mirrored there exactly.

    Args:
        prev_win: Earlier WINDOW x WINDOW uint8 window.
        cur_win: Later WINDOW x WINDOW uint8 window.
        dt: Seconds between the two windows; must be positive.

    Raises:
        ValueError: If dt is not positive.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

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
    _ema: dict[int, float] = field(default_factory=dict, repr=False, compare=False)
    _streak: dict[int, int] = field(default_factory=dict, repr=False, compare=False)

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
