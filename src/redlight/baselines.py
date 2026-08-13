"""Naive referee designs: the first ideas anyone reaches for.

`redlight.judge` measures a player's movement in body-fractions per second
by cropping their box out of the frame and resampling it to a fixed window
before scoring — that resample is what makes a stride mean the same thing
at any distance from the camera or any capture resolution. Before settling
on that design, it's worth writing down the simpler ones that come to mind
first, and being precise about where each one breaks. The benchmark this
package ships with measures against these three, not against a straw man:
each is implemented the way it would actually be written, with a sane
default, and given every chance to do well on easy footage.

None of the three resamples anything. That single omission is the common
thread behind all three failure modes below.
"""

from __future__ import annotations

import cv2
import numpy as np

from redlight.detection import Detection

MIN_CROP_PX = 8
"""Crops thinner than this carry no usable texture, so they get no score."""


def brightness_count(gray: np.ndarray, box: Detection, threshold_value: int = 30) -> float:
    """A naive referee design: count bright pixels inside the player's box.

    The idea is that a moving player disturbs enough of the scene that
    counting bright pixels stands in for counting motion. It never compares
    one frame to the next — the function takes exactly one frame — so it
    has no temporal term at all. Two very different histories that happen
    to land on the same frame score identically, because the function only
    ever sees that one frame; it cannot tell a player who just arrived at a
    pose from one who has held it the whole time.

    It is also size-dependent: this is a raw pixel total, not a fraction of
    the box, so a player who fills more of the frame reports more "motion"
    than a distant player doing the exact same thing.

    Args:
        gray: A grayscale frame.
        box: The player's box within it.
        threshold_value: Brightness a pixel must clear to count as bright.

    Returns:
        The count of pixels in the box strictly brighter than threshold_value.
    """
    frame_h, frame_w = gray.shape[:2]
    x1 = max(0, int(box.x))
    y1 = max(0, int(box.y))
    x2 = min(frame_w, int(box.x) + int(box.w))
    y2 = min(frame_h, int(box.y) + int(box.h))

    roi = gray[y1:y2, x1:x2]
    return float(np.count_nonzero(roi > threshold_value))


def frame_diff_area(prev_gray: np.ndarray, cur_gray: np.ndarray, pixel_delta: int = 25) -> float:
    """A naive referee design: count every changed pixel across the whole frame.

    This is frame differencing before anyone thinks to crop it to a player:
    subtract one frame from the next and count the pixels whose brightness
    moved by more than pixel_delta. There is no box argument, which is the
    failure mode stated as plainly as possible — the function cannot
    distinguish the player moving from anything else moving anywhere in the
    shot. A second player shifting their weight, a curtain stirring, someone
    walking past in the background: all of it adds to the same total as the
    motion actually being judged.

    Args:
        prev_gray: Earlier grayscale frame.
        cur_gray: Later grayscale frame, same shape as prev_gray.
        pixel_delta: Brightness swing a pixel must clear to count as changed.

    Returns:
        The count of changed pixels across the whole frame.
    """
    delta = np.abs(prev_gray.astype(np.int16) - cur_gray.astype(np.int16))
    return float(np.count_nonzero(delta > pixel_delta))


def raw_flow_max(
    prev_gray: np.ndarray, cur_gray: np.ndarray, box: Detection
) -> float | None:
    """A naive referee design: dense optical flow on the player's box, unresampled.

    This is what `redlight.judge.flow_score` would look like without the
    fixed-size resample its `crop_window` does first: crop the box out of
    each frame, run Farneback on it at whatever raw pixel size it happens to
    be, and report the largest displacement found. Skipping the resample
    makes the result resolution-dependent — the same body-fraction of
    movement produces more raw pixels of flow the bigger the box is on
    screen, so a player close to the camera, or the same footage captured at
    a higher resolution, reads as moving faster than a player further away
    or on a lower-resolution feed doing the identical thing.

    Args:
        prev_gray: Earlier grayscale frame.
        cur_gray: Later grayscale frame, same shape as prev_gray.
        box: The player's box in both frames.

    Returns:
        The max flow magnitude in raw pixels, or None if the visible part of
        the box is too small to say anything about.
    """
    frame_h, frame_w = prev_gray.shape[:2]
    x1 = max(0, int(box.x))
    y1 = max(0, int(box.y))
    x2 = min(frame_w, int(box.x) + int(box.w))
    y2 = min(frame_h, int(box.y) + int(box.h))

    if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
        return None

    prev_roi = prev_gray[y1:y2, x1:x2]
    cur_roi = cur_gray[y1:y2, x1:x2]

    flow = cv2.calcOpticalFlowFarneback(
        prev_roi,
        cur_roi,
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
    return float(np.max(mag))
