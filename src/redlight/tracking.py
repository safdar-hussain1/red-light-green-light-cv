"""Multi-object tracking: assigns stable IDs to detections across frames.

Pure geometry — no video I/O or model inference here, just greedy IoU
matching between the previous frame's tracks and the current frame's
detections.
"""

from __future__ import annotations

from dataclasses import dataclass

from redlight.detection import Detection


@dataclass
class Track:
    """A tracked person: a stable id, its current box, and its miss streak."""

    track_id: int
    box: Detection
    misses: int = 0


def _iou(a: Detection, b: Detection) -> float:
    """Intersection-over-union of two boxes given as (x, y, w, h)."""
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0

    area_a = a.w * a.h
    area_b = b.w * b.h
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class Tracker:
    """Greedily matches detections to prior tracks by IoU, keeping ids stable.

    Unmatched tracks coast for up to max_misses frames before dying, so a
    person who is briefly missed (occlusion, a dropped detection) keeps
    their id instead of getting a new one when they reappear.
    """

    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 15):
        self._iou_threshold = iou_threshold
        self._max_misses = max_misses
        self._tracks: dict[int, Track] = {}
        self._next_id = 1
        self._dead_ids: set[int] = set()

    def update(self, detections: list[Detection]) -> list[Track]:
        """Match detections to existing tracks, age misses, spawn new tracks.

        Returns all currently live tracks: matched this frame (misses == 0)
        or coasting (misses <= max_misses).
        """
        candidates = []
        for track_id, track in self._tracks.items():
            for det_index, det in enumerate(detections):
                iou = _iou(track.box, det)
                if iou >= self._iou_threshold:
                    candidates.append((iou, track_id, det_index))
        candidates.sort(key=lambda c: c[0], reverse=True)

        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()
        for iou, track_id, det_index in candidates:
            if track_id in matched_track_ids or det_index in matched_det_indices:
                continue
            matched_track_ids.add(track_id)
            matched_det_indices.add(det_index)
            self._tracks[track_id] = Track(
                track_id=track_id, box=detections[det_index], misses=0
            )

        for track_id, track in list(self._tracks.items()):
            if track_id in matched_track_ids:
                continue
            track.misses += 1
            if track.misses > self._max_misses:
                self._dead_ids.add(track_id)
                del self._tracks[track_id]

        for det_index, det in enumerate(detections):
            if det_index in matched_det_indices:
                continue
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = Track(track_id=track_id, box=det, misses=0)

        return list(self._tracks.values())

    def pop_dead(self) -> set[int]:
        """Return ids that exceeded max_misses since the last call, then clear them."""
        dead_ids = self._dead_ids
        self._dead_ids = set()
        return dead_ids
