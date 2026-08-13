"""Video frame sources: webcams and video files, behind one interface."""

from __future__ import annotations

import time

import cv2
import numpy as np


class SourceError(RuntimeError):
    """Raised when a camera index or video path cannot be opened."""


class FrameSource:
    """Reads BGR frames with timestamps from a camera index or video file.

    For files, the timestamp is derived from the frame index and the file's
    reported fps, so playback is reproducible. For live cameras there is no
    reliable per-frame fps, so the timestamp is a wall-clock monotonic clock
    reading taken at read time.
    """

    def __init__(self, source: str | int):
        self._source = source
        self._is_file = isinstance(source, str)
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            self._cap.release()
            raise SourceError(f"could not open video source: {source!r}")

        reported_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps: float | None = reported_fps if reported_fps and reported_fps > 0 else None
        self._frame_index = 0

    def read(self) -> tuple[np.ndarray, float] | None:
        """Return the next (bgr frame, timestamp seconds), or None at the end."""
        ok, frame = self._cap.read()
        if not ok:
            return None

        if self._is_file:
            fps = self.fps or 30.0
            timestamp = self._frame_index / fps
        else:
            timestamp = time.monotonic()
        self._frame_index += 1

        return frame, timestamp

    def release(self) -> None:
        """Release the underlying capture device or file handle."""
        self._cap.release()

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
