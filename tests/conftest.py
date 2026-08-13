"""Test configuration and fixtures."""

import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_video(tmp_path):
    """Write a tiny 8-frame, 64x48 video and return its path as a string.

    Each frame is a solid gray fill at a different brightness so frames are
    distinguishable if a test needs to check content, but the fixture exists
    mainly to exercise FrameSource against a real (if minimal) video file.
    """
    path = tmp_path / "synthetic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10.0
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 48))
    for i in range(8):
        frame = np.full((48, 64, 3), i * 30, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return str(path)
