"""Tests for person detectors."""

import numpy as np
import pytest

from redlight.config import ConfigError
from redlight.detection import Detection, make_detector
from redlight.sources import FrameSource


class TestMakeDetectorErrors:
    def test_bogus_detector_name_raises_config_error(self):
        with pytest.raises(ConfigError):
            make_detector("bogus", 0.35)


class TestHogDetector:
    def test_blank_frame_has_no_detections(self):
        detector = make_detector("hog", 0.35)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert detector.detect(frame) == []


@pytest.mark.slow
class TestYoloDetector:
    def test_finds_several_persons_in_vtest_frame_600(self):
        with FrameSource("data/vtest.avi") as source:
            frame = None
            for _ in range(601):
                result = source.read()
                assert result is not None
                frame, _ = result

        detector = make_detector("yolo", 0.35)
        detections = detector.detect(frame)

        assert len(detections) >= 3
        height, width = frame.shape[:2]
        for det in detections:
            assert isinstance(det, Detection)
            assert det.w > 0
            assert det.h > 0
            assert 0 <= det.x
            assert 0 <= det.y
            assert det.x + det.w <= width
            assert det.y + det.h <= height
