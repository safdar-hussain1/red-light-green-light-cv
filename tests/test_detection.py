"""Tests for person detectors."""

import cv2
import numpy as np
import pytest

from redlight.config import ConfigError
from redlight.detection import Detection, _hog_passes_threshold, make_detector
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


class TestHogPassesThreshold:
    """Unit tests on the pure filtering function, independent of OpenCV."""

    def test_weight_above_threshold_passes(self):
        assert _hog_passes_threshold(weight=1.2, conf=0.5) is True

    def test_weight_below_threshold_fails(self):
        assert _hog_passes_threshold(weight=0.1, conf=0.5) is False

    def test_weight_equal_to_threshold_passes(self):
        assert _hog_passes_threshold(weight=0.5, conf=0.5) is True


class TestHogDetectorAppliesConfThreshold:
    """make_detector("hog", conf) must actually filter by conf, like YOLO does."""

    def test_high_threshold_filters_out_weak_detections(self, monkeypatch):
        # Three candidate boxes whose SVM weights straddle both thresholds
        # below: 0.1 is weak, 0.6 and 1.2 are progressively stronger matches.
        boxes = np.array([[0, 0, 10, 10], [20, 20, 10, 10], [40, 40, 10, 10]])
        weights = np.array([0.1, 0.6, 1.2])

        def fake_detect_multi_scale(self, frame_bgr, **kwargs):
            return boxes, weights

        monkeypatch.setattr(
            cv2.HOGDescriptor, "detectMultiScale", fake_detect_multi_scale, raising=True
        )

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        low_threshold_detector = make_detector("hog", 0.2)
        high_threshold_detector = make_detector("hog", 0.8)

        low_count = len(low_threshold_detector.detect(frame))
        high_count = len(high_threshold_detector.detect(frame))

        # low=0.2 clears weights 0.6 and 1.2 -> 2 detections.
        # high=0.8 clears only weight 1.2 -> 1 detection.
        assert low_count == 2
        assert high_count == 1
        assert high_count < low_count


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
