"""Person detection: locate people in a single video frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from redlight.config import ConfigError

PERSON_CLASS_ID = 0
DEFAULT_YOLO_WEIGHTS = "models/yolo11n.pt"


@dataclass(frozen=True)
class Detection:
    """A person's bounding box in pixel coordinates, top-left + size."""

    x: int
    y: int
    w: int
    h: int
    conf: float = 1.0


class Detector(Protocol):
    """Anything that can find people in a BGR frame."""

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...


class YoloDetector:
    """Finds people with a YOLO11 model, filtered to the person class."""

    def __init__(self, weights_path: str = DEFAULT_YOLO_WEIGHTS, conf: float = 0.35):
        from ultralytics import YOLO

        self._model = YOLO(weights_path)
        self._conf = conf

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            frame_bgr,
            classes=[PERSON_CLASS_ID],
            conf=self._conf,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        x=int(round(x1)),
                        y=int(round(y1)),
                        w=int(round(x2 - x1)),
                        h=int(round(y2 - y1)),
                        conf=float(box.conf[0]),
                    )
                )
        return detections


def _hog_passes_threshold(weight: float, conf: float) -> bool:
    """Whether a raw HOG/SVM decision value clears the confidence threshold.

    HOG's detectMultiScale reports an unbounded SVM decision value, not a
    [0,1] probability like YOLO's confidence. Rather than invent a squashing
    function with no principled basis, `conf` is compared directly against
    that raw scale: higher means a stronger match, roughly 0.5-3.0 for solid
    detections and near/below 0 for weak or spurious ones. Callers using the
    HOG detector should tune `conf` empirically on that raw scale.
    """
    return weight >= conf


class HogDetector:
    """Finds people with OpenCV's built-in HOG + linear SVM detector.

    The classical option: no model weights to download, at the cost of
    accuracy compared to the YOLO detector -- and, on the committed benchmark,
    3.5x the time per frame.
    """

    def __init__(self, conf: float = 0.35):
        self._conf = conf
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        boxes, weights = self._hog.detectMultiScale(
            frame_bgr, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        detections: list[Detection] = []
        for (x, y, w, h), weight in zip(boxes, weights):
            weight = float(weight)
            if not _hog_passes_threshold(weight, self._conf):
                continue
            detections.append(Detection(x=int(x), y=int(y), w=int(w), h=int(h), conf=weight))
        return detections


def make_detector(name: str, conf: float) -> Detector:
    """Build a detector by name: "yolo" (accurate, needs model weights) or
    "hog" (the weights-free classical option: no model download, and 3.5x
    slower than yolo on the committed benchmark).

    Raises:
        ConfigError: If name is not a recognized detector.
    """
    if name == "yolo":
        return YoloDetector(conf=conf)
    if name == "hog":
        return HogDetector(conf=conf)
    raise ConfigError(f"unknown detector {name!r}; expected 'yolo' or 'hog'")
