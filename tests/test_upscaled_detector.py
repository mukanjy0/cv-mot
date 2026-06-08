from __future__ import annotations

import unittest

import numpy as np

from src.core.types import Detection
from src.detection.base import Detector
from src.detection.upscaled_detector import UpscaledDetector


class FakeDetector(Detector):
    def __init__(self) -> None:
        self.seen_shape: tuple[int, int] | None = None

    def predict(self, frame_bgr: np.ndarray, frame_id: int) -> list[Detection]:
        self.seen_shape = frame_bgr.shape[:2]
        return [
            Detection(
                frame_id=frame_id,
                xyxy=np.array([15, 6, 30, 18], dtype=float),
                conf=0.8,
                cls=4,
            )
        ]


class UpscaledDetectorTest(unittest.TestCase):
    def test_scales_input_and_maps_boxes_back(self) -> None:
        inner = FakeDetector()
        detector = UpscaledDetector(inner, scale=1.5)
        frame = np.zeros((20, 30, 3), dtype=np.uint8)

        detections = detector.predict(frame, frame_id=3)

        self.assertEqual(inner.seen_shape, (30, 45))
        self.assertEqual(len(detections), 1)
        np.testing.assert_allclose(
            detections[0].xyxy,
            np.array([10, 4, 20, 12], dtype=float),
        )


if __name__ == "__main__":
    unittest.main()
