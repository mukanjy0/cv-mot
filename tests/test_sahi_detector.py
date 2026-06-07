from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.detection.sahi_ultralytics_yolo import SahiUltralyticsYoloDetector


class SahiDetectorTest(unittest.TestCase):
    def test_maps_predictions_and_converts_bgr_to_rgb(self) -> None:
        predictions = [
            SimpleNamespace(
                category=SimpleNamespace(id=0),
                bbox=SimpleNamespace(to_xyxy=lambda: [1, 2, 11, 22]),
                score=SimpleNamespace(value=0.8),
            ),
            SimpleNamespace(
                category=SimpleNamespace(id=5),
                bbox=SimpleNamespace(to_xyxy=lambda: [3, 4, 13, 24]),
                score=SimpleNamespace(value=0.7),
            ),
        ]

        def fake_sliced_prediction(image: np.ndarray, _model: object, **_: object):
            np.testing.assert_array_equal(image[0, 0], np.array([30, 20, 10]))
            return SimpleNamespace(object_prediction_list=predictions)

        with patch("sahi.AutoDetectionModel.from_pretrained", return_value=object()):
            with patch(
                "sahi.predict.get_sliced_prediction",
                side_effect=fake_sliced_prediction,
            ):
                detector = SahiUltralyticsYoloDetector(model_path="unused.pt")
                frame = np.array([[[10, 20, 30]]], dtype=np.uint8)
                detections = detector.predict(frame, frame_id=7)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].frame_id, 7)
        self.assertEqual(detections[0].cls, 1)
        self.assertEqual(detections[0].conf, 0.8)


if __name__ == "__main__":
    unittest.main()
