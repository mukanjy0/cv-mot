from __future__ import annotations

import unittest

import numpy as np

from src.evaluation.evaluate_detection import BoxRecord, evaluate_records


class EvaluateDetectionTest(unittest.TestCase):
    def test_greedy_counts_and_ap50_by_frame_and_class(self) -> None:
        gt = [
            BoxRecord("seq", 1, 4, np.array([0, 0, 10, 10], dtype=float)),
            BoxRecord("seq", 2, 4, np.array([0, 0, 10, 10], dtype=float)),
        ]
        pred = [
            BoxRecord("seq", 1, 4, np.array([0, 0, 10, 10], dtype=float), conf=0.9),
            BoxRecord("seq", 2, 4, np.array([30, 30, 40, 40], dtype=float), conf=0.8),
            BoxRecord("seq", 1, 1, np.array([0, 0, 10, 10], dtype=float), conf=0.7),
        ]

        metrics = evaluate_records(gt, pred)

        self.assertEqual(metrics.gt_boxes, 2)
        self.assertEqual(metrics.predicted_boxes, 3)
        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.false_positives, 2)
        self.assertEqual(metrics.false_negatives, 1)
        self.assertAlmostEqual(metrics.precision or 0.0, 1 / 3)
        self.assertAlmostEqual(metrics.recall or 0.0, 0.5)
        self.assertAlmostEqual(metrics.ap50 or 0.0, 0.5)


if __name__ == "__main__":
    unittest.main()
