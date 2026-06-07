from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.evaluate_mot import evaluate_mot


class EvaluateMotTest(unittest.TestCase):
    def test_perfect_tracks_and_frame_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            dataset_root = root / "dataset"
            sequence = "sequence_a"
            sequence_dir = dataset_root / "sequences" / sequence
            annotation_dir = dataset_root / "annotations"
            sequence_dir.mkdir(parents=True)
            annotation_dir.mkdir(parents=True)
            for frame_id in range(1, 4):
                (sequence_dir / f"{frame_id:07d}.jpg").touch()

            (annotation_dir / f"{sequence}.txt").write_text(
                "1,1,10,20,30,40,1,1,0,0\n"
                "2,1,12,20,30,40,1,1,0,0\n"
                "2,9,0,0,50,50,1,2,0,0\n"
                "3,1,14,20,30,40,1,1,0,0\n",
                encoding="utf-8",
            )

            tracks_root = root / "tracks" / "method_a"
            tracks_root.mkdir(parents=True)
            (tracks_root / f"{sequence}.txt").write_text(
                "1,7,10,20,30,40,0.9,1,-1\n"
                "2,7,12,20,30,40,0.9,1,-1\n"
                "3,7,500,500,30,40,0.9,1,-1\n",
                encoding="utf-8",
            )

            output_dir = root / "evaluation"
            result = evaluate_mot(
                dataset_root=dataset_root,
                tracks_root=root / "tracks",
                sequences=[sequence],
                output_dir=output_dir,
                max_frames=2,
            )

            summary = result["summary_by_method"][0]
            self.assertEqual(summary["method"], "method_a")
            self.assertEqual(summary["MOTA"], 1.0)
            self.assertEqual(summary["IDF1"], 1.0)
            self.assertEqual(summary["num_frames"], 2)
            self.assertEqual(summary["FP"], 0)
            self.assertEqual(summary["FN"], 0)

            with (output_dir / "per_sequence_metrics.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sequence_name"], sequence)

            payload = json.loads(
                (output_dir / "summary_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["classes"], [1, 4])

    def test_missing_selected_prediction_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            tracks_root = root / "tracks" / "method_a"
            tracks_root.mkdir(parents=True)
            (tracks_root / "sequence_a.txt").touch()

            with self.assertRaisesRegex(FileNotFoundError, "missing prediction files"):
                evaluate_mot(
                    dataset_root=root / "unused",
                    tracks_root=root / "tracks",
                    sequences=["sequence_a", "sequence_b"],
                    output_dir=root / "evaluation",
                )


if __name__ == "__main__":
    unittest.main()
