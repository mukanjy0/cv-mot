from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from run_next_experiments import NextExperimentRunner


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class RunNextExperimentsTest(unittest.TestCase):
    def test_generates_rollups_report_and_gmc_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            output_dir = root / "next"

            for method, mota, ids in (
                ("rtdetr_botsort_conf055_gmc_on", 0.30, 5),
                ("rtdetr_botsort_conf055_gmc_off", 0.25, 8),
                ("m2_yolo26_bytetrack_conf035", 0.20, 4),
                ("yolo26_upscale15_bytetrack", 0.22, 4),
            ):
                evaluation_dir = (
                    output_dir
                    / "stages"
                    / "gmc_ablation"
                    / method
                    / "attempts"
                    / "001"
                    / "benchmark"
                    / "evaluation"
                )
                evaluation_dir.mkdir(parents=True, exist_ok=True)
                (evaluation_dir.parent / "metadata.json").write_text(
                    '{"status": "completed"}\n', encoding="utf-8"
                )
                _write_csv(
                    evaluation_dir / "summary_by_method.csv",
                    [
                        {
                            "method": method,
                            "MOTA": mota,
                            "IDF1": 0.40,
                            "IDS": ids,
                            "FP": 10,
                            "FN": 20,
                            "FPS": 3.0,
                        }
                    ],
                )
                _write_csv(
                    evaluation_dir / "per_sequence_metrics.csv",
                    [
                        {
                            "method": method,
                            "sequence_name": "uav0000305_00000_v",
                            "MOTA": mota,
                            "IDF1": 0.40,
                            "IDS": ids,
                            "FP": 10,
                            "FN": 20,
                            "FPS": 3.0,
                        }
                    ],
                )
                _write_csv(
                    evaluation_dir / "mot_diagnostics_by_sequence.csv",
                    [
                        {
                            "method": method,
                            "sequence_name": "uav0000305_00000_v",
                            "MOTA": mota,
                            "IDF1": 0.40,
                            "FP": 10,
                            "FN": 20,
                            "IDS": ids,
                            "FPS": 3.0,
                            "unique_predicted_ids": 2,
                            "short_tracks_leq_3": 1,
                            "average_predicted_track_length": 4,
                            "median_predicted_bbox_area": 100,
                            "median_gt_bbox_area": 80,
                            "car_recall_proxy": 0.3 if "upscale" not in method else 0.4,
                            "prediction_path": str(root / "tracks.txt"),
                        }
                    ],
                )

            detection_dir = (
                output_dir
                / "stages"
                / "detection_only"
                / "attempts"
                / "001"
                / "detection"
            )
            detection_dir.mkdir(parents=True)
            (detection_dir / "detection_summary.json").write_text(
                '{"status": "completed"}\n', encoding="utf-8"
            )
            detection_rows = [
                {
                    "method": "m2_yolo26_bytetrack_conf035",
                    "sequence_name": "uav0000305_00000_v",
                    "class_id": 4,
                    "class_name": "car",
                    "precision": 0.5,
                    "recall": 0.3,
                    "AP50": 0.2,
                    "false_positives": 1,
                    "false_negatives": 2,
                    "gt_boxes": 10,
                    "predicted_boxes": 9,
                }
            ]
            _write_csv(detection_dir / "detection_summary_by_sequence.csv", detection_rows)
            _write_csv(detection_dir / "detection_summary_by_method.csv", detection_rows)
            _write_csv(detection_dir / "detection_summary_by_class.csv", detection_rows)

            runner = NextExperimentRunner(
                dataset_root=root,
                output_dir=output_dir,
                sequences=[],
                max_frames=None,
                device="cpu",
                force=False,
                dry_run=False,
                render_debug=False,
                debug_sequences=[],
                video_fps=30.0,
            )
            runner._generate_summaries()

            self.assertTrue((output_dir / "next_experiments_report.md").is_file())
            self.assertTrue(
                (output_dir / "summaries" / "gmc_ablation_summary.csv").is_file()
            )
            report = (output_dir / "next_experiments_report.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Best by MOTA", report)
            self.assertIn("GMC comparison", report)


if __name__ == "__main__":
    unittest.main()
