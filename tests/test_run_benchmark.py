from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiments.run_benchmark import run_benchmark


class RunBenchmarkTest(unittest.TestCase):
    def test_writes_expected_run_layout_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            dataset_root = root / "dataset"
            sequence = "sequence_a"
            sequence_dir = dataset_root / "sequences" / sequence
            annotation_dir = dataset_root / "annotations"
            sequence_dir.mkdir(parents=True)
            annotation_dir.mkdir(parents=True)
            (sequence_dir / "0000001.jpg").touch()
            (annotation_dir / f"{sequence}.txt").write_text(
                "1,1,10,20,30,40,1,1,0,0\n",
                encoding="utf-8",
            )

            config_path = root / "method.yaml"
            config_path.write_text(
                "name: method_a\n"
                "detector:\n"
                "  type: unused\n"
                "tracker:\n"
                "  type: unused\n",
                encoding="utf-8",
            )

            def fake_run_sequence(**kwargs: object) -> dict[str, object]:
                self.assertEqual(kwargs["device"], "mps")
                tracks_path = Path(str(kwargs["tracks_path"]))
                tracks_path.parent.mkdir(parents=True)
                tracks_path.write_text(
                    "1,7,10,20,30,40,0.9,1,-1\n",
                    encoding="utf-8",
                )
                return {
                    "frames_processed": 1,
                    "tracks_produced": 1,
                    "track_rows": 1,
                    "runtime_seconds": 0.25,
                    "fps": 4.0,
                }

            with patch(
                "src.experiments.run_benchmark.run_sequence",
                side_effect=fake_run_sequence,
            ):
                output_dir = run_benchmark(
                    dataset_root=dataset_root,
                    config_paths=[config_path],
                    sequences=[sequence],
                    max_frames=1,
                    output_root=root / "benchmarks",
                    run_id="test-run",
                    device="mps",
                    command=["python", "-m", "src.experiments.run_benchmark"],
                )

            run_dir = output_dir / "runs" / "method_a" / sequence
            self.assertTrue((run_dir / "tracks.txt").is_file())
            run_metadata = json.loads(
                (run_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_metadata["num_frames_processed"], 1)
            self.assertEqual(run_metadata["fps"], 4.0)
            self.assertEqual(run_metadata["resolved_config"]["name"], "method_a")
            self.assertEqual(
                run_metadata["resolved_config"]["detector"]["device"], "mps"
            )
            self.assertEqual(run_metadata["resolved_device"], "mps")

            benchmark_metadata = json.loads(
                (output_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(benchmark_metadata["status"], "completed")
            self.assertEqual(benchmark_metadata["selected_sequences"], [sequence])
            self.assertEqual(
                benchmark_metadata["device_environment"]["resolved_device"], "mps"
            )


if __name__ == "__main__":
    unittest.main()
