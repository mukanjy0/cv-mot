from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from src.experiments.run_overnight import OvernightRunner, REQUIRED_STAGE_IDS


class RunOvernightTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
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

        stages = []
        for stage_id in REQUIRED_STAGE_IDS:
            if stage_id == "final_comparison":
                stages.append(
                    {
                        "id": stage_id,
                        "kind": "comparison",
                        "fatal": True,
                        "configs": [str(config_path)],
                        "sequences": "all",
                        "max_frames": None,
                        "save_video": False,
                        "evaluate": True,
                        "sources": [],
                    }
                )
            else:
                stages.append(
                    {
                        "id": stage_id,
                        "kind": "benchmark",
                        "fatal": stage_id == "smoke_test",
                        "configs": [str(config_path)],
                        "sequences": [sequence],
                        "max_frames": 1,
                        "save_video": False,
                        "evaluate": True,
                    }
                )
        queue_path = root / "queue.yaml"
        queue_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "all_sequences": [sequence],
                    "stages": stages,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return dataset_root, config_path, queue_path

    @staticmethod
    def _write_evaluation(benchmark_dir: Path) -> None:
        evaluation_dir = benchmark_dir / "evaluation"
        evaluation_dir.mkdir(parents=True)
        (benchmark_dir / "metadata.json").write_text(
            '{"status": "completed"}\n', encoding="utf-8"
        )
        fields = ["method", "MOTA", "IDF1", "IDS", "FP", "FN", "FPS"]
        with (evaluation_dir / "summary_by_method.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "method": "method_a",
                    "MOTA": 0.5,
                    "IDF1": 0.6,
                    "IDS": 1,
                    "FP": 2,
                    "FN": 3,
                    "FPS": 4,
                }
            )
        with (evaluation_dir / "per_sequence_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=[*fields, "sequence_name"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "method": "method_a",
                    "sequence_name": "sequence_a",
                    "MOTA": 0.5,
                    "IDF1": 0.6,
                    "IDS": 1,
                    "FP": 2,
                    "FN": 3,
                    "FPS": 4,
                }
            )

    @classmethod
    def _fake_benchmark(cls, command: list[str], **_: object) -> SimpleNamespace:
        output_root = Path(command[command.index("--output-root") + 1])
        cls._write_evaluation(output_root / "benchmark")
        return SimpleNamespace(returncode=0)

    def test_resume_skips_completed_and_force_adds_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            dataset_root, _config_path, queue_path = self._fixture(root)
            output_dir = root / "overnight"

            def make_runner(*, resume: bool = False, force: bool = False):
                return OvernightRunner(
                    queue_path=queue_path,
                    dataset_root=dataset_root,
                    output_dir=output_dir,
                    resume=resume,
                    force=force,
                    dry_run=False,
                    summarize_only=False,
                    device="mps",
                    only_stage="smoke_test",
                )

            with patch(
                "src.experiments.run_overnight._git_commit", return_value=None
            ):
                with patch(
                    "src.experiments.run_overnight.platform.platform",
                    return_value="test-platform",
                ):
                    with patch(
                        "src.experiments.run_overnight.subprocess.run",
                        side_effect=self._fake_benchmark,
                    ) as first_run:
                        make_runner().run()
                        self.assertEqual(first_run.call_count, 1)
                        command = first_run.call_args.args[0]
                        self.assertEqual(
                            command[command.index("--device") + 1], "mps"
                        )

                    with patch(
                        "src.experiments.run_overnight.subprocess.run"
                    ) as resumed_run:
                        make_runner(resume=True).run()
                        resumed_run.assert_not_called()

                    with patch(
                        "src.experiments.run_overnight.subprocess.run",
                        side_effect=self._fake_benchmark,
                    ) as forced_run:
                        make_runner(force=True).run()
                        self.assertEqual(forced_run.call_count, 1)

            attempts_dir = (
                output_dir
                / "stages"
                / "smoke_test"
                / "runs"
                / "method_a"
                / "attempts"
            )
            self.assertTrue((attempts_dir / "001").is_dir())
            self.assertTrue((attempts_dir / "002").is_dir())
            metadata = json.loads(
                (output_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["device_environment"]["resolved_device"], "mps"
            )

    def test_final_comparison_reuses_complete_full_validation_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            dataset_root, config_path, queue_path = self._fixture(root)
            queue = yaml.safe_load(queue_path.read_text(encoding="utf-8"))
            queue["stages"][-1]["sources"] = [
                {
                    "label": label,
                    "stage": "full_baseline_m1_m2_m3",
                    "config_name": "method_a",
                }
                for label in ("m1_baseline", "m2_best", "m3_best")
            ]
            queue_path.write_text(
                yaml.safe_dump(queue, sort_keys=False), encoding="utf-8"
            )

            output_dir = root / "overnight"
            runner = OvernightRunner(
                queue_path=queue_path,
                dataset_root=dataset_root,
                output_dir=output_dir,
                resume=False,
                force=False,
                dry_run=False,
                summarize_only=False,
                device=None,
                only_stage=None,
            )
            with patch(
                "src.experiments.run_overnight._git_commit", return_value=None
            ):
                with patch(
                    "src.experiments.run_overnight.platform.platform",
                    return_value="test-platform",
                ):
                    runner._initialize_output()

            benchmark_dir = root / "baseline-benchmark"
            self._write_evaluation(benchmark_dir)
            experiment_dir = (
                output_dir
                / "stages"
                / "full_baseline_m1_m2_m3"
                / "runs"
                / "method_a"
            )
            experiment_dir.mkdir(parents=True)
            (experiment_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "stage": "full_baseline_m1_m2_m3",
                        "phase": "runs",
                        "experiment_id": "method_a",
                        "source_config_path": str(config_path),
                        "resolved_config": {"name": "method_a"},
                        "sequences": ["sequence_a"],
                        "max_frames": None,
                        "latest_benchmark_dir": str(benchmark_dir),
                    }
                ),
                encoding="utf-8",
            )

            final_stage = runner.stages[-1]
            self.assertTrue(runner._build_final_comparison(final_stage))
            with (
                output_dir / "summaries" / "final_summary_by_method.csv"
            ).open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["comparison_label"] for row in rows],
                ["m1_baseline", "m2_best", "m3_best"],
            )


if __name__ == "__main__":
    unittest.main()
