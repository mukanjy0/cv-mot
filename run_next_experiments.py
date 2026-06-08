from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.core.config import load_config
from src.core.device import device_diagnostics, format_device_diagnostics
from src.data.visdrone import list_sequence_names
from src.visualization.render_debug_video import render_debug_video


REPRESENTATIVE_SEQUENCES = [
    "uav0000137_00458_v",
    "uav0000268_05773_v",
    "uav0000117_02622_v",
    "uav0000305_00000_v",
    "uav0000086_00000_v",
]

DETECTION_CONFIGS = [
    "configs/overnight/m2_yolo26_bytetrack_conf035.yaml",
    "configs/overnight/m3_rtdetr_botsort_conf055.yaml",
]

M3_COMPACT_CONFIGS = [
    "configs/next/m3_rtdetr_botsort_conf050.yaml",
    "configs/overnight/m3_rtdetr_botsort_conf055.yaml",
    "configs/next/m3_rtdetr_botsort_conf060.yaml",
    "configs/next/m3_rtdetr_botsort_conf065.yaml",
]

UPSCALING_CONFIGS = [
    "configs/overnight/m2_yolo26_bytetrack_conf035.yaml",
    "configs/next/yolo26_upscale15_bytetrack.yaml",
    "configs/next/yolo26_upscale15_botsort.yaml",
    "configs/next/yolo26_upscale20_bytetrack.yaml",
    "configs/next/yolo26_upscale20_botsort.yaml",
    "configs/next/rtdetr_upscale15_botsort.yaml",
]

SAHI_STRICT_CONFIGS = [
    "configs/next/sahi_yolo26_botsort_slice640_overlap020_conf040.yaml",
    "configs/next/sahi_yolo26_botsort_slice640_overlap020_conf050.yaml",
    "configs/next/sahi_yolo26_botsort_slice768_overlap015_conf040.yaml",
    "configs/next/sahi_yolo26_botsort_slice768_overlap015_conf050.yaml",
]

GMC_ABLATION_CONFIGS = [
    "configs/next/rtdetr_botsort_conf055_gmc_on.yaml",
    "configs/next/rtdetr_botsort_conf055_gmc_off.yaml",
    "configs/next/yolo26_botsort_gmc_on.yaml",
    "configs/next/yolo26_botsort_gmc_off.yaml",
]

BENCHMARK_STAGES = [
    ("m3_compact_tuning", M3_COMPACT_CONFIGS),
    ("upscaling_experiments", UPSCALING_CONFIGS),
    ("strict_sahi_botsort", SAHI_STRICT_CONFIGS),
    ("gmc_ablation", GMC_ABLATION_CONFIGS),
]

GMC_PAIRS = [
    ("rtdetr_botsort_conf055", "rtdetr_botsort_conf055_gmc_on", "rtdetr_botsort_conf055_gmc_off"),
    ("yolo26_botsort", "yolo26_botsort_gmc_on", "yolo26_botsort_gmc_off"),
]


@dataclass(frozen=True)
class BenchmarkRecord:
    stage: str
    experiment_id: str
    benchmark_dir: Path
    attempt: str


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(data), indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(row: Mapping[str, Any], key: str, default: float | None = None) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _preferred_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


class NextExperimentRunner:
    def __init__(
        self,
        *,
        dataset_root: Path,
        output_dir: Path,
        sequences: Sequence[str],
        max_frames: int | None,
        device: str,
        force: bool,
        dry_run: bool,
        render_debug: bool,
        debug_sequences: Sequence[str],
        video_fps: float,
    ) -> None:
        self.project_root = Path.cwd().resolve()
        self.dataset_root = dataset_root.resolve()
        self.output_dir = output_dir.resolve()
        self.sequences = list(sequences)
        self.max_frames = max_frames
        self.device = device
        self.force = force
        self.dry_run = dry_run
        self.render_debug = render_debug
        self.debug_sequences = list(debug_sequences)
        self.video_fps = video_fps
        self.logs_dir = self.output_dir / "logs"
        self.stages_dir = self.output_dir / "stages"
        self.summaries_dir = self.output_dir / "summaries"

    def run(self) -> Path:
        self._validate()
        self._initialize()
        self._run_detection_stage()
        for stage_id, config_paths in BENCHMARK_STAGES:
            for config_path in config_paths:
                self._run_benchmark(stage_id, Path(config_path))
        if not self.dry_run:
            self._generate_summaries()
            if self.render_debug:
                self._render_debug_outputs()
        return self.output_dir

    def _validate(self) -> None:
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")
        available = set(list_sequence_names(self.dataset_root))
        missing = [sequence for sequence in self.sequences if sequence not in available]
        if missing:
            raise ValueError(f"Unknown sequence names: {missing}")
        if len(set(self.sequences)) != len(self.sequences):
            raise ValueError("Sequence names must be unique")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")
        for config_path in self._all_config_paths():
            load_config(config_path)

    def _initialize(self) -> None:
        if self.dry_run:
            print(f"Output directory: {self.output_dir}")
            print(f"Sequences: {', '.join(self.sequences)}")
            print(f"Device: {self.device}")
            return
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.stages_dir.mkdir(parents=True, exist_ok=True)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "commands.txt").touch(exist_ok=True)
        metadata = {
            "timestamp": _timestamp(),
            "status": "running",
            "dataset_root": str(self.dataset_root),
            "output_directory": str(self.output_dir),
            "sequences": self.sequences,
            "max_frames": self.max_frames,
            "device": self.device,
            "force": self.force,
            "python": sys.version,
            "platform": platform.platform(),
            "command": shlex.join([sys.executable, *sys.argv]),
            "device_environment": device_diagnostics(self.device),
        }
        _write_json(self.output_dir / "metadata.json", metadata)

    def _all_config_paths(self) -> list[Path]:
        paths = [Path(path) for path in DETECTION_CONFIGS]
        for _stage_id, config_paths in BENCHMARK_STAGES:
            paths.extend(Path(path) for path in config_paths)
        return paths

    def _run_detection_stage(self) -> None:
        stage_dir = self.stages_dir / "detection_only"
        latest = self._latest_completed_detection(stage_dir)
        if latest is not None and not self.force:
            print(f"SKIP detection_only: {latest}")
            return

        attempt_dir, attempt_name = self._next_attempt_dir(stage_dir)
        detection_dir = attempt_dir / "detection"
        command = [
            sys.executable,
            "-m",
            "src.evaluation.evaluate_detection",
            "--dataset-root",
            str(self.dataset_root),
            "--configs",
            *DETECTION_CONFIGS,
            "--sequences",
            *self.sequences,
            "--output-dir",
            str(detection_dir),
            "--device",
            self.device,
        ]
        if self.max_frames is not None:
            command.extend(["--max-frames", str(self.max_frames)])
        self._run_command(
            stage="detection_only",
            experiment_id="detector_configs",
            attempt_name=attempt_name,
            attempt_dir=attempt_dir,
            command=command,
            required_output=detection_dir / "detection_summary_by_method.csv",
        )

    def _run_benchmark(self, stage_id: str, config_path: Path) -> None:
        config = load_config(config_path)
        experiment_id = str(config["name"])
        experiment_dir = self.stages_dir / stage_id / experiment_id
        latest = self._latest_completed_benchmark(experiment_dir)
        if latest is not None and not self.force:
            print(f"SKIP {stage_id}/{experiment_id}: {latest}")
            return

        attempt_dir, attempt_name = self._next_attempt_dir(experiment_dir)
        command = [
            sys.executable,
            "-m",
            "src.experiments.run_benchmark",
            "--dataset-root",
            str(self.dataset_root),
            "--configs",
            str(config_path),
            "--sequences",
            *self.sequences,
            "--output-root",
            str(attempt_dir),
            "--run-id",
            "benchmark",
            "--evaluate",
            "--device",
            self.device,
        ]
        if self.max_frames is not None:
            command.extend(["--max-frames", str(self.max_frames)])
        self._run_command(
            stage=stage_id,
            experiment_id=experiment_id,
            attempt_name=attempt_name,
            attempt_dir=attempt_dir,
            command=command,
            required_output=attempt_dir / "benchmark" / "evaluation" / "summary_by_method.csv",
        )

    def _run_command(
        self,
        *,
        stage: str,
        experiment_id: str,
        attempt_name: str,
        attempt_dir: Path,
        command: list[str],
        required_output: Path,
    ) -> None:
        command_text = shlex.join(command)
        if self.dry_run:
            print(command_text)
            return

        attempt_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = self.logs_dir / f"{stage}__{experiment_id}__{attempt_name}.stdout.log"
        stderr_path = self.logs_dir / f"{stage}__{experiment_id}__{attempt_name}.stderr.log"
        metadata_path = attempt_dir / "run_next_metadata.json"
        _write_json(
            metadata_path,
            {
                "timestamp": _timestamp(),
                "status": "running",
                "stage": stage,
                "experiment_id": experiment_id,
                "attempt": attempt_name,
                "command": command_text,
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            },
        )
        with (self.output_dir / "commands.txt").open("a", encoding="utf-8") as handle:
            handle.write(command_text + "\n")

        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        matplotlib_dir = self.output_dir / ".cache" / "matplotlib"
        matplotlib_dir.mkdir(parents=True, exist_ok=True)
        environment["MPLCONFIGDIR"] = str(matplotlib_dir)

        print(f"RUN {stage}/{experiment_id} ({attempt_name})")
        started = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout_handle:
            with stderr_path.open("w", encoding="utf-8") as stderr_handle:
                completed = subprocess.run(
                    command,
                    cwd=self.project_root,
                    env=environment,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    check=False,
                )
        elapsed = time.perf_counter() - started
        status = (
            "completed"
            if completed.returncode == 0 and required_output.is_file()
            else "failed"
        )
        _write_json(
            metadata_path,
            {
                "timestamp": _timestamp(),
                "status": status,
                "stage": stage,
                "experiment_id": experiment_id,
                "attempt": attempt_name,
                "command": command_text,
                "runtime_seconds": elapsed,
                "returncode": completed.returncode,
                "required_output": str(required_output),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            },
        )
        if status != "completed":
            raise RuntimeError(
                f"{stage}/{experiment_id} failed with return code "
                f"{completed.returncode}. See {stderr_path}"
            )
        print(f"DONE {stage}/{experiment_id}")

    def _next_attempt_dir(self, stage_dir: Path) -> tuple[Path, str]:
        attempts_dir = stage_dir / "attempts"
        existing = sorted(path.name for path in attempts_dir.glob("*") if path.is_dir())
        next_number = len(existing) + 1
        attempt_name = f"{next_number:03d}"
        return attempts_dir / attempt_name, attempt_name

    def _latest_completed_detection(self, stage_dir: Path) -> Path | None:
        latest: Path | None = None
        for summary_path in sorted(
            (stage_dir / "attempts").glob("*/detection/detection_summary.json")
        ):
            latest = summary_path.parent
        return latest

    def _latest_completed_benchmark(self, experiment_dir: Path) -> Path | None:
        latest: Path | None = None
        for metadata_path in sorted(
            (experiment_dir / "attempts").glob("*/benchmark/metadata.json")
        ):
            try:
                metadata = _read_json(metadata_path)
            except (OSError, ValueError):
                continue
            if metadata.get("status") == "completed":
                latest = metadata_path.parent
        return latest

    def _collect_latest_benchmarks(self) -> list[BenchmarkRecord]:
        records: list[BenchmarkRecord] = []
        for stage_dir in sorted(path for path in self.stages_dir.iterdir() if path.is_dir()):
            if stage_dir.name == "detection_only":
                continue
            for experiment_dir in sorted(path for path in stage_dir.iterdir() if path.is_dir()):
                latest = self._latest_completed_benchmark(experiment_dir)
                if latest is None:
                    continue
                records.append(
                    BenchmarkRecord(
                        stage=stage_dir.name,
                        experiment_id=experiment_dir.name,
                        benchmark_dir=latest,
                        attempt=latest.parent.name,
                    )
                )
        return records

    def _latest_detection_dir(self) -> Path | None:
        return self._latest_completed_detection(self.stages_dir / "detection_only")

    def _generate_summaries(self) -> None:
        method_rows: list[dict[str, Any]] = []
        sequence_rows: list[dict[str, Any]] = []
        diagnostic_rows: list[dict[str, Any]] = []

        for record in self._collect_latest_benchmarks():
            evaluation_dir = record.benchmark_dir / "evaluation"
            for row in _read_csv(evaluation_dir / "summary_by_method.csv"):
                method_rows.append(
                    {
                        "stage": record.stage,
                        "experiment_id": record.experiment_id,
                        "attempt": record.attempt,
                        **row,
                    }
                )
            for row in _read_csv(evaluation_dir / "per_sequence_metrics.csv"):
                sequence_rows.append(
                    {
                        "stage": record.stage,
                        "experiment_id": record.experiment_id,
                        "attempt": record.attempt,
                        **row,
                    }
                )
            for row in _read_csv(evaluation_dir / "mot_diagnostics_by_sequence.csv"):
                diagnostic_rows.append(
                    {
                        "stage": record.stage,
                        "experiment_id": record.experiment_id,
                        "attempt": record.attempt,
                        **row,
                    }
                )

        _write_csv(self.summaries_dir / "final_summary_by_method.csv", method_rows)
        _write_csv(self.summaries_dir / "final_summary_by_sequence.csv", sequence_rows)
        _write_csv(
            self.summaries_dir / "mot_diagnostics_by_sequence.csv",
            diagnostic_rows,
        )
        self._copy_detection_summaries()
        gmc_rows = self._write_gmc_summary(method_rows)
        self._write_report(method_rows, sequence_rows, diagnostic_rows, gmc_rows)
        self._mark_completed()

    def _copy_detection_summaries(self) -> None:
        detection_dir = self._latest_detection_dir()
        if detection_dir is None:
            return
        for filename in (
            "detection_summary_by_method.csv",
            "detection_summary_by_sequence.csv",
            "detection_summary_by_class.csv",
        ):
            rows = _read_csv(detection_dir / filename)
            _write_csv(self.summaries_dir / filename, rows)

    def _write_gmc_summary(
        self, method_rows: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        by_method = {str(row.get("method")): row for row in method_rows}
        rows: list[dict[str, Any]] = []
        for family, on_method, off_method in GMC_PAIRS:
            on_row = by_method.get(on_method)
            off_row = by_method.get(off_method)
            if on_row is None or off_row is None:
                continue
            rows.append(
                {
                    "family": family,
                    "gmc_on_method": on_method,
                    "gmc_off_method": off_method,
                    "gmc_on_MOTA": on_row.get("MOTA"),
                    "gmc_off_MOTA": off_row.get("MOTA"),
                    "MOTA_delta_on_minus_off": (
                        _float(on_row, "MOTA", 0.0) - _float(off_row, "MOTA", 0.0)
                    ),
                    "gmc_on_IDF1": on_row.get("IDF1"),
                    "gmc_off_IDF1": off_row.get("IDF1"),
                    "IDF1_delta_on_minus_off": (
                        _float(on_row, "IDF1", 0.0) - _float(off_row, "IDF1", 0.0)
                    ),
                    "gmc_on_IDS": on_row.get("IDS"),
                    "gmc_off_IDS": off_row.get("IDS"),
                    "IDS_delta_off_minus_on": (
                        _float(off_row, "IDS", 0.0) - _float(on_row, "IDS", 0.0)
                    ),
                    "gmc_on_FP": on_row.get("FP"),
                    "gmc_off_FP": off_row.get("FP"),
                    "gmc_on_FN": on_row.get("FN"),
                    "gmc_off_FN": off_row.get("FN"),
                    "gmc_on_FPS": on_row.get("FPS"),
                    "gmc_off_FPS": off_row.get("FPS"),
                }
            )
        _write_csv(self.summaries_dir / "gmc_ablation_summary.csv", rows)
        return rows

    def _best(
        self,
        rows: Sequence[Mapping[str, Any]],
        metric: str,
    ) -> Mapping[str, Any] | None:
        valid = [row for row in rows if _float(row, metric) is not None]
        if not valid:
            return None
        return max(valid, key=lambda row: _float(row, metric, float("-inf")) or float("-inf"))

    def _mean_by_method(
        self,
        rows: Sequence[Mapping[str, Any]],
        metric: str,
    ) -> dict[str, float]:
        values: dict[str, list[float]] = {}
        for row in rows:
            value = _float(row, metric)
            method = str(row.get("method", ""))
            if value is None or not method:
                continue
            values.setdefault(method, []).append(value)
        return {
            method: sum(method_values) / len(method_values)
            for method, method_values in values.items()
            if method_values
        }

    def _write_report(
        self,
        method_rows: Sequence[Mapping[str, Any]],
        sequence_rows: Sequence[Mapping[str, Any]],
        diagnostic_rows: Sequence[Mapping[str, Any]],
        gmc_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        best_mota = self._best(method_rows, "MOTA")
        best_idf1 = self._best(method_rows, "IDF1")
        best_fps = self._best(method_rows, "FPS")
        target_rows = [
            row
            for row in sequence_rows
            if row.get("sequence_name") == "uav0000305_00000_v"
        ]
        best_target = self._best(target_rows, "MOTA")
        detection_rows = _read_csv(
            self.summaries_dir / "detection_summary_by_sequence.csv"
        )
        target_detection = [
            row
            for row in detection_rows
            if row.get("sequence_name") == "uav0000305_00000_v"
            and row.get("class_id") in {"1", "4", "all"}
        ]

        car_recall = self._mean_by_method(diagnostic_rows, "car_recall_proxy")
        upscaled_methods = {
            method: value
            for method, value in car_recall.items()
            if "upscale" in method
        }
        baseline_recall = car_recall.get("m2_yolo26_bytetrack_conf035")
        sahi_methods = {
            method: value for method, value in car_recall.items() if method.startswith("sahi_")
        }

        lines = [
            "# Next Experiments Report",
            "",
            f"Updated: {_timestamp()}",
            "",
            "## Experiment List",
            "",
            "- Detection-only diagnostics: " + ", ".join(DETECTION_CONFIGS),
            "- M3 confidence sweep: 0.50, 0.55, 0.60, 0.65.",
            "- Upscaling: YOLO26 1.5x/2.0 with ByteTrack and BoT-SORT, plus RT-DETR 1.5x BoT-SORT.",
            "- Strict SAHI + BoT-SORT: slice 640/768, overlap 0.20/0.15, conf 0.40/0.50.",
            "- GMC ablation: RT-DETR and YOLO26 BoT-SORT with `cmc_method: sof` versus disabled.",
            "",
            "## Best Methods",
            "",
            self._format_best("Best by MOTA", best_mota, "MOTA"),
            self._format_best("Best by IDF1", best_idf1, "IDF1"),
            self._format_best("Best by FPS", best_fps, "FPS"),
            self._format_best("Best on uav0000305_00000_v", best_target, "MOTA"),
            "",
            "## uav0000305_00000_v Detection Diagnosis",
            "",
        ]
        if target_detection:
            for row in target_detection:
                lines.append(
                    f"- `{row.get('method')}` class `{row.get('class_name')}`: "
                    f"precision={row.get('precision')}, recall={row.get('recall')}, "
                    f"AP50={row.get('AP50')}, FP={row.get('false_positives')}, "
                    f"FN={row.get('false_negatives')}, GT={row.get('gt_boxes')}, "
                    f"pred={row.get('predicted_boxes')}."
                )
        else:
            lines.append("- Detection-only results are not available yet.")

        lines.extend(["", "## Interpretation", ""])
        lines.append(self._upscaling_sentence(baseline_recall, upscaled_methods))
        lines.append(self._sahi_sentence(method_rows, sahi_methods, baseline_recall))
        lines.append(self._gmc_sentence(gmc_rows))
        lines.extend(
            [
                "",
                "Detector-limited failures are indicated by high FN and low detection recall.",
                "Association-limited failures are indicated by high IDS, many unique IDs, and many tracks of three frames or fewer.",
                "Precision failures are indicated by high FP, especially when SAHI increases predicted boxes without improving recall.",
                "Speed tradeoffs are indicated by low FPS; compare FPS before choosing an upscaled or sliced method.",
                "",
                "## Recommendation",
                "",
                self._recommendation_sentence(best_mota),
                "",
                "Presentation narrative: start with uav0000305_00000_v as the visible failure case, separate missed cars from ID fragmentation, then show the upscaling, SAHI, and GMC ablations as targeted tests rather than blind hyperparameter tuning.",
            ]
        )
        report = "\n".join(lines) + "\n"
        (self.output_dir / "next_experiments_report.md").write_text(
            report, encoding="utf-8"
        )
        (self.summaries_dir / "next_experiments_report.md").write_text(
            report, encoding="utf-8"
        )

    def _format_best(
        self,
        label: str,
        row: Mapping[str, Any] | None,
        metric: str,
    ) -> str:
        if row is None:
            return f"- {label}: unavailable."
        return (
            f"- {label}: `{row.get('method')}` with {metric}={row.get(metric)}, "
            f"MOTA={row.get('MOTA')}, IDF1={row.get('IDF1')}, "
            f"FP={row.get('FP')}, FN={row.get('FN')}, IDS={row.get('IDS')}, "
            f"FPS={row.get('FPS')}."
        )

    def _upscaling_sentence(
        self,
        baseline_recall: float | None,
        upscaled_methods: Mapping[str, float],
    ) -> str:
        if baseline_recall is None or not upscaled_methods:
            return "- Upscaling car-recall comparison: unavailable until baseline and upscaled MOT diagnostics complete."
        best_method, best_value = max(upscaled_methods.items(), key=lambda item: item[1])
        delta = best_value - baseline_recall
        return (
            f"- Upscaling car-recall comparison: best upscaled method `{best_method}` "
            f"has proxy={best_value:.4f} versus baseline={baseline_recall:.4f} "
            f"(delta={delta:+.4f})."
        )

    def _sahi_sentence(
        self,
        method_rows: Sequence[Mapping[str, Any]],
        sahi_methods: Mapping[str, float],
        baseline_recall: float | None,
    ) -> str:
        if not sahi_methods:
            return "- SAHI comparison: unavailable until strict SAHI experiments complete."
        fp_by_method = {
            str(row.get("method")): _float(row, "FP")
            for row in method_rows
            if str(row.get("method")).startswith("sahi_")
        }
        best_method, best_recall = max(sahi_methods.items(), key=lambda item: item[1])
        fp = fp_by_method.get(best_method)
        baseline_text = (
            f" versus baseline={baseline_recall:.4f}"
            if baseline_recall is not None
            else ""
        )
        return (
            f"- SAHI comparison: best strict SAHI car proxy is `{best_method}` "
            f"at {best_recall:.4f}{baseline_text}; aggregate FP={fp}."
        )

    def _gmc_sentence(self, gmc_rows: Sequence[Mapping[str, Any]]) -> str:
        if not gmc_rows:
            return "- GMC comparison: unavailable until on/off ablation completes."
        parts = []
        for row in gmc_rows:
            parts.append(
                f"{row.get('family')}: IDS off-on delta={row.get('IDS_delta_off_minus_on')}, "
                f"MOTA on-off delta={row.get('MOTA_delta_on_minus_off')}"
            )
        return "- GMC comparison: " + "; ".join(parts) + "."

    def _recommendation_sentence(self, best_mota: Mapping[str, Any] | None) -> str:
        if best_mota is None:
            return "Recommended final method is unavailable until at least one full benchmark completes."
        return (
            f"Recommended final method: `{best_mota.get('method')}` for the current "
            "selection rule of highest full-validation MOTA, with IDF1/FPS checked "
            "for presentation tradeoffs."
        )

    def _render_debug_outputs(self) -> None:
        diagnostics = _read_csv(self.summaries_dir / "mot_diagnostics_by_sequence.csv")
        method_rows = _read_csv(self.summaries_dir / "final_summary_by_method.csv")
        best = self._best(method_rows, "MOTA")
        if best is None:
            return
        method = str(best.get("method"))
        for sequence_name in self.debug_sequences:
            candidates = [
                row
                for row in diagnostics
                if row.get("method") == method and row.get("sequence_name") == sequence_name
            ]
            if not candidates:
                continue
            tracks_path = candidates[0].get("prediction_path")
            if not tracks_path:
                continue
            output_path = (
                self.output_dir
                / "debug_videos"
                / method
                / f"{sequence_name}_debug.mp4"
            )
            render_debug_video(
                dataset_root=self.dataset_root,
                sequence_name=sequence_name,
                tracks_path=tracks_path,
                output_path=output_path,
                method_name=method,
                max_frames=self.max_frames,
                fps=self.video_fps,
                overwrite=self.force,
            )

    def _mark_completed(self) -> None:
        metadata_path = self.output_dir / "metadata.json"
        metadata = _read_json(metadata_path) if metadata_path.is_file() else {}
        metadata.update(
            {
                "status": "completed",
                "completed_at": _timestamp(),
                "summaries_dir": str(self.summaries_dir),
                "report": str(self.output_dir / "next_experiments_report.md"),
            }
        )
        _write_json(metadata_path, metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the next-step VisDrone MOT experiment and diagnostic suite."
    )
    parser.add_argument(
        "--dataset",
        "--dataset-root",
        dest="dataset_root",
        required=True,
        help="Path to VisDrone2019-MOT-val.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/next_experiments/default",
        help="Resumable suite output directory.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda:0", "mps"],
        default=None,
        help="Inference device. Defaults to MPS if available, then CUDA, then CPU.",
    )
    parser.add_argument("--force", action="store_true", help="Create new attempts.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--representative",
        action="store_true",
        help="Use a multi-sequence representative subset instead of full validation.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Explicit sequence list. Overrides --representative.",
    )
    parser.add_argument("--render-debug", action="store_true")
    parser.add_argument(
        "--debug-sequences",
        nargs="+",
        default=["uav0000305_00000_v"],
    )
    parser.add_argument("--video-fps", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    if args.sequences:
        sequences = args.sequences
    elif args.representative:
        sequences = REPRESENTATIVE_SEQUENCES
    else:
        sequences = list_sequence_names(dataset_root)
    device = args.device or _preferred_device()
    diagnostics = device_diagnostics(device)
    print(format_device_diagnostics(diagnostics))
    runner = NextExperimentRunner(
        dataset_root=dataset_root,
        output_dir=Path(args.output_dir),
        sequences=sequences,
        max_frames=args.max_frames,
        device=device,
        force=bool(args.force),
        dry_run=bool(args.dry_run),
        render_debug=bool(args.render_debug),
        debug_sequences=args.debug_sequences,
        video_fps=float(args.video_fps),
    )
    output_dir = runner.run()
    if args.dry_run:
        print("\nDry run only; no experiments were executed.")
    else:
        print(f"\nNext experiment outputs: {output_dir}")
        print(f"Report: {output_dir / 'next_experiments_report.md'}")


if __name__ == "__main__":
    main()
