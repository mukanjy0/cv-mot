from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.core.config import load_config
from src.core.device import (
    apply_device_to_detector_configs,
    device_diagnostics,
    format_device_diagnostics,
    resolve_device,
)
from src.data.visdrone import VisDroneSequence, list_sequence_names


REQUIRED_STAGE_IDS = [
    "smoke_test",
    "full_baseline_m1_m2_m3",
    "tune_m3",
    "tune_m2",
    "sahi_m4_best_effort",
    "final_comparison",
]

SUMMARY_FILENAMES = {
    "full_baseline_m1_m2_m3": "full_baseline_summary_by_method.csv",
    "tune_m3": "tuning_m3_summary.csv",
    "tune_m2": "tuning_m2_summary.csv",
    "sahi_m4_best_effort": "sahi_summary.csv",
}


@dataclass(frozen=True)
class Candidate:
    experiment_id: str
    source_path: Path
    config: dict[str, Any]


class FatalStageError(RuntimeError):
    pass


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "pyyaml is required for overnight experiment queues. Install "
            "dependencies with: python -m pip install -r requirements.txt"
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in queue: {path}")
    return data


def _write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(data), indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    default_fields: Sequence[str] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(default_fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _float(row: Mapping[str, Any], key: str, default: float) -> float:
    try:
        value = row.get(key)
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _rank_rows(rows: Sequence[Mapping[str, Any]], rule: str) -> list[dict[str, Any]]:
    def key(row: Mapping[str, Any]) -> tuple[float, ...]:
        mota = _float(row, "MOTA", float("-inf"))
        idf1 = _float(row, "IDF1", float("-inf"))
        fps = _float(row, "FPS", float("-inf"))
        fp = _float(row, "FP", float("inf"))
        fn = _float(row, "FN", float("inf"))
        if rule == "m3":
            return (-mota, -idf1, fp, -fps)
        if rule == "m2":
            return (-mota, -idf1, -fps, fp)
        if rule == "m4":
            return (-mota, -idf1, fn, fp, -fps)
        raise ValueError(f"Unsupported selection rule: {rule}")

    return [dict(row) for row in sorted(rows, key=key)]


class OvernightRunner:
    def __init__(
        self,
        *,
        queue_path: Path,
        dataset_root: Path,
        output_dir: Path,
        resume: bool,
        force: bool,
        dry_run: bool,
        summarize_only: bool,
        device: str,
        only_stage: str | None,
    ) -> None:
        self.project_root = Path.cwd().resolve()
        self.queue_path = queue_path.resolve()
        self.dataset_root = dataset_root.resolve()
        self.output_dir = output_dir.resolve()
        self.resume = resume
        self.force = force
        self.dry_run = dry_run
        self.summarize_only = summarize_only
        self.requested_device = device
        self.device = resolve_device(device)
        self.device_environment = device_diagnostics(device)
        self.only_stage = only_stage
        self.queue = _read_yaml(self.queue_path)
        self.stages = self._validate_queue()
        self.selections: dict[str, dict[str, Any]] = {}
        self.metadata: dict[str, Any] = {}

    def _validate_queue(self) -> list[dict[str, Any]]:
        if int(self.queue.get("version", 0)) != 1:
            raise ValueError("experiment_queue.yaml must set version: 1")
        all_sequences = self.queue.get("all_sequences")
        if not isinstance(all_sequences, list) or not all_sequences:
            raise ValueError("Queue must define a non-empty all_sequences list")
        if len(set(all_sequences)) != len(all_sequences):
            raise ValueError("Queue all_sequences must not contain duplicates")
        available_sequences = list_sequence_names(self.dataset_root)
        if sorted(str(name) for name in all_sequences) != available_sequences:
            raise ValueError(
                "Queue all_sequences must exactly match dataset sequences. "
                f"Queue={sorted(all_sequences)}, dataset={available_sequences}"
            )

        stages = self.queue.get("stages")
        if not isinstance(stages, list):
            raise ValueError("Queue stages must be a list")
        stage_ids = [stage.get("id") for stage in stages]
        if stage_ids != REQUIRED_STAGE_IDS:
            raise ValueError(
                "Queue stages must appear in this exact order: "
                + ", ".join(REQUIRED_STAGE_IDS)
            )
        if self.only_stage and self.only_stage not in stage_ids:
            raise ValueError(f"Unknown --only-stage value: {self.only_stage}")

        for sequence in all_sequences:
            VisDroneSequence(self.dataset_root, str(sequence))
            annotation = self.dataset_root / "annotations" / f"{sequence}.txt"
            if not annotation.is_file():
                raise FileNotFoundError(f"Missing annotation file: {annotation}")

        for stage in stages:
            sequences = self._resolve_sequences(stage.get("sequences", []))
            if stage["id"] != "final_comparison" and not sequences:
                raise ValueError(f"Stage {stage['id']} has no sequences")
            max_frames = stage.get("max_frames")
            if max_frames is not None and int(max_frames) <= 0:
                raise ValueError(f"Stage {stage['id']} max_frames must be positive")
            for item in stage.get("configs", []):
                path = self._config_item_path(item)
                load_config(path)
        return stages

    def _resolve_sequences(self, value: Any) -> list[str]:
        if value == "all":
            return [str(name) for name in self.queue["all_sequences"]]
        if not isinstance(value, list):
            raise ValueError(f"Sequences must be a list or 'all', got: {value!r}")
        return [str(name) for name in value]

    def _config_item_path(self, item: Any) -> Path:
        raw_path = item if isinstance(item, str) else item.get("path")
        if not raw_path:
            raise ValueError(f"Invalid queue config entry: {item!r}")
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    def _initialize_output(self) -> None:
        exists = self.output_dir.exists()
        if exists and not (self.resume or self.force or self.summarize_only):
            raise FileExistsError(
                f"Output directory already exists: {self.output_dir}. "
                "Use --resume, --force, or --summarize-only."
            )
        if not exists and (self.resume or self.summarize_only):
            raise FileNotFoundError(
                f"Cannot resume or summarize missing output directory: {self.output_dir}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "stages").mkdir(exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)

        metadata_path = self.output_dir / "metadata.json"
        if metadata_path.is_file():
            self.metadata = _read_json(metadata_path)
            current_queue_hash = hashlib.sha256(self.queue_path.read_bytes()).hexdigest()
            if self.metadata.get("queue_sha256") != current_queue_hash:
                raise ValueError(
                    "Queue contents differ from this run's queue snapshot. "
                    "Use the original queue or start a new output directory."
                )
            if (
                Path(str(self.metadata.get("dataset_root"))).resolve()
                != self.dataset_root
            ):
                raise ValueError(
                    "Dataset root differs from the original run. Start a new output "
                    "directory instead of mixing datasets."
                )
        else:
            queue_bytes = self.queue_path.read_bytes()
            self.metadata = {
                "timestamp": _timestamp(),
                "status": "initialized",
                "queue_path": str(self.queue_path),
                "queue_sha256": hashlib.sha256(queue_bytes).hexdigest(),
                "dataset_root": str(self.dataset_root),
                "output_directory": str(self.output_dir),
                "command": shlex.join([sys.executable, *sys.argv]),
                "args": {
                    "resume": self.resume,
                    "force": self.force,
                    "device": self.requested_device,
                    "resolved_device": self.device,
                    "only_stage": self.only_stage,
                },
                "python": sys.version,
                "platform": platform.platform(),
                "device_environment": self.device_environment,
                "git_commit": _git_commit(self.project_root),
                "stages": {},
            }
            shutil.copyfile(self.queue_path, self.output_dir / "queue_snapshot.yaml")
            plan_path = self.queue_path.parent / "plan.md"
            if plan_path.is_file():
                shutil.copyfile(plan_path, self.output_dir / "plan_snapshot.md")
            else:
                (self.output_dir / "plan_snapshot.md").write_text(
                    "# Overnight Experiment Plan\n\n"
                    f"Queue source: `{self.queue_path}`\n",
                    encoding="utf-8",
                )
            _write_json(metadata_path, self.metadata)
        self.metadata.setdefault("invocations", []).append(
            {
                "timestamp": _timestamp(),
                "command": shlex.join([sys.executable, *sys.argv]),
                "resume": self.resume,
                "force": self.force,
                "summarize_only": self.summarize_only,
                "device": self.requested_device,
                "resolved_device": self.device,
                "device_environment": self.device_environment,
                "only_stage": self.only_stage,
            }
        )
        self.metadata["last_device_environment"] = self.device_environment
        self._save_top_metadata()
        (self.output_dir / "commands.txt").touch(exist_ok=True)
        failures_path = self.output_dir / "failures.csv"
        if not failures_path.exists():
            _write_csv(
                failures_path,
                [],
                default_fields=[
                    "timestamp",
                    "stage",
                    "phase",
                    "experiment_id",
                    "config_path",
                    "fatal",
                    "returncode",
                    "error",
                    "stdout_log",
                    "stderr_log",
                ],
            )
        self._load_selections()

    def _load_selections(self) -> None:
        for stage_id in REQUIRED_STAGE_IDS:
            path = self.output_dir / "stages" / stage_id / "selection.json"
            if path.is_file():
                selection = _read_json(path)
                if selection.get("status", "selected") == "selected":
                    self.selections[stage_id] = selection

    def _save_top_metadata(self) -> None:
        _write_json(self.output_dir / "metadata.json", self.metadata)

    def _candidate(self, item: Any) -> Candidate:
        spec = {"path": item} if isinstance(item, str) else dict(item)
        source_path = self._config_item_path(spec)
        config = copy.deepcopy(load_config(source_path))

        inherit_stage = spec.get("inherit_detector_conf_from")
        if inherit_stage:
            selection = self.selections.get(str(inherit_stage))
            if selection:
                inherited_conf = selection["resolved_config"]["detector"]["conf"]
            else:
                fallback = spec.get("inherit_fallback_config")
                if not fallback:
                    raise ValueError(
                        f"Config {source_path} requires selection from {inherit_stage}"
                    )
                fallback_path = Path(str(fallback))
                if not fallback_path.is_absolute():
                    fallback_path = self.project_root / fallback_path
                inherited_conf = load_config(fallback_path)["detector"]["conf"]
            offset = float(spec.get("confidence_offset", 0.0))
            config["detector"]["conf"] = min(0.95, float(inherited_conf) + offset)

        config = apply_device_to_detector_configs(config, self.device)

        experiment_id = str(spec.get("id", config["name"]))
        if not experiment_id or any(char in experiment_id for char in "/\\"):
            raise ValueError(f"Unsafe experiment id: {experiment_id!r}")
        return Candidate(experiment_id, source_path, config)

    def _benchmark_command(
        self,
        *,
        config_path: Path,
        sequences: Sequence[str],
        max_frames: int | None,
        save_video: bool,
        evaluate: bool,
        attempt_dir: Path,
    ) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "src.experiments.run_benchmark",
            "--dataset-root",
            str(self.dataset_root),
            "--configs",
            str(config_path),
            "--sequences",
            *sequences,
            "--output-root",
            str(attempt_dir),
            "--run-id",
            "benchmark",
        ]
        if max_frames is not None:
            command.extend(["--max-frames", str(max_frames)])
        if save_video:
            command.append("--save-video")
        if evaluate:
            command.append("--evaluate")
        command.extend(["--device", self.device])
        return command

    def _experiment_dir(self, stage_id: str, phase: str, experiment_id: str) -> Path:
        return self.output_dir / "stages" / stage_id / phase / experiment_id

    def _run_experiment(
        self,
        *,
        stage: Mapping[str, Any],
        phase: str,
        candidate: Candidate,
        sequences: Sequence[str],
        max_frames: int | None,
        save_video: bool,
        evaluate: bool,
    ) -> dict[str, Any]:
        stage_id = str(stage["id"])
        experiment_dir = self._experiment_dir(
            stage_id, phase, candidate.experiment_id
        )
        metadata_path = experiment_dir / "metadata.json"
        existing = _read_json(metadata_path) if metadata_path.is_file() else None
        if existing and existing.get("status") == "completed" and not self.force:
            print(f"SKIP completed: {stage_id}/{phase}/{candidate.experiment_id}")
            return existing

        attempts = list(existing.get("attempts", [])) if existing else []
        attempt_number = len(attempts) + 1
        attempt_name = f"{attempt_number:03d}"
        attempt_dir = experiment_dir / "attempts" / attempt_name
        attempt_dir.mkdir(parents=True, exist_ok=False)
        resolved_config_path = attempt_dir / "resolved_config.yaml"
        _write_yaml(resolved_config_path, candidate.config)

        log_stem = (
            f"{stage_id}__{phase}__{candidate.experiment_id}__attempt{attempt_name}"
        )
        stdout_path = self.output_dir / "logs" / f"{log_stem}.stdout.log"
        stderr_path = self.output_dir / "logs" / f"{log_stem}.stderr.log"
        combined_path = self.output_dir / "logs" / f"{log_stem}.log"
        command = self._benchmark_command(
            config_path=resolved_config_path,
            sequences=sequences,
            max_frames=max_frames,
            save_video=save_video,
            evaluate=evaluate,
            attempt_dir=attempt_dir,
        )
        command_text = shlex.join(command)
        with (self.output_dir / "commands.txt").open("a", encoding="utf-8") as handle:
            handle.write(command_text + "\n")

        attempt = {
            "attempt": attempt_number,
            "timestamp": _timestamp(),
            "status": "running",
            "command": command_text,
            "resolved_config_path": str(resolved_config_path),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "combined_log": str(combined_path),
        }
        experiment_metadata: dict[str, Any] = {
            "timestamp": existing.get("timestamp", _timestamp()) if existing else _timestamp(),
            "status": "running",
            "stage": stage_id,
            "phase": phase,
            "experiment_id": candidate.experiment_id,
            "source_config_path": str(candidate.source_path),
            "resolved_config": candidate.config,
            "dataset_root": str(self.dataset_root),
            "sequences": list(sequences),
            "max_frames": max_frames,
            "save_video": save_video,
            "evaluate": evaluate,
            "fatal_stage": bool(stage.get("fatal", False)),
            "requested_device": self.requested_device,
            "resolved_device": self.device,
            "device_environment": self.device_environment,
            "attempts": [*attempts, attempt],
        }
        _write_json(metadata_path, experiment_metadata)

        print(f"RUN {stage_id}/{phase}/{candidate.experiment_id}")
        started = time.perf_counter()
        returncode: int | None = None
        error: str | None = None
        try:
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            matplotlib_dir = self.output_dir / ".cache" / "matplotlib"
            matplotlib_dir.mkdir(parents=True, exist_ok=True)
            environment["MPLCONFIGDIR"] = str(matplotlib_dir)
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
            returncode = completed.returncode
            benchmark_metadata_path = (
                attempt_dir / "benchmark" / "metadata.json"
            )
            benchmark_status = None
            if benchmark_metadata_path.is_file():
                benchmark_status = _read_json(benchmark_metadata_path).get("status")
            if returncode != 0 or benchmark_status != "completed":
                error = (
                    f"Benchmark failed with return code {returncode}; "
                    f"benchmark status={benchmark_status!r}"
                )
        except KeyboardInterrupt:
            error = "Interrupted by user"
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            runtime = time.perf_counter() - started
            combined_path.write_text(
                "=== STDOUT ===\n"
                + (stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else "")
                + "\n=== STDERR ===\n"
                + (stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""),
                encoding="utf-8",
            )
            final_status = "failed" if error else "completed"
            attempt.update(
                {
                    "status": final_status,
                    "completed_at": _timestamp(),
                    "runtime_seconds": runtime,
                    "returncode": returncode,
                    "error": error,
                    "benchmark_dir": str(attempt_dir / "benchmark"),
                }
            )
            experiment_metadata.update(
                {
                    "status": final_status,
                    "completed_at": _timestamp(),
                    "latest_attempt": attempt_number,
                    "latest_benchmark_dir": str(attempt_dir / "benchmark"),
                    "attempts": [*attempts, attempt],
                    "error": error,
                }
            )
            _write_json(metadata_path, experiment_metadata)
            if error:
                self._append_failure(
                    stage=stage,
                    phase=phase,
                    candidate=candidate,
                    returncode=returncode,
                    error=error,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )

        if error:
            print(f"FAILED {stage_id}/{phase}/{candidate.experiment_id}: {error}")
        else:
            print(f"DONE {stage_id}/{phase}/{candidate.experiment_id}")
        return experiment_metadata

    def _append_failure(
        self,
        *,
        stage: Mapping[str, Any],
        phase: str,
        candidate: Candidate,
        returncode: int | None,
        error: str,
        stdout_path: Path,
        stderr_path: Path,
    ) -> None:
        path = self.output_dir / "failures.csv"
        fields = [
            "timestamp",
            "stage",
            "phase",
            "experiment_id",
            "config_path",
            "fatal",
            "returncode",
            "error",
            "stdout_log",
            "stderr_log",
        ]
        write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": _timestamp(),
                    "stage": stage["id"],
                    "phase": phase,
                    "experiment_id": candidate.experiment_id,
                    "config_path": candidate.source_path,
                    "fatal": bool(stage.get("fatal", False)),
                    "returncode": returncode,
                    "error": error,
                    "stdout_log": stdout_path,
                    "stderr_log": stderr_path,
                }
            )

    def _collect_rows(
        self,
        stage_id: str,
        *,
        per_sequence: bool = False,
        phase: str | None = None,
    ) -> list[dict[str, Any]]:
        stage_dir = self.output_dir / "stages" / stage_id
        pattern = f"{phase}/*/metadata.json" if phase else "*/*/metadata.json"
        rows: list[dict[str, Any]] = []
        for metadata_path in sorted(stage_dir.glob(pattern)):
            metadata = _read_json(metadata_path)
            if metadata.get("status") != "completed":
                continue
            benchmark_dir = Path(metadata["latest_benchmark_dir"])
            filename = (
                "per_sequence_metrics.csv"
                if per_sequence
                else "summary_by_method.csv"
            )
            for row in _read_csv(benchmark_dir / "evaluation" / filename):
                rows.append(
                    {
                        "stage": stage_id,
                        "phase": metadata["phase"],
                        "experiment_id": metadata["experiment_id"],
                        "source_config_path": metadata["source_config_path"],
                        **row,
                    }
                )
        return rows

    def _write_stage_summary(self, stage_id: str) -> None:
        filename = SUMMARY_FILENAMES.get(stage_id)
        if not filename:
            return
        rows = self._collect_rows(stage_id)
        selection = self.selections.get(stage_id)
        if selection:
            screen_ranks = {
                str(row["experiment_id"]): rank
                for rank, row in enumerate(
                    selection.get("screening_ranking", []), start=1
                )
            }
            promotion_ranks = {
                str(row["experiment_id"]): rank
                for rank, row in enumerate(selection.get("ranking", []), start=1)
            }
            selected_id = selection.get("selected_experiment_id")
            for row in rows:
                experiment_id = str(row["experiment_id"])
                row["screening_rank"] = screen_ranks.get(experiment_id)
                row["promotion_rank"] = promotion_ranks.get(experiment_id)
                row["selected"] = experiment_id == selected_id
        _write_csv(
            self.output_dir / "summaries" / filename,
            rows,
            default_fields=[
                "stage",
                "phase",
                "experiment_id",
                "source_config_path",
                "method",
                "MOTA",
                "IDF1",
                "IDS",
                "FP",
                "FN",
                "FPS",
            ],
        )

    def _select_and_promote(
        self,
        stage: Mapping[str, Any],
        candidates: Sequence[Candidate],
    ) -> tuple[bool, list[str]]:
        stage_id = str(stage["id"])
        screening_rows = self._collect_rows(stage_id, phase="screen")
        ranked_screen = _rank_rows(screening_rows, str(stage["selection_rule"]))
        promote_top = int(stage.get("promote_top", 1))
        promoted_ids: list[str] = []
        for row in ranked_screen:
            experiment_id = str(row["experiment_id"])
            if experiment_id not in promoted_ids:
                promoted_ids.append(experiment_id)
            if len(promoted_ids) >= promote_top:
                break

        candidate_map = {candidate.experiment_id: candidate for candidate in candidates}
        failures: list[str] = []
        for experiment_id in promoted_ids:
            candidate = candidate_map[experiment_id]
            metadata = self._run_experiment(
                stage=stage,
                phase="promoted",
                candidate=candidate,
                sequences=self._resolve_sequences(stage["promotion_sequences"]),
                max_frames=stage.get("promotion_max_frames"),
                save_video=bool(stage.get("promotion_save_video", False)),
                evaluate=bool(stage.get("promotion_evaluate", True)),
            )
            if metadata["status"] != "completed":
                failures.append(experiment_id)
                if stage.get("fatal"):
                    break

        promotion_rows = self._collect_rows(stage_id, phase="promoted")
        ranked_promotion = _rank_rows(
            promotion_rows, str(stage["selection_rule"])
        )
        if ranked_promotion:
            best = ranked_promotion[0]
            candidate = candidate_map[str(best["experiment_id"])]
            experiment_dir = self._experiment_dir(
                stage_id, "promoted", candidate.experiment_id
            )
            selection = {
                "timestamp": _timestamp(),
                "status": "selected",
                "stage": stage_id,
                "selection_rule": stage["selection_rule"],
                "selected_experiment_id": candidate.experiment_id,
                "source_config_path": str(candidate.source_path),
                "resolved_config": candidate.config,
                "experiment_dir": str(experiment_dir),
                "metrics": best,
                "ranking": ranked_promotion,
                "screening_ranking": ranked_screen,
            }
            selection_path = (
                self.output_dir / "stages" / stage_id / "selection.json"
            )
            _write_json(selection_path, selection)
            self.selections[stage_id] = selection
            return True, failures
        unavailable_selection = {
            "timestamp": _timestamp(),
            "status": "unavailable",
            "stage": stage_id,
            "selection_rule": stage["selection_rule"],
            "reason": "No promoted full-validation result completed.",
            "screening_ranking": ranked_screen,
        }
        _write_json(
            self.output_dir / "stages" / stage_id / "selection.json",
            unavailable_selection,
        )
        self.selections.pop(stage_id, None)
        return False, failures

    def _run_standard_stage(self, stage: Mapping[str, Any]) -> bool:
        stage_id = str(stage["id"])
        candidates = [self._candidate(item) for item in stage.get("configs", [])]
        failures: list[str] = []
        for candidate in candidates:
            metadata = self._run_experiment(
                stage=stage,
                phase="runs",
                candidate=candidate,
                sequences=self._resolve_sequences(stage["sequences"]),
                max_frames=stage.get("max_frames"),
                save_video=bool(stage.get("save_video", False)),
                evaluate=bool(stage.get("evaluate", True)),
            )
            if metadata["status"] != "completed":
                failures.append(candidate.experiment_id)
                if stage.get("fatal"):
                    break
        self._write_stage_summary(stage_id)
        return not failures

    def _run_tuning_stage(self, stage: Mapping[str, Any]) -> bool:
        stage_id = str(stage["id"])
        candidates = [self._candidate(item) for item in stage.get("configs", [])]
        screening_failures: list[str] = []
        for candidate in candidates:
            metadata = self._run_experiment(
                stage=stage,
                phase="screen",
                candidate=candidate,
                sequences=self._resolve_sequences(stage["sequences"]),
                max_frames=stage.get("max_frames"),
                save_video=bool(stage.get("save_video", False)),
                evaluate=bool(stage.get("evaluate", True)),
            )
            if metadata["status"] != "completed":
                screening_failures.append(candidate.experiment_id)
                if stage.get("fatal"):
                    break
        selected, promotion_failures = self._select_and_promote(stage, candidates)
        self._write_stage_summary(stage_id)
        return selected and not screening_failures and not promotion_failures

    def _find_experiment(
        self,
        stage_id: str,
        *,
        config_name: str | None = None,
        experiment_id: str | None = None,
        preferred_phase: str | None = None,
    ) -> dict[str, Any] | None:
        stage_dir = self.output_dir / "stages" / stage_id
        phases = (
            [preferred_phase]
            if preferred_phase
            else ["promoted", "runs", "screen"]
        )
        seen: set[str] = set()
        for phase in phases:
            if not phase or phase in seen:
                continue
            seen.add(phase)
            for metadata_path in sorted((stage_dir / phase).glob("*/metadata.json")):
                metadata = _read_json(metadata_path)
                if metadata.get("status") != "completed":
                    continue
                if experiment_id and metadata.get("experiment_id") != experiment_id:
                    continue
                if config_name and metadata.get("resolved_config", {}).get("name") != config_name:
                    continue
                return metadata
        return None

    def _resolve_comparison_source(
        self, source: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        select_from = source.get("select_from")
        if select_from and str(select_from) in self.selections:
            selection = self.selections[str(select_from)]
            selected = self._find_experiment(
                str(select_from),
                experiment_id=selection["selected_experiment_id"],
                preferred_phase="promoted",
            )
            if selected is not None:
                return selected

        fallback = source.get("fallback")
        if fallback:
            return self._find_experiment(
                str(fallback["stage"]),
                config_name=str(fallback["config_name"]),
            )
        if source.get("stage"):
            return self._find_experiment(
                str(source["stage"]),
                config_name=str(source["config_name"]),
            )
        return None

    def _build_final_comparison(
        self, stage: Mapping[str, Any], *, write_stage_metadata: bool = True
    ) -> bool:
        summary_rows: list[dict[str, Any]] = []
        sequence_rows: list[dict[str, Any]] = []
        source_records: list[dict[str, Any]] = []
        missing_required: list[str] = []
        expected_sequences = self._resolve_sequences(stage["sequences"])
        expected_max_frames = stage.get("max_frames")

        for source in stage.get("sources", []):
            label = str(source["label"])
            metadata = self._resolve_comparison_source(source)
            if metadata is None:
                if not source.get("optional", False):
                    missing_required.append(label)
                continue
            if (
                metadata.get("sequences") != expected_sequences
                or metadata.get("max_frames") != expected_max_frames
            ):
                if not source.get("optional", False):
                    missing_required.append(label)
                continue
            benchmark_dir = Path(metadata["latest_benchmark_dir"])
            method_rows = _read_csv(
                benchmark_dir / "evaluation" / "summary_by_method.csv"
            )
            per_sequence_rows = _read_csv(
                benchmark_dir / "evaluation" / "per_sequence_metrics.csv"
            )
            evaluated_sequences = {
                row.get("sequence_name") for row in per_sequence_rows
            }
            if (
                len(method_rows) != 1
                or evaluated_sequences != set(expected_sequences)
            ):
                if not source.get("optional", False):
                    missing_required.append(label)
                continue
            summary_rows.append(
                {
                    "comparison_label": label,
                    "source_stage": metadata["stage"],
                    "source_experiment_id": metadata["experiment_id"],
                    "source_config_path": metadata["source_config_path"],
                    **method_rows[0],
                }
            )
            for row in per_sequence_rows:
                sequence_rows.append(
                    {
                        "comparison_label": label,
                        "source_stage": metadata["stage"],
                        "source_experiment_id": metadata["experiment_id"],
                        **row,
                    }
                )
            source_records.append(
                {
                    "label": label,
                    "stage": metadata["stage"],
                    "experiment_id": metadata["experiment_id"],
                    "benchmark_dir": str(benchmark_dir),
                }
            )

        _write_csv(
            self.output_dir / "summaries" / "final_summary_by_method.csv",
            summary_rows,
            default_fields=[
                "comparison_label",
                "method",
                "MOTA",
                "IDF1",
                "IDS",
                "FP",
                "FN",
                "FPS",
            ],
        )
        _write_csv(
            self.output_dir / "summaries" / "final_summary_by_sequence.csv",
            sequence_rows,
            default_fields=[
                "comparison_label",
                "method",
                "sequence_name",
                "MOTA",
                "IDF1",
                "IDS",
                "FP",
                "FN",
                "FPS",
            ],
        )

        status = "failed" if missing_required else "completed"
        if write_stage_metadata:
            stage_dir = self.output_dir / "stages" / str(stage["id"])
            _write_json(
                stage_dir / "metadata.json",
                {
                    "timestamp": _timestamp(),
                    "status": status,
                    "reuse_prior_full_runs": True,
                    "sources": source_records,
                    "missing_required_sources": missing_required,
                },
            )
        return not missing_required

    def _write_final_notes(self) -> None:
        final_rows = _read_csv(
            self.output_dir / "summaries" / "final_summary_by_method.csv"
        )
        failures = _read_csv(self.output_dir / "failures.csv")
        indexed = {row.get("comparison_label", ""): row for row in final_rows}
        lines = [
            "# Overnight Experiment Notes",
            "",
            f"Updated: {_timestamp()}",
            "",
            "## Selected Runs",
            "",
        ]
        if final_rows:
            for row in final_rows:
                lines.append(
                    f"- `{row['comparison_label']}`: `{row['method']}`, "
                    f"MOTA={row.get('MOTA')}, IDF1={row.get('IDF1')}, "
                    f"FP={row.get('FP')}, FN={row.get('FN')}, "
                    f"IDS={row.get('IDS')}, FPS={row.get('FPS')}"
                )
        else:
            lines.append("- No complete final comparison rows are available yet.")

        lines.extend(["", "## SAHI Status", ""])
        m2 = indexed.get("m2_best")
        m4 = indexed.get("m4_sahi_best")
        if m2 and m4:
            lines.append(
                "- SAHI completed full validation. Compare it against M2 using "
                f"MOTA {m4.get('MOTA')} vs {m2.get('MOTA')}, "
                f"IDF1 {m4.get('IDF1')} vs {m2.get('IDF1')}, "
                f"FP {m4.get('FP')} vs {m2.get('FP')}, "
                f"FN {m4.get('FN')} vs {m2.get('FN')}, and "
                f"FPS {m4.get('FPS')} vs {m2.get('FPS')}."
            )
        elif m4:
            lines.append("- SAHI completed, but no M2 comparison row is available.")
        else:
            lines.append(
                "- No full-validation SAHI result is available. Review "
                "`failures.csv` and the SAHI logs; this does not invalidate "
                "the M1/M2/M3 comparison."
            )

        lines.extend(["", "## Failures", ""])
        if failures:
            lines.append(f"- Recorded failed attempts: {len(failures)}")
            for failure in failures[-10:]:
                lines.append(
                    f"- `{failure.get('stage')}/{failure.get('experiment_id')}`: "
                    f"{failure.get('error')}"
                )
        else:
            lines.append("- No failed attempts recorded.")
        lines.extend(
            [
                "",
                "Full command history is in `commands.txt`. Per-attempt stdout, "
                "stderr, and combined logs are in `logs/`.",
            ]
        )
        (self.output_dir / "summaries" / "final_notes.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def _summarize_all(self) -> None:
        for stage_id in SUMMARY_FILENAMES:
            self._write_stage_summary(stage_id)
        final_stage = next(
            stage for stage in self.stages if stage["id"] == "final_comparison"
        )
        self._build_final_comparison(final_stage, write_stage_metadata=False)
        self._write_final_notes()

    def _dry_run_command(
        self,
        stage: Mapping[str, Any],
        phase: str,
        candidate: Candidate,
        sequences: Sequence[str],
        max_frames: int | None,
        save_video: bool,
        evaluate: bool,
    ) -> str:
        attempt_dir = (
            self.output_dir
            / "stages"
            / str(stage["id"])
            / phase
            / candidate.experiment_id
            / "attempts"
            / "001"
        )
        config_path = attempt_dir / "resolved_config.yaml"
        return shlex.join(
            self._benchmark_command(
                config_path=config_path,
                sequences=sequences,
                max_frames=max_frames,
                save_video=save_video,
                evaluate=evaluate,
                attempt_dir=attempt_dir,
            )
        )

    def print_dry_run(self) -> None:
        print(f"Queue: {self.queue_path}")
        print(f"Dataset: {self.dataset_root}")
        print(f"Output: {self.output_dir}")
        for stage in self.stages:
            if self.only_stage and stage["id"] != self.only_stage:
                continue
            print(
                f"\n[{stage['id']}] kind={stage['kind']} "
                f"fatal={bool(stage.get('fatal', False))}"
            )
            if stage["kind"] == "comparison":
                print("  Reuse prior full-validation runs and write final summaries.")
                continue
            candidates = [self._candidate(item) for item in stage.get("configs", [])]
            for candidate in candidates:
                command = self._dry_run_command(
                    stage,
                    "screen" if stage["kind"] == "tuning" else "runs",
                    candidate,
                    self._resolve_sequences(stage["sequences"]),
                    stage.get("max_frames"),
                    bool(stage.get("save_video", False)),
                    bool(stage.get("evaluate", True)),
                )
                print(f"  {command}")
            if stage["kind"] == "tuning":
                print(
                    f"  Promote top {stage.get('promote_top', 1)} using rule "
                    f"{stage['selection_rule']} and run on "
                    f"{len(self._resolve_sequences(stage['promotion_sequences']))} "
                    "sequences."
                )

    def run(self) -> Path:
        if self.dry_run:
            self.print_dry_run()
            return self.output_dir

        self._initialize_output()
        if self.summarize_only:
            self._summarize_all()
            print(f"Summaries refreshed: {self.output_dir / 'summaries'}")
            return self.output_dir

        self.metadata["status"] = "running"
        self.metadata["last_started_at"] = _timestamp()
        self._save_top_metadata()
        try:
            for stage in self.stages:
                stage_id = str(stage["id"])
                if self.only_stage and stage_id != self.only_stage:
                    continue
                print(f"\n=== Stage: {stage_id} ===")
                stage_metadata_path = (
                    self.output_dir / "stages" / stage_id / "metadata.json"
                )
                previous_stage_metadata = (
                    _read_json(stage_metadata_path)
                    if stage_metadata_path.is_file()
                    else {}
                )
                started = time.perf_counter()
                if stage["kind"] == "benchmark":
                    success = self._run_standard_stage(stage)
                elif stage["kind"] == "tuning":
                    success = self._run_tuning_stage(stage)
                elif stage["kind"] == "comparison":
                    success = self._build_final_comparison(stage)
                else:
                    raise ValueError(
                        f"Unsupported stage kind for {stage_id}: {stage['kind']}"
                    )

                elapsed = time.perf_counter() - started
                stage_status = "completed" if success else "completed_with_failures"
                if (
                    previous_stage_metadata.get("status") == "completed"
                    and success
                    and not self.force
                ):
                    stage_record = {
                        "status": "completed",
                        "completed_at": previous_stage_metadata.get(
                            "completed_at", _timestamp()
                        ),
                        "runtime_seconds": previous_stage_metadata.get(
                            "runtime_seconds", elapsed
                        ),
                        "last_verified_at": _timestamp(),
                        "last_verification_seconds": elapsed,
                        "fatal": bool(stage.get("fatal", False)),
                    }
                else:
                    stage_record = {
                        "status": stage_status,
                        "completed_at": _timestamp(),
                        "runtime_seconds": elapsed,
                        "fatal": bool(stage.get("fatal", False)),
                    }
                self.metadata["stages"][stage_id] = stage_record
                stage_metadata = (
                    _read_json(stage_metadata_path)
                    if stage_metadata_path.is_file()
                    else previous_stage_metadata
                )
                stage_metadata.update(stage_record)
                _write_json(stage_metadata_path, stage_metadata)
                self._save_top_metadata()
                if not success and stage.get("fatal"):
                    raise FatalStageError(f"Fatal stage failed: {stage_id}")

            self._summarize_all()
            self.metadata["status"] = "completed"
            self.metadata["completed_at"] = _timestamp()
            self._save_top_metadata()
        except BaseException as exc:
            self.metadata["status"] = "failed"
            self.metadata["failed_at"] = _timestamp()
            self.metadata["error"] = f"{type(exc).__name__}: {exc}"
            self._save_top_metadata()
            self._write_final_notes()
            raise
        return self.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a staged, resumable overnight MOT experiment queue."
    )
    parser.add_argument("--queue", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Run directory. Defaults to outputs/overnight/<UTC timestamp>.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda:0", "mps"],
        default="auto",
        help="Inference device. auto prefers CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--only-stage",
        choices=REQUIRED_STAGE_IDS,
        default=None,
        help="Execute only one stage. Dependencies use existing selections or fallbacks.",
    )
    args = parser.parse_args()
    if args.dry_run and args.summarize_only:
        parser.error("--dry-run and --summarize-only cannot be combined")
    if args.resume and args.force:
        parser.error("--resume and --force cannot be combined")
    if args.summarize_only and (args.resume or args.force):
        parser.error("--summarize-only cannot be combined with --resume or --force")
    if args.resume and not args.output_dir:
        parser.error("--resume requires --output-dir")
    if args.force and not args.output_dir:
        parser.error("--force requires --output-dir")
    if args.summarize_only and not args.output_dir:
        parser.error("--summarize-only requires --output-dir")
    return args


def main() -> None:
    args = parse_args()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs/overnight") / _run_id()
    )
    runner = OvernightRunner(
        queue_path=Path(args.queue),
        dataset_root=Path(args.dataset_root),
        output_dir=output_dir,
        resume=args.resume,
        force=args.force,
        dry_run=args.dry_run,
        summarize_only=args.summarize_only,
        device=args.device,
        only_stage=args.only_stage,
    )
    print(format_device_diagnostics(runner.device_environment))
    result = runner.run()
    if not args.dry_run:
        print(f"\nOvernight outputs: {result}")


if __name__ == "__main__":
    main()
