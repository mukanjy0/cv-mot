from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.core.config import load_config
from src.core.device import (
    apply_device_to_detector_configs,
    device_diagnostics,
    format_device_diagnostics,
    resolve_device,
)
from src.data.visdrone import VisDroneSequence
from src.evaluation.evaluate_mot import evaluate_mot
from src.experiments.run_sequence import run_sequence


REPRESENTATIVE_SEQUENCES = [
    "uav0000137_00458_v",
    "uav0000268_05773_v",
    "uav0000117_02622_v",
    "uav0000305_00000_v",
    "uav0000086_00000_v",
]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


def _validate_inputs(
    dataset_root: Path,
    config_paths: Sequence[Path],
    sequences: Sequence[str],
    max_frames: int | None,
) -> list[tuple[Path, dict[str, Any]]]:
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")
    if len(set(sequences)) != len(sequences):
        raise ValueError("Sequence names must be unique")

    for sequence in sequences:
        VisDroneSequence(dataset_root, sequence, max_frames=max_frames)
        annotation_path = dataset_root / "annotations" / f"{sequence}.txt"
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"Annotation file does not exist for {sequence}: {annotation_path}"
            )

    configs = [(path.resolve(), load_config(path)) for path in config_paths]
    method_names = [config["name"] for _, config in configs]
    if len(set(method_names)) != len(method_names):
        raise ValueError(f"Config names must be unique, got: {method_names}")
    return configs


def run_benchmark(
    *,
    dataset_root: str | Path,
    config_paths: Sequence[str | Path],
    sequences: Sequence[str],
    max_frames: int | None = None,
    save_video: bool = False,
    evaluate: bool = False,
    output_root: str | Path = "outputs/benchmarks",
    run_id: str | None = None,
    video_fps: float = 30.0,
    device: str = "auto",
    command: Sequence[str] | None = None,
) -> Path:
    dataset_path = Path(dataset_root).resolve()
    loaded_configs = _validate_inputs(
        dataset_path,
        [Path(path) for path in config_paths],
        sequences,
        max_frames,
    )
    resolved_device = resolve_device(device)
    diagnostics = device_diagnostics(device)
    resolved_configs = [
        (path, apply_device_to_detector_configs(config, resolved_device))
        for path, config in loaded_configs
    ]
    print(format_device_diagnostics(diagnostics))
    benchmark_id = run_id or _default_run_id()
    output_dir = Path(output_root).resolve() / benchmark_id
    if output_dir.exists():
        raise FileExistsError(
            f"Benchmark output already exists: {output_dir}. Choose another --run-id."
        )
    output_dir.mkdir(parents=True)

    git_commit = _git_commit()
    args_payload = {
        "dataset_root": str(dataset_path),
        "configs": [str(path) for path, _ in resolved_configs],
        "sequences": list(sequences),
        "max_frames": max_frames,
        "save_video": save_video,
        "evaluate": evaluate,
        "output_root": str(Path(output_root).resolve()),
        "run_id": benchmark_id,
        "video_fps": video_fps,
        "device": device,
        "resolved_device": resolved_device,
    }
    metadata: dict[str, Any] = {
        "timestamp": _timestamp(),
        "run_id": benchmark_id,
        "status": "running",
        "command": shlex.join(command) if command else None,
        "args": args_payload,
        "dataset_root": str(dataset_path),
        "selected_sequences": list(sequences),
        "config_paths": [str(path) for path, _ in resolved_configs],
        "output_directory": str(output_dir),
        "git_commit": git_commit,
        "device_environment": diagnostics,
        "runs": [],
    }
    metadata_path = output_dir / "metadata.json"
    _write_json(metadata_path, metadata)

    try:
        for config_path, config in resolved_configs:
            method = config["name"]
            for sequence in sequences:
                run_dir = output_dir / "runs" / method / sequence
                tracks_path = run_dir / "tracks.txt"
                video_path = run_dir / "video.mp4"
                started_at = _timestamp()
                wall_start = time.perf_counter()
                result = run_sequence(
                    config_path=config_path,
                    dataset_root=dataset_path,
                    sequence_name=sequence,
                    max_frames=max_frames,
                    save_video=save_video,
                    video_fps=video_fps,
                    tracks_path=tracks_path,
                    video_path=video_path,
                    device=resolved_device,
                )
                wall_runtime = time.perf_counter() - wall_start
                run_metadata = {
                    "timestamp": started_at,
                    "command": metadata["command"],
                    "args": args_payload,
                    "dataset_root": str(dataset_path),
                    "sequence_name": sequence,
                    "method": method,
                    "config_path": str(config_path),
                    "resolved_config": config,
                    "max_frames": max_frames,
                    "save_video": save_video,
                    "evaluate": evaluate,
                    "num_frames_processed": result["frames_processed"],
                    "runtime_seconds": result["runtime_seconds"],
                    "wall_runtime_seconds": wall_runtime,
                    "fps": result["fps"],
                    "tracks_produced": result["tracks_produced"],
                    "track_rows": result["track_rows"],
                    "tracks_path": str(tracks_path.resolve()),
                    "video_path": str(video_path.resolve()) if save_video else None,
                    "git_commit": git_commit,
                    "device_environment": diagnostics,
                    "resolved_device": resolved_device,
                }
                _write_json(run_dir / "metadata.json", run_metadata)
                metadata["runs"].append(run_metadata)
                _write_json(metadata_path, metadata)
                print(
                    f"Completed {method} / {sequence}: "
                    f"{result['frames_processed']} frames at {result['fps']:.2f} FPS"
                )

        if evaluate:
            evaluation = evaluate_mot(
                dataset_root=dataset_path,
                tracks_root=output_dir / "runs",
                sequences=sequences,
                output_dir=output_dir / "evaluation",
                max_frames=max_frames,
            )
            metadata["evaluation_summary"] = evaluation["summary_by_method"]

        metadata["status"] = "completed"
        metadata["completed_at"] = _timestamp()
        _write_json(metadata_path, metadata)
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["failed_at"] = _timestamp()
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(metadata_path, metadata)
        raise

    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple MOT methods on a representative VisDrone subset."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=REPRESENTATIVE_SEQUENCES,
        help="Sequence names. Defaults to the five representative sequences.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--output-root", default="outputs/benchmarks")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda:0", "mps"],
        default="auto",
        help="Inference device. auto prefers CUDA, then MPS, then CPU.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = run_benchmark(
        dataset_root=args.dataset_root,
        config_paths=args.configs,
        sequences=args.sequences,
        max_frames=args.max_frames,
        save_video=args.save_video,
        evaluate=args.evaluate,
        output_root=args.output_root,
        run_id=args.run_id,
        video_fps=args.video_fps,
        device=args.device,
        command=[sys.executable, "-m", "src.experiments.run_benchmark", *sys.argv[1:]],
    )
    print(f"\nBenchmark outputs: {output_dir}")


if __name__ == "__main__":
    main()
