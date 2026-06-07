from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.visdrone import VisDroneSequence


ALLOWED_CLASSES = frozenset({1, 4})
IOU_THRESHOLD = 0.5
MOTMETRICS_NAMES = [
    "mota",
    "motp",
    "idf1",
    "idp",
    "idr",
    "num_false_positives",
    "num_misses",
    "num_switches",
    "num_frames",
]
OUTPUT_METRICS = [
    "MOTA",
    "MOTP",
    "IDF1",
    "IDP",
    "IDR",
    "FP",
    "FN",
    "IDS",
    "num_frames",
]
METRIC_NAME_MAP = dict(zip(MOTMETRICS_NAMES, OUTPUT_METRICS, strict=True))


@dataclass(frozen=True)
class FrameObjects:
    ids: list[int]
    boxes: np.ndarray
    classes: np.ndarray


def _import_motmetrics() -> Any:
    try:
        import motmetrics as mm
    except ImportError as exc:
        raise ImportError(
            "motmetrics is required for MOT evaluation. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return mm


def _empty_frame() -> FrameObjects:
    return FrameObjects(
        ids=[],
        boxes=np.empty((0, 4), dtype=float),
        classes=np.empty((0,), dtype=int),
    )


def _parse_integer(value: str, label: str, path: Path, line_number: int) -> int:
    parsed = float(value)
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise ValueError(
            f"{path}:{line_number}: {label} must be an integer, got {value!r}"
        )
    return int(parsed)


def _read_mot_file(
    path: Path,
    *,
    kind: str,
    max_frame: int,
) -> dict[int, FrameObjects]:
    minimum_columns = 10 if kind == "ground truth" else 9
    rows_by_frame: dict[int, list[tuple[int, list[float], int]]] = defaultdict(list)
    seen_ids_by_frame: dict[int, set[int]] = defaultdict(set)
    total_rows = 0
    rows_in_range = 0
    allowed_rows = 0
    beyond_frame_rows = 0

    if not path.is_file():
        raise FileNotFoundError(f"Missing {kind} file: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row or all(not value.strip() for value in row):
                continue
            total_rows += 1
            if len(row) < minimum_columns:
                raise ValueError(
                    f"{path}:{line_number}: expected at least {minimum_columns} "
                    f"MOT columns for {kind}, found {len(row)}"
                )

            try:
                frame_id = _parse_integer(row[0], "frame ID", path, line_number)
                object_id = _parse_integer(row[1], "object ID", path, line_number)
                cls = _parse_integer(row[7], "class ID", path, line_number)
                x, y, width, height = (float(value) for value in row[2:6])
            except (TypeError, ValueError) as exc:
                if isinstance(exc, ValueError) and str(exc).startswith(str(path)):
                    raise
                raise ValueError(
                    f"{path}:{line_number}: invalid numeric MOT row: {row}"
                ) from exc

            if frame_id < 1:
                raise ValueError(
                    f"{path}:{line_number}: frame IDs must start at 1, got {frame_id}"
                )
            if kind == "predictions" and object_id < 1:
                raise ValueError(
                    f"{path}:{line_number}: prediction object IDs must be positive, "
                    f"got {object_id}"
                )
            if frame_id > max_frame:
                beyond_frame_rows += 1
                continue
            rows_in_range += 1
            if cls not in ALLOWED_CLASSES:
                continue

            allowed_rows += 1
            box = [x, y, width, height]
            if not all(math.isfinite(value) for value in box):
                raise ValueError(f"{path}:{line_number}: bounding box is not finite")
            if width <= 0 or height <= 0:
                warnings.warn(
                    f"{path}:{line_number}: skipping non-positive bounding box",
                    stacklevel=2,
                )
                continue
            if object_id in seen_ids_by_frame[frame_id]:
                raise ValueError(
                    f"{path}:{line_number}: duplicate object ID {object_id} "
                    f"in frame {frame_id}"
                )

            seen_ids_by_frame[frame_id].add(object_id)
            rows_by_frame[frame_id].append((object_id, box, cls))

    if kind == "predictions" and beyond_frame_rows:
        warnings.warn(
            f"Ignored {beyond_frame_rows} prediction rows beyond processed frame "
            f"{max_frame}: {path}",
            stacklevel=2,
        )
    if kind == "predictions" and total_rows == 0:
        warnings.warn(f"Prediction file is empty: {path}", stacklevel=2)
    elif kind == "predictions" and rows_in_range > 0 and allowed_rows == 0:
        raise ValueError(
            f"Prediction file has rows but none use evaluated classes 1 or 4: {path}"
        )

    result: dict[int, FrameObjects] = {}
    for frame_id, objects in rows_by_frame.items():
        result[frame_id] = FrameObjects(
            ids=[item[0] for item in objects],
            boxes=np.asarray([item[1] for item in objects], dtype=float),
            classes=np.asarray([item[2] for item in objects], dtype=int),
        )
    return result


def _validate_dataset_sequence(
    dataset_root: Path,
    sequence_name: str,
    max_frames: int | None,
) -> int:
    annotation_path = dataset_root / "annotations" / f"{sequence_name}.txt"
    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"Annotation file does not exist for {sequence_name}: {annotation_path}"
        )
    return len(VisDroneSequence(dataset_root, sequence_name, max_frames=max_frames))


def discover_prediction_files(
    tracks_root: str | Path,
    sequences: Sequence[str],
) -> dict[str, dict[str, Path]]:
    root = Path(tracks_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Tracks root does not exist: {root}")

    sources: dict[str, dict[str, Path]] = {}

    direct_flat = {sequence: root / f"{sequence}.txt" for sequence in sequences}
    direct_nested = {
        sequence: root / sequence / "tracks.txt" for sequence in sequences
    }
    if any(path.exists() for path in direct_flat.values()):
        _require_complete_method(root.name, direct_flat)
        sources[root.name] = direct_flat
    if any(path.exists() for path in direct_nested.values()):
        if sources:
            raise ValueError(
                f"Ambiguous prediction layout under {root}: both flat and nested "
                "sequence files were found"
            )
        _require_complete_method(root.name, direct_nested)
        sources[root.name] = direct_nested

    for method_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        flat = {sequence: method_dir / f"{sequence}.txt" for sequence in sequences}
        nested = {
            sequence: method_dir / sequence / "tracks.txt" for sequence in sequences
        }
        has_flat = any(path.exists() for path in flat.values())
        has_nested = any(path.exists() for path in nested.values())
        if has_flat and has_nested:
            raise ValueError(
                f"Ambiguous prediction layout for method {method_dir.name}: "
                "both <sequence>.txt and <sequence>/tracks.txt were found"
            )
        if has_flat or has_nested:
            files = flat if has_flat else nested
            _require_complete_method(method_dir.name, files)
            if method_dir.name in sources:
                raise ValueError(f"Duplicate method name discovered: {method_dir.name}")
            sources[method_dir.name] = files

    if not sources:
        raise FileNotFoundError(
            f"No prediction files found under {root}. Expected either "
            "<sequence>.txt, <method>/<sequence>.txt, or "
            "<method>/<sequence>/tracks.txt."
        )
    return sources


def _require_complete_method(method: str, files: Mapping[str, Path]) -> None:
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Method {method!r} is missing prediction files for selected sequences: "
            + ", ".join(missing)
        )


def _load_run_performance(prediction_path: Path) -> tuple[float | None, float | None]:
    metadata_path = prediction_path.parent / "metadata.json"
    if not metadata_path.is_file():
        return None, None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        runtime = metadata.get("runtime_seconds")
        fps = metadata.get("fps")
        return (
            float(runtime) if runtime is not None else None,
            float(fps) if fps is not None else None,
        )
    except (OSError, ValueError, TypeError) as exc:
        warnings.warn(
            f"Could not read run performance metadata {metadata_path}: {exc}",
            stacklevel=2,
        )
        return None, None


def _build_accumulator(
    mm: Any,
    ground_truth: Mapping[int, FrameObjects],
    predictions: Mapping[int, FrameObjects],
    num_frames: int,
) -> Any:
    accumulator = mm.MOTAccumulator(auto_id=False)
    for frame_id in range(1, num_frames + 1):
        gt = ground_truth.get(frame_id, _empty_frame())
        pred = predictions.get(frame_id, _empty_frame())
        distances = _iou_distance_matrix(gt.boxes, pred.boxes)
        if distances.size:
            distances[gt.classes[:, None] != pred.classes[None, :]] = np.nan
        accumulator.update(gt.ids, pred.ids, distances, frameid=frame_id)
    return accumulator


def _iou_distance_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """Match motmetrics IoU distance without its removed NumPy 1.x API call."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.empty((len(gt_boxes), len(pred_boxes)), dtype=float)

    gt_xy1 = gt_boxes[:, None, :2]
    gt_xy2 = gt_xy1 + gt_boxes[:, None, 2:]
    pred_xy1 = pred_boxes[None, :, :2]
    pred_xy2 = pred_xy1 + pred_boxes[None, :, 2:]

    intersection_size = np.maximum(
        np.minimum(gt_xy2, pred_xy2) - np.maximum(gt_xy1, pred_xy1),
        0.0,
    )
    intersection = np.prod(intersection_size, axis=2)
    gt_area = np.prod(gt_boxes[:, 2:], axis=1)[:, None]
    pred_area = np.prod(pred_boxes[:, 2:], axis=1)[None, :]
    union = gt_area + pred_area - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=float),
        where=union > 0,
    )
    distance = 1.0 - iou
    distance[distance > IOU_THRESHOLD] = np.nan
    return distance


def _number(value: Any) -> float | int | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _metrics_row(summary: Any, name: str) -> dict[str, float | int | None]:
    row: dict[str, float | int | None] = {}
    for source_name, output_name in METRIC_NAME_MAP.items():
        value = _number(summary.loc[name, source_name])
        if output_name in {"FP", "FN", "IDS", "num_frames"} and value is not None:
            value = int(round(value))
        row[output_name] = value
    return row


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty metrics CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_by_sequence(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    sequence_names = sorted({str(row["sequence_name"]) for row in rows})
    for sequence_name in sequence_names:
        output: dict[str, Any] = {"sequence_name": sequence_name}
        indexed = {
            str(row["method"]): row
            for row in rows
            if row["sequence_name"] == sequence_name
        }
        for method in methods:
            for metric in [*OUTPUT_METRICS, "runtime_seconds", "FPS"]:
                output[f"{method}_{metric}"] = indexed[method].get(metric)
        result.append(output)
    return result


def evaluate_mot(
    *,
    dataset_root: str | Path,
    tracks_root: str | Path,
    sequences: Sequence[str],
    output_dir: str | Path,
    max_frames: int | None = None,
) -> dict[str, Any]:
    if not sequences:
        raise ValueError("At least one sequence is required")
    if len(set(sequences)) != len(sequences):
        raise ValueError("Sequence names must be unique")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")

    mm = _import_motmetrics()
    dataset_path = Path(dataset_root).resolve()
    output_path = Path(output_dir)
    prediction_files = discover_prediction_files(tracks_root, sequences)

    frame_counts = {
        sequence: _validate_dataset_sequence(dataset_path, sequence, max_frames)
        for sequence in sequences
    }
    ground_truth = {
        sequence: _read_mot_file(
            dataset_path / "annotations" / f"{sequence}.txt",
            kind="ground truth",
            max_frame=frame_counts[sequence],
        )
        for sequence in sequences
    }

    per_sequence_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    metric_host = mm.metrics.create()

    for method, method_files in prediction_files.items():
        accumulators: list[Any] = []
        runtimes: list[float] = []
        runtime_frames = 0
        for sequence in sequences:
            prediction_path = method_files[sequence]
            predictions = _read_mot_file(
                prediction_path,
                kind="predictions",
                max_frame=frame_counts[sequence],
            )
            accumulator = _build_accumulator(
                mm,
                ground_truth[sequence],
                predictions,
                frame_counts[sequence],
            )
            accumulators.append(accumulator)
            sequence_summary = metric_host.compute(
                accumulator,
                metrics=MOTMETRICS_NAMES,
                name=sequence,
            )
            runtime, fps = _load_run_performance(prediction_path)
            row: dict[str, Any] = {
                "method": method,
                "sequence_name": sequence,
                **_metrics_row(sequence_summary, sequence),
                "runtime_seconds": runtime,
                "FPS": fps,
                "prediction_path": str(prediction_path.resolve()),
            }
            per_sequence_rows.append(row)
            if runtime is not None and runtime > 0:
                runtimes.append(runtime)
                runtime_frames += frame_counts[sequence]

        method_summary = metric_host.compute_many(
            accumulators,
            names=list(sequences),
            metrics=MOTMETRICS_NAMES,
            generate_overall=True,
        )
        summary_rows.append(
            {
                "method": method,
                **_metrics_row(method_summary, "OVERALL"),
                "runtime_seconds": sum(runtimes) if runtimes else None,
                "FPS": (
                    runtime_frames / sum(runtimes)
                    if runtimes and sum(runtimes) > 0
                    else None
                ),
            }
        )

    methods = list(prediction_files)
    sequence_summary_rows = _summary_by_sequence(per_sequence_rows, methods)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_csv(output_path / "per_sequence_metrics.csv", per_sequence_rows)
    _write_csv(output_path / "summary_metrics.csv", summary_rows)
    _write_csv(output_path / "summary_by_method.csv", summary_rows)
    _write_csv(output_path / "summary_by_sequence.csv", sequence_summary_rows)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_path),
        "tracks_root": str(Path(tracks_root).resolve()),
        "sequences": list(sequences),
        "classes": sorted(ALLOWED_CLASSES),
        "iou_threshold": IOU_THRESHOLD,
        "max_frames": max_frames,
        "per_sequence": per_sequence_rows,
        "summary_by_method": summary_rows,
        "summary_by_sequence": sequence_summary_rows,
    }
    (output_path / "summary_metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VisDrone MOT predictions for pedestrian and car classes."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to VisDrone2019-MOT-val containing sequences/ and annotations/.",
    )
    parser.add_argument(
        "--tracks-root",
        required=True,
        help="Prediction root using a supported MOT output layout.",
    )
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Evaluate only the first N frames, matching a subset tracking run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_mot(
        dataset_root=args.dataset_root,
        tracks_root=args.tracks_root,
        sequences=args.sequences,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
    )
    print("\nMOT evaluation summary")
    for row in payload["summary_by_method"]:
        print(
            f"  {row['method']}: MOTA={row['MOTA']}, IDF1={row['IDF1']}, "
            f"IDS={row['IDS']}, FP={row['FP']}, FN={row['FN']}, FPS={row['FPS']}"
        )
    print(f"  outputs: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
