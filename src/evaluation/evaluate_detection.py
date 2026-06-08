from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.core.bbox import iou_matrix_xyxy, xywh_to_xyxy
from src.core.config import load_config
from src.core.device import apply_device_to_detector_configs, resolve_device
from src.data.visdrone import VisDroneSequence, list_sequence_names
from src.evaluation.evaluate_mot import ALLOWED_CLASSES, _read_mot_file
from src.experiments.run_sequence import build_detector


IOU_THRESHOLD = 0.50
CLASS_NAMES = {
    1: "pedestrian",
    4: "car",
}
AP50_DEFINITION = "all-point interpolated precision envelope"


@dataclass(frozen=True)
class BoxRecord:
    sequence_name: str
    frame_id: int
    cls: int
    xyxy: np.ndarray
    conf: float = 1.0


@dataclass(frozen=True)
class DetectionMetrics:
    gt_boxes: int
    predicted_boxes: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    ap50: float | None


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


def _number(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _group_key(record: BoxRecord) -> tuple[str, int, int]:
    return record.sequence_name, record.frame_id, record.cls


def _match_counts(
    gt_records: Sequence[BoxRecord],
    pred_records: Sequence[BoxRecord],
    iou_threshold: float,
) -> tuple[int, int]:
    gt_by_key: dict[tuple[str, int, int], list[np.ndarray]] = defaultdict(list)
    pred_by_key: dict[tuple[str, int, int], list[BoxRecord]] = defaultdict(list)
    for record in gt_records:
        gt_by_key[_group_key(record)].append(record.xyxy)
    for record in pred_records:
        pred_by_key[_group_key(record)].append(record)

    true_positives = 0
    false_positives = 0
    for key, predictions in pred_by_key.items():
        gt_boxes = gt_by_key.get(key, [])
        matched_gt: set[int] = set()
        sorted_predictions = sorted(
            predictions, key=lambda record: record.conf, reverse=True
        )
        if gt_boxes:
            gt_array = np.asarray(gt_boxes, dtype=float)
        else:
            gt_array = np.empty((0, 4), dtype=float)

        for prediction in sorted_predictions:
            if len(gt_array) == 0:
                false_positives += 1
                continue
            ious = iou_matrix_xyxy(
                gt_array, np.asarray(prediction.xyxy, dtype=float).reshape(1, 4)
            ).reshape(-1)
            if matched_gt:
                ious[list(matched_gt)] = -1.0
            best_idx = int(np.argmax(ious))
            if float(ious[best_idx]) >= iou_threshold:
                true_positives += 1
                matched_gt.add(best_idx)
            else:
                false_positives += 1

    return true_positives, false_positives


def _ap50(
    gt_records: Sequence[BoxRecord],
    pred_records: Sequence[BoxRecord],
    iou_threshold: float,
) -> float | None:
    if not gt_records:
        return None
    if not pred_records:
        return 0.0

    gt_by_key: dict[tuple[str, int, int], list[np.ndarray]] = defaultdict(list)
    for record in gt_records:
        gt_by_key[_group_key(record)].append(record.xyxy)

    matched_gt: set[tuple[tuple[str, int, int], int]] = set()
    tp_values: list[float] = []
    fp_values: list[float] = []
    for prediction in sorted(pred_records, key=lambda record: record.conf, reverse=True):
        key = _group_key(prediction)
        gt_boxes = gt_by_key.get(key, [])
        if not gt_boxes:
            tp_values.append(0.0)
            fp_values.append(1.0)
            continue

        gt_array = np.asarray(gt_boxes, dtype=float)
        ious = iou_matrix_xyxy(
            gt_array, np.asarray(prediction.xyxy, dtype=float).reshape(1, 4)
        ).reshape(-1)
        for gt_idx in range(len(gt_array)):
            if (key, gt_idx) in matched_gt:
                ious[gt_idx] = -1.0
        best_idx = int(np.argmax(ious))
        if float(ious[best_idx]) >= iou_threshold:
            matched_gt.add((key, best_idx))
            tp_values.append(1.0)
            fp_values.append(0.0)
        else:
            tp_values.append(0.0)
            fp_values.append(1.0)

    tp_cumsum = np.cumsum(np.asarray(tp_values, dtype=float))
    fp_cumsum = np.cumsum(np.asarray(fp_values, dtype=float))
    recalls = tp_cumsum / max(1, len(gt_records))
    precisions = np.divide(
        tp_cumsum,
        tp_cumsum + fp_cumsum,
        out=np.zeros_like(tp_cumsum),
        where=(tp_cumsum + fp_cumsum) > 0,
    )

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(len(mpre) - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
    changing = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changing + 1] - mrec[changing]) * mpre[changing + 1]))


def evaluate_records(
    gt_records: Sequence[BoxRecord],
    pred_records: Sequence[BoxRecord],
    iou_threshold: float = IOU_THRESHOLD,
) -> DetectionMetrics:
    true_positives, false_positives = _match_counts(
        gt_records, pred_records, iou_threshold
    )
    false_negatives = max(0, len(gt_records) - true_positives)
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    elif gt_records:
        precision = 0.0
    else:
        precision = None
    recall = true_positives / len(gt_records) if gt_records else None
    return DetectionMetrics(
        gt_boxes=len(gt_records),
        predicted_boxes=len(pred_records),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        ap50=_ap50(gt_records, pred_records, iou_threshold),
    )


def _metrics_row(
    *,
    method: str,
    class_id: int | str,
    class_name: str,
    metrics: DetectionMetrics,
    sequence_name: str | None = None,
    runtime_seconds: float | None = None,
    fps: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": method,
        "class_id": class_id,
        "class_name": class_name,
        "iou_threshold": IOU_THRESHOLD,
        "gt_boxes": metrics.gt_boxes,
        "predicted_boxes": metrics.predicted_boxes,
        "true_positives": metrics.true_positives,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
        "precision": _number(metrics.precision),
        "recall": _number(metrics.recall),
        "AP50": _number(metrics.ap50),
        "ap50_definition": AP50_DEFINITION,
        "runtime_seconds": _number(runtime_seconds),
        "FPS": _number(fps),
    }
    if sequence_name is not None:
        row = {"method": method, "sequence_name": sequence_name, **row}
    return row


def _load_ground_truth(
    dataset_root: Path,
    sequence_name: str,
    max_frames: int | None,
) -> list[BoxRecord]:
    num_frames = len(VisDroneSequence(dataset_root, sequence_name, max_frames=max_frames))
    frames = _read_mot_file(
        dataset_root / "annotations" / f"{sequence_name}.txt",
        kind="ground truth",
        max_frame=num_frames,
    )
    records: list[BoxRecord] = []
    for frame_id, objects in frames.items():
        for box_xywh, cls in zip(objects.boxes, objects.classes, strict=False):
            cls_int = int(cls)
            if cls_int not in ALLOWED_CLASSES:
                continue
            records.append(
                BoxRecord(
                    sequence_name=sequence_name,
                    frame_id=frame_id,
                    cls=cls_int,
                    xyxy=xywh_to_xyxy(np.asarray(box_xywh, dtype=float)),
                )
            )
    return records


def _run_detector_on_sequence(
    *,
    detector: Any,
    dataset_root: Path,
    sequence_name: str,
    max_frames: int | None,
) -> tuple[list[BoxRecord], float, int]:
    records: list[BoxRecord] = []
    sequence = VisDroneSequence(dataset_root, sequence_name, max_frames=max_frames)
    processed_frames = 0
    runtime_seconds = 0.0

    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError(
            "tqdm is required for detection evaluation. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    for frame_id, _frame_path, frame_bgr in tqdm(
        sequence, total=len(sequence), desc=f"detect:{sequence_name}"
    ):
        started = time.perf_counter()
        detections = detector.predict(frame_bgr, frame_id)
        runtime_seconds += time.perf_counter() - started
        processed_frames += 1
        for detection in detections:
            cls = int(detection.cls)
            if cls not in ALLOWED_CLASSES:
                continue
            records.append(
                BoxRecord(
                    sequence_name=sequence_name,
                    frame_id=frame_id,
                    cls=cls,
                    xyxy=np.asarray(detection.xyxy, dtype=float),
                    conf=float(detection.conf),
                )
            )

    return records, runtime_seconds, processed_frames


def _select(
    records: Sequence[BoxRecord],
    *,
    sequence_name: str | None = None,
    class_id: int | None = None,
) -> list[BoxRecord]:
    return [
        record
        for record in records
        if (sequence_name is None or record.sequence_name == sequence_name)
        and (class_id is None or record.cls == class_id)
    ]


def evaluate_detection(
    *,
    dataset_root: str | Path,
    config_paths: Sequence[str | Path],
    sequences: Sequence[str] | None,
    output_dir: str | Path,
    max_frames: int | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    dataset_path = Path(dataset_root).resolve()
    output_path = Path(output_dir)
    sequence_names = list(sequences) if sequences else list_sequence_names(dataset_path)
    if not sequence_names:
        raise ValueError("At least one sequence is required")
    if len(set(sequence_names)) != len(sequence_names):
        raise ValueError("Sequence names must be unique")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")

    gt_by_sequence = {
        sequence_name: _load_ground_truth(dataset_path, sequence_name, max_frames)
        for sequence_name in sequence_names
    }
    all_gt_records = [
        record
        for sequence_records in gt_by_sequence.values()
        for record in sequence_records
    ]

    resolved_device = resolve_device(device)
    by_sequence_rows: list[dict[str, Any]] = []
    by_class_rows: list[dict[str, Any]] = []
    by_method_rows: list[dict[str, Any]] = []
    method_payloads: list[dict[str, Any]] = []

    for config_path in config_paths:
        source_path = Path(config_path).resolve()
        config = apply_device_to_detector_configs(load_config(source_path), resolved_device)
        method = str(config["name"])
        detector = build_detector(config["detector"])
        pred_by_sequence: dict[str, list[BoxRecord]] = {}
        runtime_by_sequence: dict[str, float] = {}
        frames_by_sequence: dict[str, int] = {}

        for sequence_name in sequence_names:
            predictions, runtime_seconds, processed_frames = _run_detector_on_sequence(
                detector=detector,
                dataset_root=dataset_path,
                sequence_name=sequence_name,
                max_frames=max_frames,
            )
            pred_by_sequence[sequence_name] = predictions
            runtime_by_sequence[sequence_name] = runtime_seconds
            frames_by_sequence[sequence_name] = processed_frames

            for class_id in sorted(CLASS_NAMES):
                metrics = evaluate_records(
                    _select(gt_by_sequence[sequence_name], class_id=class_id),
                    _select(predictions, class_id=class_id),
                )
                by_sequence_rows.append(
                    _metrics_row(
                        method=method,
                        sequence_name=sequence_name,
                        class_id=class_id,
                        class_name=CLASS_NAMES[class_id],
                        metrics=metrics,
                        runtime_seconds=runtime_seconds,
                        fps=(
                            processed_frames / runtime_seconds
                            if runtime_seconds > 0
                            else None
                        ),
                    )
                )
            all_sequence_metrics = evaluate_records(
                gt_by_sequence[sequence_name],
                predictions,
            )
            by_sequence_rows.append(
                _metrics_row(
                    method=method,
                    sequence_name=sequence_name,
                    class_id="all",
                    class_name="all",
                    metrics=all_sequence_metrics,
                    runtime_seconds=runtime_seconds,
                    fps=processed_frames / runtime_seconds if runtime_seconds > 0 else None,
                )
            )

        all_pred_records = [
            record
            for sequence_records in pred_by_sequence.values()
            for record in sequence_records
        ]
        runtime_total = sum(runtime_by_sequence.values())
        frames_total = sum(frames_by_sequence.values())
        class_metrics_by_id: dict[int, DetectionMetrics] = {}
        for class_id in sorted(CLASS_NAMES):
            metrics = evaluate_records(
                _select(all_gt_records, class_id=class_id),
                _select(all_pred_records, class_id=class_id),
            )
            class_metrics_by_id[class_id] = metrics
            by_class_rows.append(
                _metrics_row(
                    method=method,
                    class_id=class_id,
                    class_name=CLASS_NAMES[class_id],
                    metrics=metrics,
                    runtime_seconds=runtime_total,
                    fps=frames_total / runtime_total if runtime_total > 0 else None,
                )
            )

        method_metrics = evaluate_records(all_gt_records, all_pred_records)
        method_row = _metrics_row(
            method=method,
            class_id="all",
            class_name="all",
            metrics=method_metrics,
            runtime_seconds=runtime_total,
            fps=frames_total / runtime_total if runtime_total > 0 else None,
        )
        for class_id, class_name in CLASS_NAMES.items():
            metrics = class_metrics_by_id[class_id]
            prefix = class_name
            method_row[f"{prefix}_precision"] = _number(metrics.precision)
            method_row[f"{prefix}_recall"] = _number(metrics.recall)
            method_row[f"{prefix}_AP50"] = _number(metrics.ap50)
            method_row[f"{prefix}_gt_boxes"] = metrics.gt_boxes
            method_row[f"{prefix}_predicted_boxes"] = metrics.predicted_boxes
            method_row[f"{prefix}_false_positives"] = metrics.false_positives
            method_row[f"{prefix}_false_negatives"] = metrics.false_negatives
        by_method_rows.append(method_row)
        method_payloads.append(
            {
                "method": method,
                "source_config_path": str(source_path),
                "runtime_seconds": runtime_total,
                "frames": frames_total,
            }
        )

    output_path.mkdir(parents=True, exist_ok=True)
    _write_csv(output_path / "detection_summary_by_method.csv", by_method_rows)
    _write_csv(output_path / "detection_summary_by_sequence.csv", by_sequence_rows)
    _write_csv(output_path / "detection_summary_by_class.csv", by_class_rows)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_path),
        "config_paths": [str(Path(path).resolve()) for path in config_paths],
        "methods": method_payloads,
        "sequences": sequence_names,
        "classes": CLASS_NAMES,
        "iou_threshold": IOU_THRESHOLD,
        "ap50_definition": AP50_DEFINITION,
        "max_frames": max_frames,
        "device": device,
        "resolved_device": resolved_device,
        "summary_by_method": by_method_rows,
        "summary_by_sequence": by_sequence_rows,
        "summary_by_class": by_class_rows,
    }
    (output_path / "detection_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate detector outputs before tracking on VisDrone classes 1 and 4."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Sequence names. Defaults to all validation sequences.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda:0", "mps"],
        default="auto",
        help="Inference device. auto follows the project resolver.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_detection(
        dataset_root=args.dataset_root,
        config_paths=args.configs,
        sequences=args.sequences,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        device=args.device,
    )
    print("\nDetection-only evaluation summary")
    for row in payload["summary_by_method"]:
        print(
            f"  {row['method']}: precision={row['precision']}, "
            f"recall={row['recall']}, AP50={row['AP50']}, "
            f"FP={row['false_positives']}, FN={row['false_negatives']}"
        )
    print(f"  outputs: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
