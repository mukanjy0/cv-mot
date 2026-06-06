from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from src.core.bbox import iou_matrix_xyxy
from src.core.types import Detection, Track


def detections_to_boxmot_array(detections: list[Detection]) -> np.ndarray:
    if not detections:
        return np.empty((0, 6), dtype=float)

    rows = [
        [
            float(det.xyxy[0]),
            float(det.xyxy[1]),
            float(det.xyxy[2]),
            float(det.xyxy[3]),
            float(det.conf),
            float(det.cls),
        ]
        for det in detections
    ]
    return np.asarray(rows, dtype=float)


def import_boxmot_class(class_names: tuple[str, ...]) -> type[Any]:
    try:
        import boxmot
    except ImportError as exc:
        raise ImportError(
            "BoxMOT is required for this tracker. Install it with "
            "`pip install boxmot`, or install all project dependencies with "
            "`pip install -r requirements.txt`. Method 1 does not require BoxMOT."
        ) from exc

    for class_name in class_names:
        tracker_cls = getattr(boxmot, class_name, None)
        if tracker_cls is not None:
            return tracker_cls

    available = ", ".join(name for name in dir(boxmot) if "track" in name.lower())
    raise ImportError(
        f"BoxMOT is installed but none of these tracker classes were found: "
        f"{', '.join(class_names)}. Available tracker-like names: {available}"
    )


def instantiate_boxmot_tracker(tracker_cls: type[Any], params: dict[str, Any]) -> Any:
    signature = inspect.signature(tracker_cls)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if accepts_kwargs:
        kwargs = params
    else:
        kwargs = {
            key: value for key, value in params.items() if key in signature.parameters
        }

    try:
        return tracker_cls(**kwargs)
    except TypeError as exc:
        raise TypeError(
            f"Failed to initialize BoxMOT tracker {tracker_cls.__name__} with "
            f"parameters {kwargs}. Check the installed BoxMOT version and tracker "
            "constructor signature."
        ) from exc


def update_boxmot_tracker(
    tracker: Any,
    dets: np.ndarray,
    frame_bgr: np.ndarray,
) -> np.ndarray:
    try:
        output = tracker.update(dets, frame_bgr)
    except TypeError as exc:
        raise TypeError(
            "BoxMOT tracker.update did not accept (detections, frame_bgr). "
            "This project expects the BoxMOT API where detections are an Nx6 "
            "array [x1,y1,x2,y2,conf,cls] and the image is passed separately."
        ) from exc

    if output is None:
        return np.empty((0, 0), dtype=float)
    if hasattr(output, "detach"):
        output = output.detach().cpu().numpy()
    output_array = np.asarray(output, dtype=float)
    if output_array.size == 0:
        return np.empty((0, 0), dtype=float)
    if output_array.ndim == 1:
        output_array = output_array.reshape(1, -1)
    return output_array


def boxmot_output_to_tracks(
    outputs: np.ndarray,
    detections: list[Detection],
    frame_id: int,
) -> list[Track]:
    if outputs.size == 0:
        return []

    tracks: list[Track] = []
    for row in outputs:
        if len(row) < 5:
            continue

        xyxy = np.asarray(row[:4], dtype=float)
        track_id = int(row[4])
        conf = float(row[5]) if len(row) > 5 else 1.0
        cls = int(row[6]) if len(row) > 6 else _nearest_detection_cls(xyxy, detections)

        tracks.append(
            Track(
                frame_id=frame_id,
                track_id=track_id,
                xyxy=xyxy,
                conf=conf,
                cls=cls,
            )
        )

    return sorted(tracks, key=lambda t: t.track_id)


def _nearest_detection_cls(xyxy: np.ndarray, detections: list[Detection]) -> int:
    if not detections:
        return -1
    det_boxes = np.asarray([det.xyxy for det in detections], dtype=float)
    ious = iou_matrix_xyxy(xyxy.reshape(1, 4), det_boxes).reshape(-1)
    best_idx = int(np.argmax(ious))
    return int(detections[best_idx].cls)
