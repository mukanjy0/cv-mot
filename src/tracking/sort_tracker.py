from __future__ import annotations

from typing import Any

import numpy as np

from src.core.bbox import iou_matrix_xyxy
from src.core.types import Detection, Track
from src.tracking.base import Tracker


def _convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert xyxy box to SORT Kalman measurement [cx, cy, scale, ratio]."""

    x1, y1, x2, y2 = np.asarray(bbox, dtype=float).reshape(4)
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    scale = w * h
    ratio = w / max(h, 1e-6)
    return np.array([cx, cy, scale, ratio], dtype=float).reshape((4, 1))


def _convert_x_to_bbox(x: np.ndarray) -> np.ndarray:
    """Convert SORT Kalman state [cx, cy, scale, ratio, ...] to xyxy box."""

    cx, cy, scale, ratio = x[:4].reshape(-1)
    scale = max(float(scale), 0.0)
    ratio = max(float(ratio), 1e-6)
    w = np.sqrt(scale * ratio)
    h = scale / max(w, 1e-6)
    return np.array(
        [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
        dtype=float,
    )


def _associate_detections_to_trackers(
    detections: np.ndarray,
    trackers: np.ndarray,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections), dtype=int),
            np.empty((0,), dtype=int),
        )
    if len(detections) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.empty((0,), dtype=int),
            np.arange(len(trackers), dtype=int),
        )

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError(
            "scipy is required for SORT assignment. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    iou_matrix = iou_matrix_xyxy(detections, trackers)
    det_indices, trk_indices = linear_sum_assignment(-iou_matrix)
    candidate_matches = np.stack([det_indices, trk_indices], axis=1)

    unmatched_detections = []
    for det_idx in range(len(detections)):
        if det_idx not in candidate_matches[:, 0]:
            unmatched_detections.append(det_idx)

    unmatched_trackers = []
    for trk_idx in range(len(trackers)):
        if trk_idx not in candidate_matches[:, 1]:
            unmatched_trackers.append(trk_idx)

    matches = []
    for det_idx, trk_idx in candidate_matches:
        if iou_matrix[det_idx, trk_idx] < iou_threshold:
            unmatched_detections.append(int(det_idx))
            unmatched_trackers.append(int(trk_idx))
        else:
            matches.append([int(det_idx), int(trk_idx)])

    return (
        np.asarray(matches, dtype=int).reshape(-1, 2),
        np.asarray(unmatched_detections, dtype=int),
        np.asarray(unmatched_trackers, dtype=int),
    )


class _KalmanBoxTracker:
    count = 0

    def __init__(self, detection: Detection) -> None:
        try:
            from filterpy.kalman import KalmanFilter
        except ImportError as exc:
            raise ImportError(
                "filterpy is required for SORT Kalman filtering. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=float,
        )
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = _convert_bbox_to_z(detection.xyxy)

        self.time_since_update = 0
        self.id = _KalmanBoxTracker.count
        _KalmanBoxTracker.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.conf = float(detection.conf)
        self.cls = int(detection.cls)

    def update(self, detection: Detection) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.conf = float(detection.conf)
        self.cls = int(detection.cls)
        self.kf.update(_convert_bbox_to_z(detection.xyxy))

    def predict(self) -> np.ndarray:
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def get_state(self) -> np.ndarray:
        return _convert_x_to_bbox(self.kf.x)


class SortTracker(Tracker):
    """Small self-contained SORT wrapper.

    It accepts explicit detector outputs and returns explicit Track objects, keeping
    detector and tracker responsibilities decoupled for future methods.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.trackers: list[_KalmanBoxTracker] = []
        self.frame_count = 0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SortTracker":
        return cls(
            max_age=int(config.get("max_age", 30)),
            min_hits=int(config.get("min_hits", 3)),
            iou_threshold=float(config.get("iou_threshold", 0.3)),
        )

    def update(
        self,
        detections: list[Detection],
        frame_bgr: np.ndarray,
        frame_id: int,
    ) -> list[Track]:
        del frame_bgr
        self.frame_count += 1

        det_boxes = (
            np.asarray([det.xyxy for det in detections], dtype=float)
            if detections
            else np.empty((0, 4), dtype=float)
        )

        predicted_boxes = []
        invalid_tracker_indices = []
        for idx, tracker in enumerate(self.trackers):
            predicted = tracker.predict()
            if np.any(np.isnan(predicted)):
                invalid_tracker_indices.append(idx)
            else:
                predicted_boxes.append(predicted)

        for idx in reversed(invalid_tracker_indices):
            self.trackers.pop(idx)

        tracker_boxes = (
            np.asarray(predicted_boxes, dtype=float)
            if predicted_boxes
            else np.empty((0, 4), dtype=float)
        )
        matches, unmatched_dets, _unmatched_trks = _associate_detections_to_trackers(
            det_boxes, tracker_boxes, self.iou_threshold
        )

        for det_idx, trk_idx in matches:
            self.trackers[int(trk_idx)].update(detections[int(det_idx)])

        for det_idx in unmatched_dets:
            self.trackers.append(_KalmanBoxTracker(detections[int(det_idx)]))

        tracks: list[Track] = []
        for tracker in reversed(self.trackers):
            if tracker.time_since_update < 1 and (
                tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                tracks.append(
                    Track(
                        frame_id=frame_id,
                        track_id=tracker.id + 1,
                        xyxy=tracker.get_state(),
                        conf=tracker.conf,
                        cls=tracker.cls,
                    )
                )

            if tracker.time_since_update > self.max_age:
                self.trackers.remove(tracker)

        return sorted(tracks, key=lambda t: t.track_id)
