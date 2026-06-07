from __future__ import annotations

from typing import Any

import numpy as np

from src.core.types import Detection, Track
from src.tracking.base import Tracker
from src.tracking.boxmot_adapter import (
    boxmot_output_to_tracks,
    detections_to_boxmot_array,
    update_boxmot_tracker,
)


def _import_bytetrack() -> type[Any]:
    try:
        from boxmot.trackers.bbox.bytetrack.bytetrack import ByteTrack
    except ImportError as exc:
        raise ImportError(
            "Method 2 requires BoxMOT's ByteTrack implementation at "
            "`boxmot.trackers.bbox.bytetrack.bytetrack.ByteTrack`. Install "
            "the optional dependencies with "
            "`python -m pip install -r requirements-trackers.txt`."
        ) from exc
    return ByteTrack


class ByteTrackTracker(Tracker):
    """ByteTrack adapter that consumes Detection objects and calls no detector."""

    def __init__(
        self,
        min_conf: float = 0.10,
        track_thresh: float = 0.25,
        match_thresh: float = 0.80,
        track_buffer: int = 30,
        frame_rate: int = 30,
    ) -> None:
        tracker_cls = _import_bytetrack()
        self.tracker = tracker_cls(
            min_conf=float(min_conf),
            track_thresh=float(track_thresh),
            match_thresh=float(match_thresh),
            track_buffer=int(track_buffer),
            frame_rate=int(frame_rate),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ByteTrackTracker":
        return cls(
            min_conf=float(config.get("min_conf", 0.10)),
            track_thresh=float(config.get("track_thresh", 0.25)),
            match_thresh=float(config.get("match_thresh", 0.80)),
            track_buffer=int(config.get("track_buffer", 30)),
            frame_rate=int(config.get("frame_rate", 30)),
        )

    def update(
        self,
        detections: list[Detection],
        frame_bgr: np.ndarray,
        frame_id: int,
    ) -> list[Track]:
        dets = detections_to_boxmot_array(detections)
        outputs = update_boxmot_tracker(self.tracker, dets, frame_bgr)
        return boxmot_output_to_tracks(outputs, detections, frame_id)
