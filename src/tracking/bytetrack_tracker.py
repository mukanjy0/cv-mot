from __future__ import annotations

from typing import Any

import numpy as np

from src.core.types import Detection, Track
from src.tracking.base import Tracker
from src.tracking.boxmot_adapter import (
    boxmot_output_to_tracks,
    detections_to_boxmot_array,
    import_boxmot_class,
    instantiate_boxmot_tracker,
    update_boxmot_tracker,
)


class ByteTrackTracker(Tracker):
    """ByteTrack adapter that consumes Detection objects and calls no detector."""

    def __init__(
        self,
        track_high_thresh: float = 0.25,
        track_low_thresh: float = 0.10,
        new_track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.80,
        fuse_score: bool = True,
    ) -> None:
        tracker_cls = import_boxmot_class(("ByteTrack", "BYTETracker", "ByteTracker"))
        params = {
            "track_high_thresh": float(track_high_thresh),
            "track_low_thresh": float(track_low_thresh),
            "new_track_thresh": float(new_track_thresh),
            "track_buffer": int(track_buffer),
            "match_thresh": float(match_thresh),
            "fuse_score": bool(fuse_score),
        }
        self.tracker = instantiate_boxmot_tracker(tracker_cls, params)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ByteTrackTracker":
        return cls(
            track_high_thresh=float(config.get("track_high_thresh", 0.25)),
            track_low_thresh=float(config.get("track_low_thresh", 0.10)),
            new_track_thresh=float(config.get("new_track_thresh", 0.25)),
            track_buffer=int(config.get("track_buffer", 30)),
            match_thresh=float(config.get("match_thresh", 0.80)),
            fuse_score=bool(config.get("fuse_score", True)),
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
