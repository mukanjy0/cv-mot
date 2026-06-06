from __future__ import annotations

from typing import Any

import numpy as np

from src.core.types import Detection, Track
from src.detection.ultralytics_common import resolve_device
from src.tracking.base import Tracker
from src.tracking.boxmot_adapter import (
    boxmot_output_to_tracks,
    detections_to_boxmot_array,
    import_boxmot_class,
    instantiate_boxmot_tracker,
    update_boxmot_tracker,
)


class BotSortTracker(Tracker):
    """BoT-SORT adapter backed by BoxMOT.

    ReID defaults to disabled so the initial smoke path does not depend on
    downloading appearance weights.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.25,
        track_low_thresh: float = 0.10,
        new_track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.80,
        proximity_thresh: float = 0.50,
        appearance_thresh: float = 0.25,
        with_reid: bool = False,
        gmc_method: str = "sparseOptFlow",
        fuse_score: bool = True,
        device: str = "auto",
        half: bool = False,
    ) -> None:
        tracker_cls = import_boxmot_class(("BotSort", "BoTSORT", "BOTSORT"))
        params = {
            "track_high_thresh": float(track_high_thresh),
            "track_low_thresh": float(track_low_thresh),
            "new_track_thresh": float(new_track_thresh),
            "track_buffer": int(track_buffer),
            "match_thresh": float(match_thresh),
            "proximity_thresh": float(proximity_thresh),
            "appearance_thresh": float(appearance_thresh),
            "with_reid": bool(with_reid),
            "gmc_method": gmc_method,
            "fuse_score": bool(fuse_score),
            "device": resolve_device(device),
            "half": bool(half),
        }
        self.tracker = instantiate_boxmot_tracker(tracker_cls, params)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BotSortTracker":
        return cls(
            track_high_thresh=float(config.get("track_high_thresh", 0.25)),
            track_low_thresh=float(config.get("track_low_thresh", 0.10)),
            new_track_thresh=float(config.get("new_track_thresh", 0.25)),
            track_buffer=int(config.get("track_buffer", 30)),
            match_thresh=float(config.get("match_thresh", 0.80)),
            proximity_thresh=float(config.get("proximity_thresh", 0.50)),
            appearance_thresh=float(config.get("appearance_thresh", 0.25)),
            with_reid=bool(config.get("with_reid", False)),
            gmc_method=str(config.get("gmc_method", "sparseOptFlow")),
            fuse_score=bool(config.get("fuse_score", True)),
            device=str(config.get("device", "auto")),
            half=bool(config.get("half", False)),
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
