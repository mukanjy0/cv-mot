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


def _import_botsort() -> type[Any]:
    try:
        from boxmot.trackers.bbox.botsort.botsort import BotSort
    except ImportError as exc:
        raise ImportError(
            "Method 3 requires BoxMOT's BotSort implementation at "
            "`boxmot.trackers.bbox.botsort.botsort.BotSort`. Install "
            "the optional dependencies with "
            "`python -m pip install -r requirements-trackers.txt`."
        ) from exc
    return BotSort


def _parse_cmc_method(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "off", "false", "disabled", "null"}:
        return None
    return text


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
        cmc_method: str | None = "sof",
        frame_rate: int = 30,
        fuse_first_associate: bool = True,
    ) -> None:
        tracker_cls = _import_botsort()
        self.tracker = tracker_cls(
            track_high_thresh=float(track_high_thresh),
            track_low_thresh=float(track_low_thresh),
            new_track_thresh=float(new_track_thresh),
            track_buffer=int(track_buffer),
            match_thresh=float(match_thresh),
            proximity_thresh=float(proximity_thresh),
            appearance_thresh=float(appearance_thresh),
            with_reid=bool(with_reid),
            cmc_method=_parse_cmc_method(cmc_method),
            frame_rate=int(frame_rate),
            fuse_first_associate=bool(fuse_first_associate),
        )

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
            cmc_method=_parse_cmc_method(config.get("cmc_method", "sof")),
            frame_rate=int(config.get("frame_rate", 30)),
            fuse_first_associate=bool(config.get("fuse_first_associate", True)),
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
