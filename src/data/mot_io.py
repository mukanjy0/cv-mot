from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from src.core.bbox import xyxy_to_xywh
from src.core.types import Track


def write_tracks_mot(path: str | Path, tracks_by_frame: Mapping[int, Sequence[Track]]) -> Path:
    """Write MOT Challenge-style predictions.

    Output columns:
    frame,id,bb_left,bb_top,bb_width,bb_height,conf,class,vis
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        for frame_id in sorted(tracks_by_frame):
            for track in sorted(tracks_by_frame[frame_id], key=lambda t: t.track_id):
                x, y, w, h = xyxy_to_xywh(track.xyxy)
                f.write(
                    f"{int(frame_id)},{int(track.track_id)},"
                    f"{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                    f"{float(track.conf):.4f},{int(track.cls)},-1\n"
                )

    return output_path
