from __future__ import annotations

import cv2
import numpy as np

from src.core.types import Track


VISDRONE_CLASS_NAMES = {
    1: "pedestrian",
    4: "car",
}


def _track_color(track_id: int, cls: int) -> tuple[int, int, int]:
    base = {
        1: np.array([38, 132, 255], dtype=int),
        4: np.array([88, 196, 112], dtype=int),
    }.get(cls, np.array([220, 220, 220], dtype=int))
    jitter = np.array(
        [(track_id * 37) % 55, (track_id * 17) % 55, (track_id * 29) % 55],
        dtype=int,
    )
    color = np.minimum(base + jitter, 255)
    return int(color[0]), int(color[1]), int(color[2])


def draw_tracks(frame_bgr: np.ndarray, tracks: list[Track]) -> np.ndarray:
    frame = frame_bgr.copy()
    for track in tracks:
        x1, y1, x2, y2 = track.xyxy.astype(int)
        color = _track_color(track.track_id, track.cls)
        label_name = VISDRONE_CLASS_NAMES.get(track.cls, f"class {track.cls}")
        label = f"{label_name} #{track.track_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

        text_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        text_w, text_h = text_size
        label_y1 = max(0, y1 - text_h - baseline - 4)
        label_y2 = label_y1 + text_h + baseline + 4
        label_x2 = min(frame.shape[1] - 1, x1 + text_w + 6)

        cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 3, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return frame
