from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoSink:
    """Small OpenCV MP4 writer that never opens GUI windows."""

    def __init__(
        self,
        path: str | Path,
        frame_size: tuple[int, int],
        fps: float = 30.0,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        width, height = frame_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.path), fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.path}")

    def write(self, frame_bgr: np.ndarray) -> None:
        self.writer.write(frame_bgr)

    def close(self) -> None:
        self.writer.release()

    def __enter__(self) -> "VideoSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()
