from __future__ import annotations

from typing import Any

import numpy as np

from src.core.bbox import area_xyxy
from src.core.types import Detection
from src.detection.base import Detector


def _resolve_interpolation(value: str | int) -> int:
    if isinstance(value, int):
        return value

    import cv2

    name = value.strip()
    if not name.startswith("INTER_"):
        name = f"INTER_{name.upper()}"
    if not hasattr(cv2, name):
        raise ValueError(f"Unsupported OpenCV interpolation mode: {value!r}")
    return int(getattr(cv2, name))


class UpscaledDetector(Detector):
    """Resize a frame before detection and scale detections back afterward."""

    def __init__(
        self,
        detector: Detector,
        *,
        enabled: bool = True,
        scale: float = 1.5,
        interpolation: str | int = "INTER_LINEAR",
        max_size: int | None = None,
    ) -> None:
        if scale <= 0:
            raise ValueError("Upscale scale must be positive")
        if max_size is not None and max_size <= 0:
            raise ValueError("Upscale max_size must be positive when provided")

        self.detector = detector
        self.enabled = bool(enabled)
        self.scale = float(scale)
        self.interpolation = _resolve_interpolation(interpolation)
        self.max_size = int(max_size) if max_size is not None else None

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        detector: Detector,
    ) -> "UpscaledDetector":
        return cls(
            detector=detector,
            enabled=bool(config.get("enabled", True)),
            scale=float(config.get("scale", 1.5)),
            interpolation=config.get("interpolation", "INTER_LINEAR"),
            max_size=(
                int(config["max_size"])
                if config.get("max_size") is not None
                else None
            ),
        )

    def _effective_scale(self, width: int, height: int) -> float:
        effective = self.scale
        if self.max_size is not None:
            longest_scaled_edge = max(width, height) * effective
            if longest_scaled_edge > self.max_size:
                effective = self.max_size / max(width, height)
        return max(1.0, effective)

    def predict(self, frame_bgr: np.ndarray, frame_id: int) -> list[Detection]:
        if not self.enabled or self.scale == 1.0:
            return self.detector.predict(frame_bgr, frame_id)

        import cv2

        height, width = frame_bgr.shape[:2]
        effective_scale = self._effective_scale(width, height)
        if effective_scale == 1.0:
            return self.detector.predict(frame_bgr, frame_id)

        resized_width = int(round(width * effective_scale))
        resized_height = int(round(height * effective_scale))
        resized = cv2.resize(
            frame_bgr,
            (resized_width, resized_height),
            interpolation=self.interpolation,
        )

        detections = self.detector.predict(resized, frame_id)
        scaled: list[Detection] = []
        for detection in detections:
            xyxy = np.asarray(detection.xyxy, dtype=float) / effective_scale
            xyxy[[0, 2]] = np.clip(xyxy[[0, 2]], 0.0, float(width))
            xyxy[[1, 3]] = np.clip(xyxy[[1, 3]], 0.0, float(height))
            if area_xyxy(xyxy) <= 0:
                continue
            scaled.append(
                Detection(
                    frame_id=frame_id,
                    xyxy=xyxy,
                    conf=float(detection.conf),
                    cls=int(detection.cls),
                )
            )
        return scaled
