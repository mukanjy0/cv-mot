from __future__ import annotations

from typing import Any

import numpy as np

from src.core.device import resolve_device
from src.core.types import Detection
from src.detection.base import Detector
from src.detection.ultralytics_common import (
    detections_from_ultralytics_result,
    parse_class_mapping,
)


class UltralyticsYoloDetector(Detector):
    """YOLO detector wrapper that maps COCO classes into VisDrone classes."""

    def __init__(
        self,
        model_path: str,
        imgsz: int = 960,
        conf: float = 0.25,
        iou: float = 0.70,
        device: str = "auto",
        half: bool = False,
        coco_to_visdrone: dict[int, int] | None = None,
        allowed_classes: list[int] | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required for UltralyticsYoloDetector. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = resolve_device(device)
        self.half = half
        mapping_config = {
            "coco_to_visdrone": coco_to_visdrone or {0: 1, 2: 4},
            "allowed_classes": allowed_classes or [1, 4],
        }
        self.coco_to_visdrone, self.allowed_classes = parse_class_mapping(mapping_config)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "UltralyticsYoloDetector":
        return cls(
            model_path=config["model_path"],
            imgsz=int(config.get("imgsz", 960)),
            conf=float(config.get("conf", 0.25)),
            iou=float(config.get("iou", 0.70)),
            device=str(config.get("device", "auto")),
            half=bool(config.get("half", False)),
            coco_to_visdrone=config.get("coco_to_visdrone"),
            allowed_classes=config.get("allowed_classes"),
        )

    def predict(self, frame_bgr: np.ndarray, frame_id: int) -> list[Detection]:
        results = self.model.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        return detections_from_ultralytics_result(
            result, frame_id, self.coco_to_visdrone, self.allowed_classes
        )
