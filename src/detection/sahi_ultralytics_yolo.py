from __future__ import annotations

from typing import Any

import numpy as np

from src.core.bbox import area_xyxy, as_xyxy_array
from src.core.device import resolve_device
from src.core.types import Detection
from src.detection.base import Detector
from src.detection.ultralytics_common import parse_class_mapping


class SahiUltralyticsYoloDetector(Detector):
    """Optional sliced-inference YOLO detector backed by SAHI."""

    def __init__(
        self,
        model_path: str,
        imgsz: int = 640,
        conf: float = 0.35,
        device: str = "auto",
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.20,
        overlap_width_ratio: float = 0.20,
        perform_standard_pred: bool = False,
        postprocess_type: str = "GREEDYNMM",
        postprocess_match_metric: str = "IOU",
        postprocess_match_threshold: float = 0.50,
        min_bbox_area: float = 0.0,
        coco_to_visdrone: dict[int, int] | None = None,
        allowed_classes: list[int] | None = None,
    ) -> None:
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
        except ImportError as exc:
            raise ImportError(
                "Method 4 requires the optional SAHI dependencies. Install them "
                "with: python -m pip install -r requirements-sahi.txt"
            ) from exc

        if slice_height <= 0 or slice_width <= 0:
            raise ValueError("SAHI slice dimensions must be positive")
        for label, overlap in (
            ("overlap_height_ratio", overlap_height_ratio),
            ("overlap_width_ratio", overlap_width_ratio),
        ):
            if not 0 <= overlap < 1:
                raise ValueError(f"{label} must be in [0, 1)")

        self.get_sliced_prediction = get_sliced_prediction
        self.slice_height = int(slice_height)
        self.slice_width = int(slice_width)
        self.overlap_height_ratio = float(overlap_height_ratio)
        self.overlap_width_ratio = float(overlap_width_ratio)
        self.perform_standard_pred = bool(perform_standard_pred)
        self.postprocess_type = postprocess_type
        self.postprocess_match_metric = postprocess_match_metric
        self.postprocess_match_threshold = float(postprocess_match_threshold)
        self.min_bbox_area = float(min_bbox_area)
        if self.min_bbox_area < 0:
            raise ValueError("min_bbox_area must be non-negative")

        mapping_config = {
            "coco_to_visdrone": coco_to_visdrone or {0: 1, 2: 4},
            "allowed_classes": allowed_classes or [1, 4],
        }
        self.coco_to_visdrone, self.allowed_classes = parse_class_mapping(mapping_config)

        self.model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=float(conf),
            device=resolve_device(device),
            image_size=int(imgsz),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SahiUltralyticsYoloDetector":
        return cls(
            model_path=config["model_path"],
            imgsz=int(config.get("imgsz", 640)),
            conf=float(config.get("conf", 0.35)),
            device=str(config.get("device", "auto")),
            slice_height=int(config.get("slice_height", 640)),
            slice_width=int(config.get("slice_width", 640)),
            overlap_height_ratio=float(config.get("overlap_height_ratio", 0.20)),
            overlap_width_ratio=float(config.get("overlap_width_ratio", 0.20)),
            perform_standard_pred=bool(config.get("perform_standard_pred", False)),
            postprocess_type=str(config.get("postprocess_type", "GREEDYNMM")),
            postprocess_match_metric=str(
                config.get("postprocess_match_metric", "IOU")
            ),
            postprocess_match_threshold=float(
                config.get("postprocess_match_threshold", 0.50)
            ),
            min_bbox_area=float(config.get("min_bbox_area", 0.0)),
            coco_to_visdrone=config.get("coco_to_visdrone"),
            allowed_classes=config.get("allowed_classes"),
        )

    def predict(self, frame_bgr: np.ndarray, frame_id: int) -> list[Detection]:
        frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        result = self.get_sliced_prediction(
            frame_rgb,
            self.model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            perform_standard_pred=self.perform_standard_pred,
            postprocess_type=self.postprocess_type,
            postprocess_match_metric=self.postprocess_match_metric,
            postprocess_match_threshold=self.postprocess_match_threshold,
            postprocess_class_agnostic=False,
            verbose=0,
            progress_bar=False,
        )

        detections: list[Detection] = []
        height, width = frame_bgr.shape[:2]
        for prediction in result.object_prediction_list:
            visdrone_cls = self.coco_to_visdrone.get(int(prediction.category.id))
            if visdrone_cls is None or visdrone_cls not in self.allowed_classes:
                continue
            xyxy = as_xyxy_array(prediction.bbox.to_xyxy())
            xyxy[[0, 2]] = np.clip(xyxy[[0, 2]], 0.0, float(width))
            xyxy[[1, 3]] = np.clip(xyxy[[1, 3]], 0.0, float(height))
            if area_xyxy(xyxy) < self.min_bbox_area:
                continue
            detections.append(
                Detection(
                    frame_id=frame_id,
                    xyxy=xyxy,
                    conf=float(prediction.score.value),
                    cls=int(visdrone_cls),
                )
            )
        return detections
