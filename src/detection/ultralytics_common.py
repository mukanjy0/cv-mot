from __future__ import annotations

from typing import Any

import numpy as np

from src.core.bbox import as_xyxy_array
from src.core.device import resolve_device
from src.core.types import Detection


def parse_class_mapping(config: dict[str, Any]) -> tuple[dict[int, int], set[int]]:
    coco_to_visdrone = {
        int(k): int(v) for k, v in config.get("coco_to_visdrone", {0: 1, 2: 4}).items()
    }
    allowed_classes = set(int(cls) for cls in config.get("allowed_classes", [1, 4]))
    return coco_to_visdrone, allowed_classes


def detections_from_ultralytics_result(
    result: Any,
    frame_id: int,
    coco_to_visdrone: dict[int, int],
    allowed_classes: set[int],
) -> list[Detection]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxys = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    coco_classes = boxes.cls.detach().cpu().numpy().astype(int)

    detections: list[Detection] = []
    for xyxy, det_conf, coco_cls in zip(xyxys, confs, coco_classes, strict=False):
        visdrone_cls = coco_to_visdrone.get(int(coco_cls))
        if visdrone_cls is None or visdrone_cls not in allowed_classes:
            continue
        detections.append(
            Detection(
                frame_id=frame_id,
                xyxy=as_xyxy_array(xyxy),
                conf=float(det_conf),
                cls=int(visdrone_cls),
            )
        )
    return detections
