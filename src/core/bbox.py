from __future__ import annotations

import numpy as np


def as_xyxy_array(box: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(box, dtype=float).reshape(4)
    x1, y1, x2, y2 = arr
    return np.array([x1, y1, x2, y2], dtype=float)


def xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    box = as_xyxy_array(xyxy)
    x1, y1, x2, y2 = box
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=float)


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    box = np.asarray(xywh, dtype=float).reshape(4)
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=float)


def area_xyxy(xyxy: np.ndarray) -> float:
    box = as_xyxy_array(xyxy)
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    a = as_xyxy_array(box_a)
    b = as_xyxy_array(box_b)

    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])

    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    union = area_xyxy(a) + area_xyxy(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def iou_matrix_xyxy(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Return IoU matrix with shape (len(boxes_a), len(boxes_b))."""

    boxes_a = np.asarray(boxes_a, dtype=float)
    boxes_b = np.asarray(boxes_b, dtype=float)
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=float)

    xx1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    yy1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    xx2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    yy2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = np.maximum(0.0, xx2 - xx1)
    inter_h = np.maximum(0.0, yy2 - yy1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(
        0.0, boxes_a[:, 3] - boxes_a[:, 1]
    )
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(
        0.0, boxes_b[:, 3] - boxes_b[:, 1]
    )
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
