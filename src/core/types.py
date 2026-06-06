from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Detection:
    """Single-frame detector output in internal xyxy coordinates."""

    frame_id: int
    xyxy: np.ndarray
    conf: float
    cls: int


@dataclass(slots=True)
class Track:
    """Tracker output for one object in one frame."""

    frame_id: int
    track_id: int
    xyxy: np.ndarray
    conf: float
    cls: int
