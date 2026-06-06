from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.core.types import Detection, Track


class Tracker(ABC):
    @abstractmethod
    def update(
        self,
        detections: list[Detection],
        frame_bgr: np.ndarray,
        frame_id: int,
    ) -> list[Track]:
        """Advance tracker state with detections from one frame."""
