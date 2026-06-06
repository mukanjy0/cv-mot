from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.core.types import Detection


class Detector(ABC):
    @abstractmethod
    def predict(self, frame_bgr: np.ndarray, frame_id: int) -> list[Detection]:
        """Run object detection for one BGR frame."""
