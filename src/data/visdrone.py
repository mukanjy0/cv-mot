from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_sequence_names(dataset_root: str | Path) -> list[str]:
    sequences_dir = Path(dataset_root) / "sequences"
    if not sequences_dir.exists():
        raise FileNotFoundError(
            f"Expected VisDrone sequences directory at: {sequences_dir}"
        )
    return sorted(path.name for path in sequences_dir.iterdir() if path.is_dir())


class VisDroneSequence:
    """Frame loader for one VisDrone2019-MOT sequence.

    Detection/tracking code should consume only frames from this loader. Ground-truth
    annotations intentionally stay out of this path and are reserved for evaluation.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        sequence_name: str,
        max_frames: int | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.sequence_name = sequence_name
        self.sequence_dir = self.dataset_root / "sequences" / sequence_name
        self.max_frames = max_frames

        if not self.sequence_dir.exists():
            available = []
            sequences_dir = self.dataset_root / "sequences"
            if sequences_dir.exists():
                available = sorted(p.name for p in sequences_dir.iterdir() if p.is_dir())[:10]
            hint = f" Available examples: {', '.join(available)}" if available else ""
            raise FileNotFoundError(
                f"Sequence folder does not exist: {self.sequence_dir}.{hint}"
            )

        self.frame_paths = self._load_frame_paths()
        if not self.frame_paths:
            raise ValueError(f"No image frames found in: {self.sequence_dir}")

        if max_frames is not None:
            if max_frames <= 0:
                raise ValueError("max_frames must be positive when provided")
            self.frame_paths = self.frame_paths[:max_frames]

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __iter__(self) -> Iterator[tuple[int, Path, np.ndarray]]:
        import cv2

        for frame_id, frame_path in enumerate(self.frame_paths, start=1):
            frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise ValueError(f"Failed to read frame: {frame_path}")
            yield frame_id, frame_path, frame_bgr

    def _load_frame_paths(self) -> list[Path]:
        return sorted(
            [
                path
                for path in self.sequence_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=lambda p: p.name,
        )
