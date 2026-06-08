from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.core.bbox import xywh_to_xyxy
from src.core.types import Track
from src.data.visdrone import VisDroneSequence
from src.evaluation.evaluate_mot import _empty_frame, _read_mot_file
from src.visualization.render_tracks_video import _load_tracks
from src.visualization.video import VideoSink


CLASS_NAMES = {
    1: "pedestrian",
    4: "car",
}
GT_COLOR = (60, 120, 255)
PRED_COLOR = (80, 210, 100)


def _draw_label(
    frame: np.ndarray,
    label: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x, y = origin
    text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    text_w, text_h = text_size
    y1 = max(0, y - text_h - baseline - 4)
    y2 = y1 + text_h + baseline + 4
    x2 = min(frame.shape[1] - 1, x + text_w + 6)
    cv2.rectangle(frame, (x, y1), (x2, y2), color, -1)
    cv2.putText(
        frame,
        label,
        (x + 3, y2 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        thickness=1,
        lineType=cv2.LINE_AA,
    )


def _draw_box(
    frame: np.ndarray,
    xyxy: np.ndarray,
    label: str,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    x1, y1, x2, y2 = xyxy.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=thickness)
    _draw_label(frame, label, (x1, y1), color)


def _ground_truth_tracks(
    dataset_root: str | Path,
    sequence_name: str,
    max_frames: int | None,
) -> dict[int, list[Track]]:
    num_frames = len(VisDroneSequence(dataset_root, sequence_name, max_frames=max_frames))
    objects_by_frame = _read_mot_file(
        Path(dataset_root) / "annotations" / f"{sequence_name}.txt",
        kind="ground truth",
        max_frame=num_frames,
    )
    tracks_by_frame: dict[int, list[Track]] = {}
    for frame_id in range(1, num_frames + 1):
        objects = objects_by_frame.get(frame_id, _empty_frame())
        tracks: list[Track] = []
        for object_id, box_xywh, cls in zip(
            objects.ids, objects.boxes, objects.classes, strict=False
        ):
            tracks.append(
                Track(
                    frame_id=frame_id,
                    track_id=int(object_id),
                    xyxy=xywh_to_xyxy(np.asarray(box_xywh, dtype=float)),
                    conf=1.0,
                    cls=int(cls),
                )
            )
        tracks_by_frame[frame_id] = tracks
    return tracks_by_frame


def render_debug_video(
    *,
    dataset_root: str | Path,
    sequence_name: str,
    tracks_path: str | Path,
    output_path: str | Path,
    method_name: str,
    max_frames: int | None = None,
    fps: float = 30.0,
    overwrite: bool = False,
) -> Path:
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Debug video already exists: {output}")

    gt_by_frame = _ground_truth_tracks(dataset_root, sequence_name, max_frames)
    pred_by_frame = _load_tracks(Path(tracks_path))
    sequence = VisDroneSequence(dataset_root, sequence_name, max_frames=max_frames)
    sink: VideoSink | None = None
    try:
        for frame_id, _frame_path, frame_bgr in tqdm(
            sequence, total=len(sequence), desc=f"debug:{sequence_name}"
        ):
            frame = frame_bgr.copy()
            for gt in gt_by_frame.get(frame_id, []):
                class_name = CLASS_NAMES.get(gt.cls, f"class {gt.cls}")
                _draw_box(
                    frame,
                    gt.xyxy,
                    f"GT {class_name} #{gt.track_id}",
                    GT_COLOR,
                    thickness=2,
                )
            for pred in pred_by_frame.get(frame_id, []):
                class_name = CLASS_NAMES.get(pred.cls, f"class {pred.cls}")
                _draw_box(
                    frame,
                    pred.xyxy,
                    f"P {class_name} #{pred.track_id}",
                    PRED_COLOR,
                    thickness=2,
                )

            overlay = f"{method_name} | {sequence_name} | frame {frame_id}"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(
                frame,
                overlay,
                (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            if sink is None:
                height, width = frame.shape[:2]
                sink = VideoSink(output, (width, height), fps=fps)
            sink.write(frame)
    finally:
        if sink is not None:
            sink.close()

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render GT-vs-predicted MOT debug MP4 from existing tracks.txt."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--tracks-path", required=True)
    parser.add_argument("--sequence-name", required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = render_debug_video(
        dataset_root=args.dataset_root,
        sequence_name=args.sequence_name,
        tracks_path=args.tracks_path,
        output_path=args.output_path,
        method_name=args.method_name,
        max_frames=args.max_frames,
        fps=args.fps,
        overwrite=args.overwrite,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
