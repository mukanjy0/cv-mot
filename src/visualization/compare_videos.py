from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _open_capture(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def _draw_label(frame: np.ndarray, label: str) -> np.ndarray:
    output = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_w, text_h = text_size
    cv2.rectangle(output, (0, 0), (text_w + 18, text_h + baseline + 18), (0, 0, 0), -1)
    cv2.putText(
        output,
        label,
        (9, text_h + 9),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return output


def create_comparison_video(
    video_paths: list[Path],
    labels: list[str],
    output_path: Path,
) -> Path:
    if len(video_paths) < 2:
        raise ValueError("Provide at least two videos to compare.")
    if len(labels) != len(video_paths):
        raise ValueError("Number of labels must match number of videos.")

    caps = [_open_capture(path) for path in video_paths]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
        widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in caps]
        heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps]
        target_height = min(height for height in heights if height > 0)
        target_widths = [
            max(1, int(width * target_height / height))
            for width, height in zip(widths, heights, strict=False)
        ]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (sum(target_widths), target_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open output video writer: {output_path}")

        try:
            while True:
                frames = []
                for cap, label, target_width in zip(caps, labels, target_widths, strict=False):
                    ok, frame = cap.read()
                    if not ok:
                        return output_path
                    resized = cv2.resize(frame, (target_width, target_height))
                    frames.append(_draw_label(resized, label))
                writer.write(np.hstack(frames))
        finally:
            writer.release()
    finally:
        for cap in caps:
            cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a side-by-side MP4 from existing method visualization videos."
    )
    parser.add_argument("--videos", nargs="+", required=True, help="Input MP4 paths.")
    parser.add_argument("--labels", nargs="+", required=True, help="Labels for each video.")
    parser.add_argument("--output", required=True, help="Output comparison MP4 path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = create_comparison_video(
        video_paths=[Path(path) for path in args.videos],
        labels=list(args.labels),
        output_path=Path(args.output),
    )
    print(f"Wrote comparison video: {output_path}")


if __name__ == "__main__":
    main()
