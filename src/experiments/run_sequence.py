from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from src.core.config import load_config
from src.core.device import apply_device_to_detector_configs, resolve_device
from src.core.types import Track
from src.data.mot_io import write_tracks_mot
from src.data.visdrone import VisDroneSequence, list_sequence_names
from src.detection.base import Detector
from src.detection.sahi_ultralytics_yolo import SahiUltralyticsYoloDetector
from src.detection.ultralytics_rtdetr import UltralyticsRtDetrDetector
from src.detection.ultralytics_yolo import UltralyticsYoloDetector
from src.tracking.base import Tracker
from src.tracking.botsort_tracker import BotSortTracker
from src.tracking.bytetrack_tracker import ByteTrackTracker
from src.tracking.sort_tracker import SortTracker


def build_detector(config: dict[str, Any]) -> Detector:
    detector_type = config.get("type")
    if detector_type == "ultralytics_yolo":
        return UltralyticsYoloDetector.from_config(config)
    if detector_type == "sahi_ultralytics_yolo":
        return SahiUltralyticsYoloDetector.from_config(config)
    if detector_type == "ultralytics_rtdetr":
        return UltralyticsRtDetrDetector.from_config(config)
    raise ValueError(f"Unsupported detector type: {detector_type}")


def build_tracker(config: dict[str, Any]) -> Tracker:
    tracker_type = config.get("type")
    if tracker_type == "sort":
        return SortTracker.from_config(config)
    if tracker_type == "bytetrack":
        return ByteTrackTracker.from_config(config)
    if tracker_type == "botsort":
        return BotSortTracker.from_config(config)
    raise ValueError(f"Unsupported tracker type: {tracker_type}")


def run_sequence(
    config_path: str | Path,
    dataset_root: str | Path,
    sequence_name: str,
    max_frames: int | None = None,
    save_video: bool = False,
    video_fps: float = 30.0,
    tracks_path: str | Path | None = None,
    video_path: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    if device is not None:
        config = apply_device_to_detector_configs(config, resolve_device(device))
    experiment_name = config["name"]

    sequence = VisDroneSequence(dataset_root, sequence_name, max_frames=max_frames)
    detector = build_detector(config["detector"])
    tracker = build_tracker(config["tracker"])

    output_config = config.get("output", {})
    resolved_tracks_path = (
        Path(tracks_path)
        if tracks_path is not None
        else (
            Path(output_config.get("tracks_dir", "outputs/tracks"))
            / experiment_name
            / f"{sequence_name}.txt"
        )
    )
    resolved_video_path = (
        Path(video_path)
        if video_path is not None
        else (
            Path(output_config.get("videos_dir", "outputs/videos"))
            / experiment_name
            / f"{sequence_name}.mp4"
        )
    )

    tracks_by_frame: dict[int, list[Track]] = {}
    video_sink = None
    processing_seconds = 0.0

    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError(
            "tqdm is required for progress reporting. "
            "Install dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    if save_video:
        from src.visualization.draw import draw_tracks
        from src.visualization.video import VideoSink

    try:
        for frame_id, _frame_path, frame_bgr in tqdm(
            sequence, total=len(sequence), desc=sequence_name
        ):
            t0 = time.perf_counter()
            detections = detector.predict(frame_bgr, frame_id)
            tracks = tracker.update(detections, frame_bgr, frame_id)
            processing_seconds += time.perf_counter() - t0

            tracks_by_frame[frame_id] = tracks

            if save_video:
                if video_sink is None:
                    height, width = frame_bgr.shape[:2]
                    video_sink = VideoSink(
                        resolved_video_path, (width, height), fps=video_fps
                    )
                video_sink.write(draw_tracks(frame_bgr, tracks))
    finally:
        if video_sink is not None:
            video_sink.close()

    write_tracks_mot(resolved_tracks_path, tracks_by_frame)

    unique_track_ids = {
        track.track_id for frame_tracks in tracks_by_frame.values() for track in frame_tracks
    }
    frame_count = len(tracks_by_frame)
    fps = frame_count / processing_seconds if processing_seconds > 0 else 0.0

    return {
        "sequence_name": sequence_name,
        "frames_processed": frame_count,
        "tracks_produced": len(unique_track_ids),
        "track_rows": sum(len(v) for v in tracks_by_frame.values()),
        "runtime_seconds": processing_seconds,
        "fps": fps,
        "tracks_path": resolved_tracks_path,
        "video_path": resolved_video_path if save_video else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run detector + tracker on one VisDrone2019-MOT sequence."
    )
    parser.add_argument(
        "--config",
        help="Path to YAML experiment config. Required unless --list-sequences is used.",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to VisDrone2019-MOT-val containing sequences/ and annotations/.",
    )
    parser.add_argument("--sequence-name", help="Sequence folder name to process.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional first N frames for smoke testing.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Write an MP4 visualization under the configured videos directory.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=30.0,
        help="FPS for optional visualization video.",
    )
    parser.add_argument(
        "--list-sequences",
        action="store_true",
        help="List available sequence names and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_sequences:
        for name in list_sequence_names(args.dataset_root):
            print(name)
        return

    if not args.sequence_name:
        raise SystemExit("--sequence-name is required unless --list-sequences is used")
    if not args.config:
        raise SystemExit("--config is required unless --list-sequences is used")

    summary = run_sequence(
        config_path=args.config,
        dataset_root=args.dataset_root,
        sequence_name=args.sequence_name,
        max_frames=args.max_frames,
        save_video=args.save_video,
        video_fps=args.video_fps,
    )

    print("\nMOT run summary")
    print(f"  sequence: {summary['sequence_name']}")
    print(f"  frames processed: {summary['frames_processed']}")
    print(f"  unique tracks produced: {summary['tracks_produced']}")
    print(f"  MOT rows written: {summary['track_rows']}")
    print(f"  detection+tracking seconds: {summary['runtime_seconds']:.2f}")
    print(f"  detection+tracking FPS: {summary['fps']:.2f}")
    print(f"  tracks: {summary['tracks_path']}")
    if summary["video_path"] is not None:
        print(f"  video: {summary['video_path']}")


if __name__ == "__main__":
    main()
