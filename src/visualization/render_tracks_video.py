from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.core.types import Track
from src.data.visdrone import VisDroneSequence
from src.visualization.draw import draw_tracks
from src.visualization.video import VideoSink


def _load_tracks(path: Path) -> dict[int, list[Track]]:
    tracks_by_frame: dict[int, list[Track]] = defaultdict(list)
    if not path.is_file():
        raise FileNotFoundError(f"Tracks file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(",")
            if len(parts) < 8:
                raise ValueError(
                    f"Expected at least 8 comma-separated fields in {path}:{line_number}"
                )

            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])
            cls = int(float(parts[7]))
            tracks_by_frame[frame_id].append(
                Track(
                    frame_id=frame_id,
                    track_id=track_id,
                    xyxy=np.array([x, y, x + w, y + h], dtype=float),
                    conf=conf,
                    cls=cls,
                )
            )

    return dict(tracks_by_frame)


def render_tracks_video(
    *,
    dataset_root: str | Path,
    sequence_name: str,
    tracks_path: str | Path,
    output_path: str | Path | None = None,
    fps: float = 30.0,
    overwrite: bool = False,
) -> Path:
    tracks_file = Path(tracks_path)
    video_path = Path(output_path) if output_path is not None else tracks_file.parent / "video.mp4"
    if video_path.exists() and not overwrite:
        raise FileExistsError(f"Video already exists: {video_path}")

    tracks_by_frame = _load_tracks(tracks_file)
    sequence = VisDroneSequence(dataset_root, sequence_name)
    sink: VideoSink | None = None
    try:
        for _, (frame_id, _, frame_bgr) in enumerate(
            tqdm(sequence, total=len(sequence), desc=sequence_name)
        ):
            if sink is None:
                height, width = frame_bgr.shape[:2]
                sink = VideoSink(video_path, (width, height), fps=fps)
            sink.write(draw_tracks(frame_bgr, tracks_by_frame.get(frame_id, [])))
    finally:
        if sink is not None:
            sink.close()

    return video_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render visualization MP4s from existing MOT tracks.txt files."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--tracks-root", required=True)
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Sequence names. Defaults to every child directory under --tracks-root.",
    )
    parser.add_argument("--tracks-name", default="tracks.txt")
    parser.add_argument("--output-name", default="video.mp4")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracks_root = Path(args.tracks_root)
    sequences = args.sequences or sorted(
        path.name for path in tracks_root.iterdir() if path.is_dir()
    )
    for sequence_name in sequences:
        tracks_path = tracks_root / sequence_name / args.tracks_name
        output_path = tracks_root / sequence_name / args.output_name
        video_path = render_tracks_video(
            dataset_root=args.dataset_root,
            sequence_name=sequence_name,
            tracks_path=tracks_path,
            output_path=output_path,
            fps=args.fps,
            overwrite=args.overwrite,
        )
        print(f"Wrote {video_path}")


if __name__ == "__main__":
    main()
