# VisDrone Multi-Object Tracking

Modular Computer Vision assignment code for Multi-Object Tracking on
`VisDrone2019-MOT-val`.

The pipeline is intentionally split into separate stages:

1. Load frames from a VisDrone sequence.
2. Run a detector with `model.predict()`.
3. Filter and map detections into VisDrone classes.
4. Pass detections to a tracker.
5. Write MOT-format prediction rows.
6. Optionally write an MP4 visualization.

This first implementation smoke-tests Method 1: YOLOv11 + SORT. It does not
implement SAHI or motmetrics yet. Method 2 and Method 3 configs are included
for YOLO26 + ByteTrack and RT-DETR + BoT-SORT visual testing.

## Dataset Layout

Pass `--dataset-root` as the folder containing this structure:

```text
VisDrone2019-MOT-val/
  sequences/
    <sequence_name>/
      0000001.jpg
      0000002.jpg
      ...
  annotations/
    <sequence_name>.txt
  det/
    ...
```

The `det/` folder is ignored. Ground-truth annotations are not used during
detection or tracking; they are reserved for a later evaluation step.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For CUDA, install a PyTorch build that matches your GPU before or alongside
Ultralytics. CPU runs work too, just slower.

## Smoke Test

List available sequences:

```bash
python -m src.experiments.run_sequence \
  --dataset-root VisDrone2019-MOT-val \
  --list-sequences
```

Run the first 100 frames of one sequence:

```bash
python -m src.experiments.run_sequence \
  --config configs/m1_yolo11_sort_smoke.yaml \
  --dataset-root VisDrone2019-MOT-val \
  --sequence-name uav0000137_00458_v \
  --max-frames 100 \
  --save-video
```

On first use, Ultralytics may download model weights such as `yolo11n.pt`,
`yolo26n.pt`, or `rtdetr-l.pt` unless the files already exist locally.

Additional visual testing commands for Method 1, Method 2, Method 3, longer
runs, full sequences, and side-by-side videos are in
[`docs/testing_commands.md`](docs/testing_commands.md).

## Classes

Only these VisDrone classes are kept:

- VisDrone class `1`: pedestrian
- VisDrone class `4`: car

The default COCO-to-VisDrone mapping is explicit in
`configs/m1_yolo11_sort_smoke.yaml`:

```yaml
coco_to_visdrone:
  0: 1
  2: 4
allowed_classes: [1, 4]
```

## Output Format

Predictions are written to:

```text
outputs/tracks/<experiment_name>/<sequence_name>.txt
```

Rows use MOT format:

```text
frame,id,bb_left,bb_top,bb_width,bb_height,conf,class,vis
```

Frame IDs start at `1`. Internally boxes are `xyxy`; output boxes are converted
to `xywh`. Prediction visibility is always `-1`.

Visualization videos are written to:

```text
outputs/videos/<experiment_name>/<sequence_name>.mp4
```

## Troubleshooting

- If `ultralytics` is missing, run `pip install -r requirements.txt`.
- If `boxmot` is missing, Method 1 still works, but Method 2 and Method 3 need
  `pip install boxmot` or `pip install -r requirements.txt`.
- If model weights cannot download, place a local model file somewhere and set
  `detector.model_path` in the YAML config to that path.
- If GPU execution fails, set `detector.device: cpu` and `half: false`.
- If OpenCV cannot write MP4, try installing a full OpenCV build or run without
  `--save-video` to still produce MOT text predictions.
