# Qualitative Testing Commands

These commands generate MOT text predictions and optional MP4 visualizations for
side-by-side qualitative review. The dataset root should point at the folder
containing `sequences/`, `annotations/`, and `det/`. The pipeline ignores `det/`
and does not use annotations during detection or tracking.

Replace `VisDrone2019-MOT-val` with your dataset path if needed, and replace
`<sequence_name>` with one of the listed sequence names.

## A. List Available Sequences

```bash
python -m src.experiments.run_sequence \
  --dataset-root /path/to/VisDrone2019-MOT-val \
  --list-sequences
```

PowerShell local example:

```powershell
python -m src.experiments.run_sequence `
  --dataset-root VisDrone2019-MOT-val `
  --list-sequences
```

## B. Run Method 1: YOLOv11 + SORT

```bash
python -m src.experiments.run_sequence \
  --config configs/m1_yolo11_sort_smoke.yaml \
  --dataset-root /path/to/VisDrone2019-MOT-val \
  --sequence-name <sequence_name> \
  --max-frames 100 \
  --save-video
```

## C. Run Method 2: YOLO26 + ByteTrack

```bash
python -m src.experiments.run_sequence \
  --config configs/m2_yolo26_bytetrack_smoke.yaml \
  --dataset-root /path/to/VisDrone2019-MOT-val \
  --sequence-name <sequence_name> \
  --max-frames 100 \
  --save-video
```

## D. Run Method 3: RT-DETR + BoT-SORT

```bash
python -m src.experiments.run_sequence \
  --config configs/m3_rtdetr_botsort_smoke.yaml \
  --dataset-root /path/to/VisDrone2019-MOT-val \
  --sequence-name <sequence_name> \
  --max-frames 100 \
  --save-video
```

## E. Run One Sequence Across All Three Methods

PowerShell:

```powershell
$seq = "uav0000137_00458_v"
$root = "VisDrone2019-MOT-val"

python -m src.experiments.run_sequence `
  --config configs/m1_yolo11_sort_smoke.yaml `
  --dataset-root $root `
  --sequence-name $seq `
  --max-frames 100 `
  --save-video

python -m src.experiments.run_sequence `
  --config configs/m2_yolo26_bytetrack_smoke.yaml `
  --dataset-root $root `
  --sequence-name $seq `
  --max-frames 100 `
  --save-video

python -m src.experiments.run_sequence `
  --config configs/m3_rtdetr_botsort_smoke.yaml `
  --dataset-root $root `
  --sequence-name $seq `
  --max-frames 100 `
  --save-video
```

Bash:

```bash
seq="uav0000137_00458_v"
root="/path/to/VisDrone2019-MOT-val"

python -m src.experiments.run_sequence \
  --config configs/m1_yolo11_sort_smoke.yaml \
  --dataset-root "$root" \
  --sequence-name "$seq" \
  --max-frames 100 \
  --save-video

python -m src.experiments.run_sequence \
  --config configs/m2_yolo26_bytetrack_smoke.yaml \
  --dataset-root "$root" \
  --sequence-name "$seq" \
  --max-frames 100 \
  --save-video

python -m src.experiments.run_sequence \
  --config configs/m3_rtdetr_botsort_smoke.yaml \
  --dataset-root "$root" \
  --sequence-name "$seq" \
  --max-frames 100 \
  --save-video
```

## F. Run Longer Visual Tests

Use `--max-frames 300`:

```bash
python -m src.experiments.run_sequence \
  --config configs/m2_yolo26_bytetrack_smoke.yaml \
  --dataset-root /path/to/VisDrone2019-MOT-val \
  --sequence-name <sequence_name> \
  --max-frames 300 \
  --save-video
```

## G. Run a Full Sequence

Omit `--max-frames`:

```bash
python -m src.experiments.run_sequence \
  --config configs/m3_rtdetr_botsort_smoke.yaml \
  --dataset-root /path/to/VisDrone2019-MOT-val \
  --sequence-name <sequence_name> \
  --save-video
```

## H. Create Side-by-Side Comparison Videos

After generating method videos for the same sequence:

```bash
python -m src.visualization.compare_videos \
  --videos \
    outputs/videos/m1_yolo11_sort_smoke/<sequence_name>.mp4 \
    outputs/videos/m2_yolo26_bytetrack_smoke/<sequence_name>.mp4 \
    outputs/videos/m3_rtdetr_botsort_smoke/<sequence_name>.mp4 \
  --labels "YOLO11+SORT" "YOLO26+ByteTrack" "RT-DETR+BoT-SORT" \
  --output outputs/videos/comparisons/<sequence_name>_m1_m2_m3.mp4
```

PowerShell local example:

```powershell
$seq = "uav0000137_00458_v"

python -m src.visualization.compare_videos `
  --videos `
    "outputs/videos/m1_yolo11_sort_smoke/$seq.mp4" `
    "outputs/videos/m2_yolo26_bytetrack_smoke/$seq.mp4" `
    "outputs/videos/m3_rtdetr_botsort_smoke/$seq.mp4" `
  --labels "YOLO11+SORT" "YOLO26+ByteTrack" "RT-DETR+BoT-SORT" `
  --output "outputs/videos/comparisons/$($seq)_m1_m2_m3.mp4"
```

Pairwise comparisons:

```bash
python -m src.visualization.compare_videos \
  --videos outputs/videos/m1_yolo11_sort_smoke/<sequence_name>.mp4 outputs/videos/m2_yolo26_bytetrack_smoke/<sequence_name>.mp4 \
  --labels "YOLO11+SORT" "YOLO26+ByteTrack" \
  --output outputs/videos/comparisons/<sequence_name>_m1_m2.mp4

python -m src.visualization.compare_videos \
  --videos outputs/videos/m1_yolo11_sort_smoke/<sequence_name>.mp4 outputs/videos/m3_rtdetr_botsort_smoke/<sequence_name>.mp4 \
  --labels "YOLO11+SORT" "RT-DETR+BoT-SORT" \
  --output outputs/videos/comparisons/<sequence_name>_m1_m3.mp4

python -m src.visualization.compare_videos \
  --videos outputs/videos/m2_yolo26_bytetrack_smoke/<sequence_name>.mp4 outputs/videos/m3_rtdetr_botsort_smoke/<sequence_name>.mp4 \
  --labels "YOLO26+ByteTrack" "RT-DETR+BoT-SORT" \
  --output outputs/videos/comparisons/<sequence_name>_m2_m3.mp4
```

## I. Suggested Qualitative Evaluation Protocol

When reviewing output videos, inspect:

- Are pedestrian and car detections visually reasonable?
- Are track IDs stable over time?
- Are IDs lost after occlusion or object crossings?
- Are there duplicate boxes on the same object?
- Are small objects missed?
- Does ByteTrack recover low-confidence objects better than SORT?
- Does BoT-SORT reduce ID switches under camera motion?
- Is RT-DETR too slow compared to YOLO on your hardware?
- Are there obvious false positives, especially on background clutter?

## J. Expected Output Paths

MOT prediction text:

```text
outputs/tracks/<experiment_name>/<sequence_name>.txt
```

Visualization video:

```text
outputs/videos/<experiment_name>/<sequence_name>.mp4
```

Comparison video:

```text
outputs/videos/comparisons/<sequence_name>_m1_m2_m3.mp4
```

Prediction rows use:

```text
frame,id,bb_left,bb_top,bb_width,bb_height,conf,class,vis
```

For predictions, `vis` is always `-1`.
