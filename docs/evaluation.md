# Representative Benchmark and MOT Evaluation

The benchmark wrapper runs the existing detector/tracker methods without using
`model.track()`. It evaluates only VisDrone class `1` (pedestrian) and class
`4` (car). The dataset `det/` directory is ignored.

## Run the representative benchmark

```bash
python -m src.experiments.run_benchmark \
  --dataset-root VisDrone2019-MOT-val \
  --configs \
    configs/m1_yolo11_sort_smoke.yaml \
    configs/m2_yolo26_bytetrack_smoke.yaml \
    configs/m3_rtdetr_botsort_smoke.yaml \
  --sequences \
    uav0000137_00458_v \
    uav0000268_05773_v \
    uav0000117_02622_v \
    uav0000305_00000_v \
    uav0000086_00000_v \
  --max-frames 300 \
  --save-video \
  --evaluate
```

`--sequences` defaults to those five representative sequences. Omit
`--save-video` for a faster quantitative run. Use `--run-id NAME` to choose a
stable output folder name; otherwise a UTC timestamp is used.

Outputs are written under:

```text
outputs/benchmarks/<run_id>/
  metadata.json
  runs/<method>/<sequence>/
    tracks.txt
    metadata.json
    video.mp4
  evaluation/
    per_sequence_metrics.csv
    summary_metrics.csv
    summary_by_method.csv
    summary_by_sequence.csv
    mot_diagnostics_by_sequence.csv
    summary_metrics.json
```

`per_sequence_metrics.csv` contains one row per method and sequence.
`summary_by_method.csv` and `summary_metrics.csv` contain aggregate metrics per
method. `summary_by_sequence.csv` is a wide comparison table with method
metrics grouped by sequence. `summary_metrics.json` contains all tables plus
the evaluation settings. FPS is populated when prediction files have adjacent
benchmark metadata.

`mot_diagnostics_by_sequence.csv` adds one row per method and sequence with
MOTA, IDF1, FP, FN, IDS, FPS, unique predicted IDs, tracks with length <= 3
frames, average predicted track length, median predicted bbox area, median GT
bbox area, and a car recall proxy computed from class-4 track/GT IoU matches.

## Evaluate existing tracks

Existing project outputs use `<tracks-root>/<method>/<sequence>.txt`:

```bash
python -m src.evaluation.evaluate_mot \
  --dataset-root VisDrone2019-MOT-val \
  --tracks-root outputs/tracks \
  --sequences \
    uav0000137_00458_v \
    uav0000268_05773_v \
    uav0000117_02622_v \
    uav0000305_00000_v \
    uav0000086_00000_v \
  --max-frames 300 \
  --output-dir outputs/evaluation/existing_tracks
```

Pass the same `--max-frames` value used to generate predictions. Evaluation
filters both ground truth and predictions to frames `1..N`. Prediction files
must include the project MOT columns:

```text
frame,id,bb_left,bb_top,bb_width,bb_height,conf,class,vis
```

Matching uses IoU distance with a `0.5` threshold and prevents matches across
classes. MOTA, MOTP, IDF1, IDP, IDR, FP, FN, IDS, and frame count follow
`motmetrics`; MOTP therefore uses its distance convention.

## Detection-only diagnostics

Evaluate detector outputs before tracking:

```bash
python -m src.evaluation.evaluate_detection \
  --dataset-root VisDrone2019-MOT-val \
  --configs \
    configs/overnight/m2_yolo26_bytetrack_conf035.yaml \
    configs/overnight/m3_rtdetr_botsort_conf055.yaml \
  --sequences \
    uav0000137_00458_v \
    uav0000305_00000_v \
  --output-dir outputs/evaluation/detection_only \
  --device mps
```

The evaluator runs `model.predict()` through the project detector wrappers and
does not use `model.track()` or the dataset `det/` folder. It writes:

```text
detection_summary_by_method.csv
detection_summary_by_sequence.csv
detection_summary_by_class.csv
detection_summary.json
```

Metrics are computed for VisDrone classes `1` and `4` only. Matching is
one-to-one per frame and class at IoU 0.50, greedily ordered by detector
confidence. AP50 is an all-point interpolated precision-envelope area over the
confidence-ranked predictions.

## Next experiment suite

The end-to-end next-step entrypoint is:

```bash
python run_next_experiments.py \
  --dataset VisDrone2019-MOT-val \
  --device mps
```

It is resumable under `outputs/next_experiments/default/`, writes per-command
logs, and produces final method/sequence summaries, detection diagnostics,
MOT diagnostics, `gmc_ablation_summary.csv`, and `next_experiments_report.md`.
Use `--render-debug` to create optional GT-vs-track MP4s for selected
sequences, including `uav0000305_00000_v`.
