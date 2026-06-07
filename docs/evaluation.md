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
    summary_metrics.json
```

`per_sequence_metrics.csv` contains one row per method and sequence.
`summary_by_method.csv` and `summary_metrics.csv` contain aggregate metrics per
method. `summary_by_sequence.csv` is a wide comparison table with method
metrics grouped by sequence. `summary_metrics.json` contains all tables plus
the evaluation settings. FPS is populated when prediction files have adjacent
benchmark metadata.

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
