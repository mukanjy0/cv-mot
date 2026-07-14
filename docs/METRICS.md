# Metrics

This project reports both tracking metrics and detection-only diagnostics.
Tracking metrics come from `motmetrics`; detection-only metrics are computed by
the project evaluator before any tracker is applied.

## Tracking Metrics

| Metric | Meaning | How to read it |
|---|---|---|
| MOTA | Multi-object tracking accuracy | Higher is better; penalizes FP, FN, and IDS |
| MOTP | Localization distance/precision from `motmetrics` | Lower distance is better under the library convention used here |
| IDF1 | Identity F1 score | Higher is better; emphasizes identity consistency |
| IDP | Identity precision | Higher is better |
| IDR | Identity recall | Higher is better |
| FP | False positives | Lower is better |
| FN | False negatives | Lower is better |
| IDS | Identity switches | Lower is better |
| FPS | Runtime throughput from benchmark metadata | Higher is faster |

The core evaluator lives in `src/evaluation/evaluate_mot.py`.

## Matching Assumptions

- Evaluation keeps only VisDrone class `1` pedestrian and class `4` car.
- Matching prevents cross-class matches.
- The IoU threshold is 0.50.
- Prediction rows must use:

```text
frame,id,bb_left,bb_top,bb_width,bb_height,conf,class,vis
```

## Detection-Only Metrics

The detection-only evaluator lives in `src/evaluation/evaluate_detection.py`.
It evaluates detector outputs before tracking.

| Metric | Meaning |
|---|---|
| GT boxes | Number of ground-truth boxes after class filtering |
| Predicted boxes | Number of detector boxes after class mapping/filtering |
| TP | One-to-one matched predicted boxes at IoU 0.50 |
| FP | Predicted boxes not matched to GT |
| FN | GT boxes not matched by predictions |
| Precision | `TP / (TP + FP)` |
| Recall | `TP / (TP + FN)` |
| AP50 | Area under precision-recall curve at IoU 0.50 |

Detection matching is frame-by-frame and class-by-class. Detections are sorted
by confidence for AP50. The AP50 implementation uses an all-point interpolated
precision envelope.

## Diagnostic Metrics

`mot_diagnostics_by_sequence.csv` adds interpretability fields:

| Field | Use |
|---|---|
| `unique_predicted_ids` | High values can indicate fragmentation or over-creation |
| `short_tracks_leq_3` | Many very short tracks suggest false tracks or unstable association |
| `average_predicted_track_length` | Short average length can indicate track fragmentation |
| `median_predicted_bbox_area` | Helps identify small-object-heavy predictions |
| `median_gt_bbox_area` | Helps compare prediction scale to GT scale |
| `car_recall_proxy` | Approximate class-4 track/GT match ratio |

The car recall proxy is useful for comparing small-object modes, but it is not
a replacement for full detection AP/recall or MOTA.

## Rounding in Documentation

Documentation and slides use:

- 3 decimals for MOTA, IDF1, precision, recall, and AP50;
- `k` notation for large FP/FN counts;
- integer IDS values;
- 1 decimal for FPS.

Raw values are preserved in CSV files under `results/`.

## Common Interpretation Traps

- MOTA can fall even when recall improves if FP rises sharply.
- IDF1 can favor recall-oriented methods that are not the best MOTA choice.
- A low IDS count does not prove good detection recall.
- FPS is hardware-dependent and should be compared only within the same run
  environment.
- Smoke tests and frame-limited results are not directly comparable to full
  validation summaries.
