# Experiments

This document is the canonical experiment map for the repository. It separates
completed full-validation runs from quick/debug runs so results are not mixed
across protocols.

## Dataset and Split

- Dataset: `VisDrone2019-MOT-val`
- Local sequence directory: `VisDrone2019-MOT-val/sequences/`
- Ground truth: `VisDrone2019-MOT-val/annotations/`
- Ignored input: `VisDrone2019-MOT-val/det/`
- Evaluated classes:
  - VisDrone class `1`: pedestrian
  - VisDrone class `4`: car

The completed next-step suite covers the validation sequences available in the
local `VisDrone2019-MOT-val` folder and reports aggregate summaries over 2846
frames in `results/2-tuning/final_summary_by_method.csv`.

## Protocol Tiers

| Tier | Purpose | Canonical location | Compare across methods? |
|---|---|---|---|
| Smoke runs | Verify code path, CLI, videos | `outputs/`, ad hoc benchmark folders | No |
| Representative benchmark | Fast multi-sequence checks, often frame-limited | `results/0-rep-test/` and benchmark outputs | Only within same command/settings |
| Overnight tuning | Staged M1/M2/M3/M4 screening | `results/1-overnight/` | Yes, with protocol note |
| Next-step suite | Full validation diagnostics and targeted ablations | `results/2-tuning/` | Yes, canonical |

Use `results/2-tuning/` for README tables and reported results unless
explicitly describing older or frame-limited runs.

## Canonical Commands

Run the main next-step experiment suite:

```bash
python run_next_experiments.py \
  --dataset VisDrone2019-MOT-val \
  --device mps
```

Use `--device cuda` on CUDA machines or `--device cpu` for CPU-only execution.
The suite is resumable. Pass `--force` only when intentionally creating new
attempts.

Run a representative benchmark:

```bash
python -m src.experiments.run_benchmark \
  --dataset-root VisDrone2019-MOT-val \
  --configs \
    configs/m1_yolo11_sort_smoke.yaml \
    configs/m2_yolo26_bytetrack_smoke.yaml \
    configs/m3_rtdetr_botsort_smoke.yaml \
  --max-frames 300 \
  --save-video \
  --evaluate
```

Run detection-only diagnostics:

```bash
python -m src.evaluation.evaluate_detection \
  --dataset-root VisDrone2019-MOT-val \
  --configs \
    configs/overnight/m2_yolo26_bytetrack_conf035.yaml \
    configs/overnight/m3_rtdetr_botsort_conf055.yaml \
  --output-dir outputs/evaluation/detection_only \
  --device mps
```

Render tracks from existing `tracks.txt` files:

```bash
python -m src.visualization.render_tracks_video \
  --dataset-root VisDrone2019-MOT-val \
  --tracks-root outputs/next_experiments/default/stages/gmc_ablation/rtdetr_botsort_conf055_gmc_on/attempts/001/benchmark/runs \
  --sequences uav0000339_00001_v \
  --overwrite
```

## Completed Detector/Tracker Families

| Family | Configs | Notes |
|---|---|---|
| YOLOv11 + SORT | `configs/m1_yolo11_sort_smoke.yaml` | geometric baseline |
| YOLO26 + ByteTrack | `configs/m2_yolo26_bytetrack_smoke.yaml`, `configs/overnight/m2_*.yaml` | fast confidence-aware baseline |
| RT-DETR + BoT-SORT | `configs/m3_rtdetr_botsort_smoke.yaml`, `configs/overnight/m3_*.yaml`, `configs/next/m3_*.yaml` | strongest balanced family |
| YOLO26 + BoT-SORT GMC ablation | `configs/next/yolo26_botsort_gmc_*.yaml` | isolates GMC with YOLO detector |
| RT-DETR + BoT-SORT GMC ablation | `configs/next/rtdetr_botsort_conf055_gmc_*.yaml` | final selected method comes from GMC-on row |
| YOLO26 upscaling | `configs/next/yolo26_upscale*.yaml` | small-object recall-oriented experiment |
| RT-DETR upscaling | `configs/next/rtdetr_upscale15_botsort.yaml` | expensive RT-DETR upscaling check |
| YOLO26 SAHI strict | `configs/next/sahi_yolo26_botsort_*.yaml` | sliced inference recall/FP tradeoff |

## Canonical Results

Source: `results/2-tuning/final_summary_by_method.csv`.

| Method | MOTA | IDF1 | FP | FN | IDS | FPS | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| `rtdetr_botsort_conf055_gmc_on` | 0.214 | 0.436 | 8.8k | 41.5k | 102 | 4.2 | best MOTA, selected final |
| `m3_rtdetr_botsort_conf050` | 0.211 | 0.465 | 11.6k | 38.9k | 133 | 4.0 | best IDF1, more recall-oriented |
| `m3_rtdetr_botsort_conf060` | 0.206 | 0.402 | 6.6k | 44.2k | 99 | 4.0 | stricter threshold, fewer FP |
| `m3_rtdetr_botsort_conf065` | 0.187 | 0.365 | 4.8k | 47.3k | 95 | 3.9 | too recall-poor |
| `yolo26_botsort_gmc_on` | 0.180 | 0.370 | 6.4k | 46.1k | 176 | 39.0 | faster with GMC benefit |
| `m2_yolo26_bytetrack_conf035` | 0.169 | 0.325 | 6.0k | 47.0k | 332 | 48.9 | fast baseline |
| `yolo26_upscale20_botsort` | 0.110 | 0.463 | 23.8k | 33.1k | 295 | 14.0 | high recall, high FP |
| `sahi_yolo26_botsort_slice768_overlap015_conf040` | 0.092 | 0.423 | 22.5k | 35.5k | 281 | 3.8 | high recall, high FP |

## GMC Ablation

Source: `results/2-tuning/gmc_ablation_summary.csv`.

| Family | MOTA off | MOTA on | IDF1 off | IDF1 on | IDS off | IDS on |
|---|---:|---:|---:|---:|---:|---:|
| RT-DETR + BoT-SORT conf055 | 0.206 | 0.214 | 0.391 | 0.436 | 294 | 102 |
| YOLO26 + BoT-SORT | 0.171 | 0.180 | 0.332 | 0.370 | 323 | 176 |

GMC improves identity stability in aggregate. `uav0000182_00000_v` is noted in
the slides as a sequence-level exception, which is why conclusions are stated
as aggregate rather than universal.

## Detection-Only Crossroad Diagnosis

Source: `results/2-tuning/detection_summary_by_sequence.csv`.

For `uav0000305_00000_v`:

| Detector | Class | Precision | Recall | AP50 | FP | FN |
|---|---|---:|---:|---:|---:|---:|
| YOLO26 | car | 0.814 | 0.160 | 0.141 | 132 | 3040 |
| RT-DETR conf055 | car | 0.890 | 0.314 | 0.298 | 141 | 2482 |
| YOLO26 | pedestrian | 0.000 | 0.000 | 0.000 | 0 | 540 |
| RT-DETR conf055 | pedestrian | 0.000 | 0.000 | 0.000 | 8 | 540 |

This is the main evidence for a detector-limited failure mode on the crossroad
sequence.

## Output Formats

Track files are written in MOT-style rows:

```text
frame,id,bb_left,bb_top,bb_width,bb_height,conf,class,vis
```

Frame IDs start at `1`. Internally the project uses `xyxy` boxes and converts
to `xywh` for output.

## What Not To Claim

- Do not claim original implementation of YOLO, RT-DETR, ByteTrack, BoT-SORT,
  or SAHI.
- Do not compare smoke/frame-limited results against full-validation results
  without noting the protocol mismatch.
- Do not claim detector fine-tuning; these are evaluation and integration
  experiments.
