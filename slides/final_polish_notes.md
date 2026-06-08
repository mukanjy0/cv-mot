# Final Polish Notes

## Tables Changed

- `Baseline Run: First Diagnosis`
  - Increased table from `\scriptsize` to `\footnotesize`.
  - Switched to a full-width `tabular*` layout for projector readability.
  - Kept the original columns: Method, MOTA, IDF1, IDS, FP, FN, FPS.

- `RT-DETR Confidence Tuning`
  - Increased table from `\scriptsize` to `\footnotesize`.
  - Switched to a full-width `tabular*` layout.
  - Kept the original columns and shortened the selection sentence.

- `GMC Ablation: Camera Motion`
  - Increased table from `\scriptsize` to `\small`.
  - Kept the camera-motion focus columns: Family, GMC, MOTA, IDF1, IDS, FPS.

- `Detection Diagnosis: uav0000305_00000_v`
  - Increased table from `\scriptsize` to `\footnotesize`.
  - Kept detection-only columns: Detector, Class, Precision, Recall, AP50, FP, FN.

- `SAHI and Upscaling: Small-Object Tradeoff`
  - Increased table from `\scriptsize` to `\footnotesize`.
  - Added missing `FN` and `IDS` columns.
  - Shortened display names to keep the table readable.

## SAHI/Upscaling Values Added

Source for MOTA, IDF1, FP, FN, and IDS:

- `results/2-tuning/final_summary_by_method.csv`

Added values:

| Display row | Source method | FN | IDS |
|---|---|---:|---:|
| YOLO26 baseline | `m2_yolo26_bytetrack_conf035` | 47.0k | 332 |
| Upscale 2.0 + BoT-SORT | `yolo26_upscale20_botsort` | 33.1k | 295 |
| SAHI 768/0.15/conf0.40 | `sahi_yolo26_botsort_slice768_overlap015_conf040` | 35.5k | 281 |
| RT-DETR conf055 + GMC | `rtdetr_botsort_conf055_gmc_on` | 41.5k | 102 |

Car recall proxy values:

- Baseline `0.299`, upscaling `0.499`, and SAHI `0.473` are from `results/2-tuning/next_experiments_report.md`.
- RT-DETR conf055 + GMC `0.294` is from `results/2-tuning/mot_diagnostics_by_sequence.csv` for `uav0000305_00000_v`, where `car_recall_proxy=0.2943615257048093`.

## Rounding

- MOTA, IDF1, recall, precision, and AP50 are rounded to 3 decimals.
- FP/FN counts are shown in `k` notation where large.
- IDS values are shown as integers.
- FPS values are rounded to 1 decimal in most presentation tables.

## Remaining Manual Checks

- The deck still contains one harmless LaTeX underfull hbox warning on the tracking-method comparison slide. It does not affect compilation or the metric-table slides.
- If presenting under strict time limits, use the shorter subset in `slides/visual_revision_notes.md`.
