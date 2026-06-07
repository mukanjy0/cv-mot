# Safe Overnight MOT Experiment Plan

## Objective

Run a bounded, reproducible comparison over all seven VisDrone2019-MOT
validation sequences without training models or replacing existing outputs.
The pipeline runs methods 1-3 first, screens compact M3 and M2 grids, attempts
SAHI as an isolated optional stage, and produces one final comparison from
full-validation runs.

The runner executes one config per subprocess. Each subprocess gets its own
numbered attempt, resolved config snapshot, metadata, stdout, stderr, and
combined log. Completed experiments are skipped on resume. `--force` creates a
new attempt and never deletes an older attempt.

## Validation Sequences

All full-validation runs use:

1. `uav0000086_00000_v`
2. `uav0000117_02622_v`
3. `uav0000137_00458_v`
4. `uav0000182_00000_v`
5. `uav0000268_05773_v`
6. `uav0000305_00000_v`
7. `uav0000339_00001_v`

Prior 300-frame results identify `uav0000117_02622_v` as the M3 FP stress
sequence: M3 produced 7,909 FP there, the highest among the five sequences
already evaluated. Compact tuning screens therefore use its first 300 frames.

## Stage 1: Smoke Test

Goal: prove that model loading, detector/tracker imports, MOT writing, and
evaluation work before long jobs begin.

Configs:

- `configs/m1_yolo11_sort_smoke.yaml`
- `configs/m2_yolo26_bytetrack_smoke.yaml`
- `configs/m3_rtdetr_botsort_smoke.yaml`

Run: first 20 frames of `uav0000117_02622_v`, no video, evaluation enabled.

Stop condition: any failure is fatal. Do not continue into overnight work if
the smoke test cannot produce tracks and metrics for all three methods.

## Stage 2: Full Baselines

Goal: establish complete seven-sequence M1/M2/M3 baselines.

Configs: the same three baseline configs above.

Run: all frames of all seven sequences, no video, evaluation enabled.

Stop condition: any failure is fatal. These baselines provide fallback configs
for later stages and are required for a defensible final comparison.

Expected summary:

- `summaries/full_baseline_summary_by_method.csv`

## Stage 3: Compact M3 Tuning

Goal: reduce M3 false positives while preserving its FN and IDF1 advantages.

Screen these six configs on the first 300 frames of the FP stress sequence:

- `configs/m3_rtdetr_botsort_smoke.yaml` (current baseline, confidence 0.25)
- `configs/overnight/m3_rtdetr_botsort_conf035.yaml`
- `configs/overnight/m3_rtdetr_botsort_conf045.yaml`
- `configs/overnight/m3_rtdetr_botsort_conf055.yaml`
- `configs/overnight/m3_rtdetr_botsort_conf045_strict_new.yaml`
- `configs/overnight/m3_rtdetr_botsort_conf055_strict_new.yaml`

The two best screening configs are promoted to all seven full sequences.
Selection then uses only the promoted full-validation results.

Continue condition: tuning is non-fatal. If some configs fail, rank the
successful configs. If no promoted config completes, final comparison falls
back to the full M3 baseline.

Expected files:

- `summaries/tuning_m3_summary.csv`
- `stages/tune_m3/selection.json`

## Stage 4: Compact M2 Tuning

Goal: improve M2 MOTA without losing its speed advantage.

Screen these five configs on the first 300 frames of the FP stress sequence:

- `configs/m2_yolo26_bytetrack_smoke.yaml` (current baseline, confidence 0.20)
- `configs/overnight/m2_yolo26_bytetrack_conf025.yaml`
- `configs/overnight/m2_yolo26_bytetrack_conf035.yaml`
- `configs/overnight/m2_yolo26_bytetrack_conf045.yaml`
- `configs/overnight/m2_yolo26_bytetrack_conf035_low_min.yaml`

The last config tests BoxMOT's lower `min_conf`. Because detector confidence is
0.35, detections below 0.35 never reach ByteTrack; this candidate documents
that interaction and may match the normal 0.35 result.

The top two screening configs are promoted to all seven sequences. If tuning
does not complete, final comparison uses the full M2 baseline.

Expected files:

- `summaries/tuning_m2_summary.csv`
- `stages/tune_m2/selection.json`

## Stage 5: SAHI Method 4, Best Effort

Goal: test whether sliced YOLO26 inference recovers small objects sufficiently
to justify its expected speed cost.

The first three candidates inherit the selected M2 detector confidence. The
strict candidate uses that confidence plus 0.10:

- `configs/overnight/m4_sahi_m2_slice640_overlap020.yaml`
- `configs/overnight/m4_sahi_m2_slice640_overlap030.yaml`
- `configs/overnight/m4_sahi_m2_slice512_overlap020.yaml`
- `configs/overnight/m4_sahi_m2_slice640_overlap020_strict.yaml`

Screen: first 100 frames of `uav0000117_02622_v`.

Promotion: only the best successful candidate runs on all seven sequences,
with visualization videos saved for each sequence.
SAHI uses sliced inference only, class-aware `GREEDYNMM`, IoU matching at 0.5,
and the existing ByteTrack wrapper. It never uses `model.track()`.

Continue condition: the entire stage is non-fatal. Missing SAHI dependencies,
model incompatibility, runtime failures, or no successful promotion are
recorded in `failures.csv` and `summaries/final_notes.md`; M1-M3 continue.

Install the optional dependency separately:

```bash
python -m pip install -r requirements-sahi.txt
```

The integration follows the official SAHI Ultralytics sliced-inference API:
<https://obss.github.io/sahi/notebooks/inference_for_ultralytics/>.

## Stage 6: Final Comparison

Goal: create a full-validation comparison of:

- M1 baseline
- best promoted M2, or M2 baseline fallback
- best promoted M3, or M3 baseline fallback
- best promoted SAHI M4, only if available

This stage reuses prior full-validation outputs. It does not rerun identical
inference. Every source is checked for all seven sequences and full-frame
execution before it can enter the final summaries.

Stop condition: missing M1, M2, or M3 full-validation sources are fatal.
Missing M4 is allowed.

Expected files:

- `summaries/final_summary_by_method.csv`
- `summaries/final_summary_by_sequence.csv`
- `summaries/final_notes.md`

## Output Structure

```text
outputs/overnight/<run_id>/
  metadata.json
  plan_snapshot.md
  queue_snapshot.yaml
  commands.txt
  failures.csv
  logs/
    <stage>__<phase>__<experiment>__attemptNNN.stdout.log
    <stage>__<phase>__<experiment>__attemptNNN.stderr.log
    <stage>__<phase>__<experiment>__attemptNNN.log
  stages/
    smoke_test/
    full_baseline_m1_m2_m3/
    tune_m3/
    tune_m2/
    sahi_m4_best_effort/
    final_comparison/
  summaries/
    full_baseline_summary_by_method.csv
    tuning_m3_summary.csv
    tuning_m2_summary.csv
    sahi_summary.csv
    final_summary_by_method.csv
    final_summary_by_sequence.csv
    final_notes.md
```

Each experiment directory contains `metadata.json` and
`attempts/NNN/resolved_config.yaml`. The benchmark's tracks, evaluation, and
per-sequence metadata remain under that attempt's `benchmark/` directory.

## Overnight Safety Rules

- Never run with `--force` for the first execution.
- Resume the same output directory after interruption.
- A changed queue or dataset root is rejected on resume and force reruns. Use a
  fresh output directory for changed inputs.
- No stage deletes output files.
- Videos are disabled to reduce disk use and avoid encoding failures.
- Baseline and smoke failures stop the queue.
- Tuning and SAHI failures are recorded and use documented fallbacks.
- No model training, arbitrary frame slicing, or combinatorial search occurs.
