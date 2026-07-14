# Reproducibility

This project is designed to make experiments resumable and auditable. The main
outputs are MOT `tracks.txt` files, CSV summaries, and per-run metadata/logs.

## Environment

Recommended:

- Python 3.12 or 3.13 for ByteTrack/BoT-SORT via BoxMOT;
- Python virtual environment;
- `requirements.txt` for core dependencies;
- `requirements-trackers.txt` for optional BoxMOT trackers;
- `requirements-sahi.txt` for optional SAHI experiments.

Setup:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-trackers.txt
python -m pip install -r requirements-sahi.txt
```

Validation:

```bash
python -m src.experiments.run_sequence --help
python -m src.evaluation.evaluate_mot --help
python -m src.evaluation.evaluate_detection --help
python run_next_experiments.py --help
```

## Expected Dataset Layout

```text
VisDrone2019-MOT-val/
  sequences/
    <sequence_name>/
      0000001.jpg
      ...
  annotations/
    <sequence_name>.txt
  det/
```

The `det/` directory is ignored. All reported predictions come from configured
detectors.

## Model Weights

Ultralytics may download weights on first use. Local model files can also be
placed in the repo root or another path and referenced from YAML configs:

```yaml
detector:
  model_path: yolo26n.pt
```

Large weights are ignored by `.gitignore`.

## Canonical Experiment Command

```bash
python run_next_experiments.py \
  --dataset VisDrone2019-MOT-val \
  --device mps
```

Device options:

- `mps` for Apple Silicon;
- `cuda` for NVIDIA GPU;
- `cpu` for CPU-only environments.

The runner writes to `outputs/next_experiments/default/` and skips completed
attempts on repeated runs. Use `--force` only when intentionally rerunning.

## Output Layout

```text
outputs/next_experiments/default/
  logs/
  stages/
    <stage>/<experiment_id>/attempts/<attempt>/
      benchmark/
        runs/<method>/<sequence>/tracks.txt
        evaluation/
  summaries/
    detection_summary_by_method.csv
    detection_summary_by_sequence.csv
    detection_summary_by_class.csv
    final_summary_by_method.csv
    final_summary_by_sequence.csv
    mot_diagnostics_by_sequence.csv
    gmc_ablation_summary.csv
    next_experiments_report.md
```

Curated copies of the main outputs are kept in `results/2-tuning/` for quick
review and reproducible reporting.

## Rebuilding Summaries

Evaluation from existing tracks:

```bash
python -m src.evaluation.evaluate_mot \
  --dataset-root VisDrone2019-MOT-val \
  --tracks-root <tracks-root> \
  --output-dir outputs/evaluation/existing_tracks
```

Detection-only evaluation:

```bash
python -m src.evaluation.evaluate_detection \
  --dataset-root VisDrone2019-MOT-val \
  --configs \
    configs/overnight/m2_yolo26_bytetrack_conf035.yaml \
    configs/overnight/m3_rtdetr_botsort_conf055.yaml \
  --output-dir outputs/evaluation/detection_only \
  --device mps
```

## Visual Outputs

Render tracks from existing `tracks.txt` outputs:

```bash
python -m src.visualization.render_tracks_video \
  --dataset-root VisDrone2019-MOT-val \
  --tracks-root <runs-root> \
  --sequences uav0000339_00001_v \
  --overwrite
```

The Beamer deck uses still frames, not embedded video:

```bash
cd slides
latexmk -pdf main.tex
```

## Testing

Run the unit/smoke tests:

```bash
python -m pytest
```

The tests cover core evaluation and adapter behavior, including detection
evaluation, MOT evaluation, benchmark output, upscaling wrapper behavior, and
SAHI detector handling.

## Reproducibility Caveats

- FPS is hardware-dependent.
- Ultralytics model revisions and BoxMOT versions can affect exact outputs.
- Some trackers may have nondeterministic behavior depending on backend and
hardware. Treat exact identity IDs as implementation details; evaluate with
metrics.
- Do not compare frame-limited smoke tests directly against full-validation
summaries.
