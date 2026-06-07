# Overnight Experiment Commands

Run commands from the repository root with the Python environment used for
Methods 2 and 3.

## Install Dependencies

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-trackers.txt
python -m pip install -r requirements-sahi.txt
```

SAHI is separate and optional. If its installation fails, Methods 1-3 can
still run and the SAHI stage will be recorded as non-fatal.

## Smoke-Test Dry Run

Print and validate the smoke commands without running inference:

```bash
python -m src.experiments.run_overnight \
  --queue docs/overnight_experiments/experiment_queue.yaml \
  --dataset-root VisDrone2019-MOT-val \
  --output-dir outputs/overnight/dry_run_check \
  --only-stage smoke_test \
  --dry-run
```

Run the actual 20-frame smoke stage by removing `--dry-run` and choosing a
fresh output directory:

```bash
python -m src.experiments.run_overnight \
  --queue docs/overnight_experiments/experiment_queue.yaml \
  --dataset-root VisDrone2019-MOT-val \
  --output-dir outputs/overnight/smoke_check \
  --only-stage smoke_test
```

## Full Overnight Execution

```bash
python -m src.experiments.run_overnight \
  --queue docs/overnight_experiments/experiment_queue.yaml \
  --dataset-root VisDrone2019-MOT-val \
  --output-dir "outputs/overnight/$(date -u +%Y%m%dT%H%M%SZ)"
```

Record the generated directory name before leaving the process unattended.
Alternatively, omit `--output-dir`; the runner creates the same UTC-style
default path and prints it at completion.

## Resume an Interrupted Run

Use the exact original output directory:

```bash
python -m src.experiments.run_overnight \
  --queue docs/overnight_experiments/experiment_queue.yaml \
  --dataset-root VisDrone2019-MOT-val \
  --output-dir outputs/overnight/<run_id> \
  --resume
```

Completed experiments are skipped. Failed or interrupted experiments receive a
new numbered attempt.

## Force Rerun

Force every selected experiment to receive a new attempt without deleting the
old attempts:

```bash
python -m src.experiments.run_overnight \
  --queue docs/overnight_experiments/experiment_queue.yaml \
  --dataset-root VisDrone2019-MOT-val \
  --output-dir outputs/overnight/<run_id> \
  --force
```

To force only one stage, add for example:

```text
--only-stage sahi_m4_best_effort
```

## Refresh Summaries Only

```bash
python -m src.experiments.run_overnight \
  --queue docs/overnight_experiments/experiment_queue.yaml \
  --dataset-root VisDrone2019-MOT-val \
  --output-dir outputs/overnight/<run_id> \
  --summarize-only
```

This reads completed benchmark CSV files and does not run inference.

## Device Notes

Every current config uses `detector.device: auto`. The project resolves `auto`
to `cuda:0` when CUDA is available, then `mps` when Apple MPS is available,
and otherwise `cpu`.

Override every materialized config with one of:

```text
--device cpu
--device mps
--device cuda:0
```

Use the same override when resuming. `half` remains controlled by each config
and defaults to `false`. Verify a short smoke stage before an unattended
CUDA/MPS run.
