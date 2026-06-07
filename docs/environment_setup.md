# Environment Setup

## Create and activate a virtual environment

Python 3.12 or 3.13 is recommended when ByteTrack or BoT-SORT support is
needed. The core YOLO + SORT environment also installs on Python 3.14.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
```

Confirm every install command targets the virtual environment:

```bash
which python
python --version
python -m pip --version
python -m pip list
```

Use `where python` instead of `which python` on Windows.

## Install core dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

This installs the dependencies for dataset loading, visualization, MOT output,
Ultralytics detectors, and the built-in SORT tracker. GPU support is not
required.

## Install optional trackers

ByteTrack and BoT-SORT adapters use BoxMOT:

```bash
python -m pip install -r requirements-trackers.txt
```

BoxMOT is intentionally separate from the core environment. Its current
releases do not support Python 3.14, so the requirements marker skips it on
that interpreter. Create a Python 3.12 or 3.13 virtual environment to use
Methods 2 and 3.

## Verify the environment

```bash
python -c "import cv2, numpy, scipy, yaml, tqdm, ultralytics; print('core deps ok')"
python -m src.experiments.run_sequence --help
```

When optional trackers are installed:

```bash
python -c "import boxmot; print('boxmot ok')"
```

## Common failures

- Wrong interpreter: activate `.venv` and compare `which python` with
  `python -m pip --version`.
- Stale packaging tools: run
  `python -m pip install --upgrade pip setuptools wheel`.
- Python 3.14 plus BoxMOT: use Python 3.12 or 3.13 for the optional tracker
  environment. Do not add BoxMOT back to the base requirements.
- PyTorch/CUDA mismatch: use the default CPU-capable install unless a matching
  CUDA build is explicitly required.
- OpenCV package conflicts: avoid installing multiple OpenCV variants in the
  same environment.
- Apple Silicon source builds: prefer supported Python versions with published
  arm64 wheels instead of forcing old NumPy or torchvision releases to build.
