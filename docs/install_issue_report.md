# Dependency Installation Issue Report

## Error encountered

The active environment was `/Users/katharsis/Developer/cv/mot/.venv` with
Python 3.14.5 on macOS arm64. Running:

```bash
python -m pip install -r requirements.txt
```

made pip backtrack through many BoxMOT and FilterPy versions, try an old NumPy
source distribution, and finally fail with:

```text
pip._vendor.pyproject_hooks._impl.BackendUnavailable:
Cannot import 'setuptools.build_meta'
```

A direct BoxMOT resolution exposed the underlying incompatibilities:

```text
Could not find a version that satisfies the requirement
torchvision<0.18.0,>=0.17.1; sys_platform == "darwin"
```

## Root cause

BoxMOT was included in the base requirements even though it is only needed by
ByteTrack and BoT-SORT. Current BoxMOT releases exclude Python 3.14. The older
BoxMOT candidate considered by pip pins `numpy==1.26.4` and requires an old
macOS torchvision range; compatible Python 3.14 wheels do not exist. Resolver
backtracking obscured this with the later build-backend error.

The core Ultralytics, OpenCV, NumPy, SciPy, FilterPy, PyYAML, and tqdm set
resolves independently on this Python 3.14 arm64 environment.

## Changes

- `requirements.txt`: core YOLO + SORT dependencies only.
- `requirements-trackers.txt`: optional BoxMOT dependency, skipped on Python
  3.14 with an environment marker.
- `src/tracking/boxmot_adapter.py`: actionable lazy ImportError for optional
  trackers.
- `README.md` and `docs/environment_setup.md`: virtual environment,
  installation, verification, and troubleshooting instructions.

## Install and verify

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -c "import cv2, numpy, scipy, yaml, tqdm, ultralytics; print('core deps ok')"
python -m src.experiments.run_sequence --help
```

For ByteTrack or BoT-SORT, create a Python 3.12 or 3.13 virtual environment,
then run:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-trackers.txt
python -c "import boxmot; print('boxmot ok')"
```

## Verification result

In the original Python 3.14.5 virtual environment, the core requirements
installed successfully. The core import command printed `core deps ok`,
`python -m pip check` reported no broken requirements, and
`python -m src.experiments.run_sequence --help` exited successfully.

BoxMOT was not installed, as expected on Python 3.14. Constructing a ByteTrack
adapter produced the documented optional-dependency message without preventing
the project or CLI from importing.
