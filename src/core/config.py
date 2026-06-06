from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "pyyaml is required to load experiment configs. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if "name" not in config:
        raise ValueError("Config is missing required field: name")
    if "detector" not in config:
        raise ValueError("Config is missing required section: detector")
    if "tracker" not in config:
        raise ValueError("Config is missing required section: tracker")

    return config
