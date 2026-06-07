from __future__ import annotations

import copy
import sys
from typing import Any, Mapping


def resolve_device(device: str | None) -> str:
    """Resolve an explicit or automatic PyTorch inference device."""

    if device not in (None, "auto"):
        return str(device)

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda:0"

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"

    return "cpu"


def device_diagnostics(device: str | None) -> dict[str, Any]:
    """Return reproducibility details for device selection."""

    diagnostics: dict[str, Any] = {
        "sys_executable": sys.executable,
        "torch_version": None,
        "torch_cuda_is_available": False,
        "torch_mps_is_built": False,
        "torch_mps_is_available": False,
        "requested_device": "auto" if device is None else device,
    }
    try:
        import torch
    except ImportError:
        diagnostics["resolved_device"] = resolve_device(device)
        return diagnostics

    mps = getattr(torch.backends, "mps", None)
    diagnostics.update(
        {
            "torch_version": torch.__version__,
            "torch_cuda_is_available": bool(torch.cuda.is_available()),
            "torch_mps_is_built": bool(mps is not None and mps.is_built()),
            "torch_mps_is_available": bool(mps is not None and mps.is_available()),
            "resolved_device": resolve_device(device),
        }
    )
    return diagnostics


def apply_device_to_detector_configs(
    config: Mapping[str, Any], device: str
) -> dict[str, Any]:
    """Copy a config and set device on every nested detector mapping."""

    resolved = copy.deepcopy(dict(config))

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            detector = value.get("detector")
            if isinstance(detector, dict):
                detector["device"] = device
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(resolved)
    return resolved


def format_device_diagnostics(diagnostics: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "Device environment:",
            f"  sys.executable: {diagnostics['sys_executable']}",
            f"  torch.__version__: {diagnostics['torch_version']}",
            "  torch.cuda.is_available(): "
            f"{diagnostics['torch_cuda_is_available']}",
            "  torch.backends.mps.is_built(): "
            f"{diagnostics['torch_mps_is_built']}",
            "  torch.backends.mps.is_available(): "
            f"{diagnostics['torch_mps_is_available']}",
            f"  resolved device: {diagnostics['resolved_device']}",
        ]
    )
