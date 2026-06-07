from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from src.core.device import apply_device_to_detector_configs, resolve_device


class DeviceTest(unittest.TestCase):
    def test_auto_prefers_cuda_then_mps(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=True):
            with patch.object(torch.backends.mps, "is_available", return_value=True):
                self.assertEqual(resolve_device("auto"), "cuda:0")

        with patch.object(torch.cuda, "is_available", return_value=False):
            with patch.object(torch.backends.mps, "is_available", return_value=True):
                self.assertEqual(resolve_device("auto"), "mps")

        with patch.object(torch.cuda, "is_available", return_value=False):
            with patch.object(torch.backends.mps, "is_available", return_value=False):
                self.assertEqual(resolve_device("auto"), "cpu")

    def test_explicit_device_is_unchanged(self) -> None:
        self.assertEqual(resolve_device("mps"), "mps")
        self.assertEqual(resolve_device("cuda:0"), "cuda:0")

    def test_device_is_applied_to_nested_detector_configs(self) -> None:
        config = {
            "detector": {"device": "auto"},
            "nested": {"detector": {"type": "secondary"}},
        }
        resolved = apply_device_to_detector_configs(config, "mps")
        self.assertEqual(resolved["detector"]["device"], "mps")
        self.assertEqual(resolved["nested"]["detector"]["device"], "mps")
        self.assertEqual(config["detector"]["device"], "auto")


if __name__ == "__main__":
    unittest.main()
