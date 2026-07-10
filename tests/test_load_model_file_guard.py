# Copyright © 2024 Apple Inc.

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx

from mlx_lm.utils import load_model

# A minimal, valid custom architecture module. When dropped inside the model
# directory and referenced by ``model_file``, ``load_model`` should import and
# instantiate it.
CUSTOM_ARCH_SOURCE = """
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelArgs:
    hidden_size: int = 4

    @classmethod
    def from_dict(cls, config):
        return cls(hidden_size=config.get("hidden_size", 4))


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.linear = nn.Linear(args.hidden_size, args.hidden_size)

    def __call__(self, x):
        return self.linear(x)
"""


class TestLoadModelFileGuard(unittest.TestCase):
    def _make_model_dir(self, model_file):
        """Create a temp model dir whose config points at ``model_file``."""
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        model_path = Path(tmp.name)
        config = {
            "model_type": "custom",
            "model_file": model_file,
            "hidden_size": 4,
        }
        with open(model_path / "config.json", "w") as f:
            json.dump(config, f)
        # A valid custom architecture living inside the model directory.
        with open(model_path / "arch.py", "w") as f:
            f.write(CUSTOM_ARCH_SOURCE)
        # Fake weights so the "strict" weight-file check is satisfied.
        mx.save_safetensors(
            str(model_path / "model.safetensors"),
            {"linear.weight": mx.zeros((4, 4)), "linear.bias": mx.zeros((4,))},
        )
        return model_path

    def test_absolute_model_file_rejected(self):
        model_path = self._make_model_dir("/etc/passwd_arch.py")
        with self.assertRaises(ValueError):
            load_model(model_path, trust_remote_code=True)

    def test_escaping_model_file_rejected(self):
        model_path = self._make_model_dir("../evil_arch.py")
        with self.assertRaises(ValueError):
            load_model(model_path, trust_remote_code=True)

    def test_non_python_model_file_rejected(self):
        model_path = self._make_model_dir("arch.txt")
        with self.assertRaises(ValueError):
            load_model(model_path, trust_remote_code=True)

    def test_legit_relative_model_file_loads(self):
        model_path = self._make_model_dir("arch.py")
        model, config = load_model(model_path, trust_remote_code=True)
        self.assertEqual(config["model_file"], "arch.py")
        # The custom Model was imported and instantiated.
        self.assertTrue(hasattr(model, "linear"))


if __name__ == "__main__":
    unittest.main()
