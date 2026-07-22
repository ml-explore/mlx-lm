"""Regression coverage for scripts/laguna_quant_predicate.py.

`scripts/` is not a package (no __init__.py) and the module's own usage
example (see its docstring) assumes it is imported with `scripts/` on
sys.path, so we add it explicitly here rather than relying on package
machinery.
"""

import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from laguna_quant_predicate import build_laguna_quant_predicate  # noqa: E402


class DummyModule:
    """Stand-in for mlx.nn.Module: build_laguna_quant_predicate's predicate
    never inspects the module argument, only the path string."""


class TestLagunaQuantPredicate(unittest.TestCase):
    def _check_paths(self, expert_bits: int, attention_bits: int = 8):
        predicate = build_laguna_quant_predicate(
            expert_bits=expert_bits, attention_bits=attention_bits
        )
        module = DummyModule()

        switch_mlp_paths = [
            "model.layers.1.mlp.switch_mlp",
            "model.layers.3.mlp.switch_mlp.gate_proj",
        ]
        for path in switch_mlp_paths:
            with self.subTest(path=path):
                self.assertEqual(
                    predicate(path, module),
                    {"group_size": 64, "bits": expert_bits, "mode": "affine"},
                )

        non_switch_mlp_paths = [
            "model.layers.1.self_attn.q_proj",
            "model.layers.1.mlp.gate",
            "model.layers.1.mlp.shared_expert.gate_proj",
        ]
        for path in non_switch_mlp_paths:
            with self.subTest(path=path):
                self.assertEqual(
                    predicate(path, module),
                    {"group_size": 64, "bits": attention_bits, "mode": "affine"},
                )

    def test_expert_bits_4(self):
        self._check_paths(expert_bits=4)

    def test_expert_bits_6(self):
        self._check_paths(expert_bits=6)

    def test_expert_bits_2_with_4bit_attention(self):
        # The compatibility report's `4bit-Att-2bit-Ex` recipe: routed
        # experts dominate Laguna's footprint (~116B of ~117.6B stored
        # parameters), so this is the variant that actually shrinks a
        # 118B model down to a 64 GB-Mac-friendly size.
        self._check_paths(expert_bits=2, attention_bits=4)

    def test_invalid_expert_bits_raises(self):
        with self.assertRaises(ValueError):
            build_laguna_quant_predicate(expert_bits=5)


if __name__ == "__main__":
    unittest.main()
