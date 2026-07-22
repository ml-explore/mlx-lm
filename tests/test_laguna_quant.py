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


class TestLagunaQuantPredicateAgainstRealModel(unittest.TestCase):
    """`build_laguna_quant_predicate`'s own unit tests above call the
    returned predicate directly, in isolation, which only proves what the
    predicate *would* return for a path -- not what `mlx_lm.convert`
    actually does with that return value. `mlx_lm.utils.quantize_model`
    wraps any custom predicate with its own `hasattr(module,
    "to_quantized")` check, so paths for modules that can't be quantized
    (like Laguna's `Router`, which stores `weight` as a raw array, and
    every `RMSNorm`) never reach our predicate at all, regardless of what
    it would have returned for them. This test exercises the real
    `quantize_model` entry point against an actual (tiny) `Model` instance
    to confirm that end-to-end behavior, not just the predicate's isolated
    return values.
    """

    def test_router_and_norms_stay_full_precision_through_quantize_model(self):
        import mlx.core as mx
        import mlx.nn as nn

        from mlx_lm.models import laguna
        from mlx_lm.models.switch_layers import QuantizedSwitchLinear
        from mlx_lm.utils import quantize_model

        args = laguna.ModelArgs(
            model_type="laguna",
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=16,
            rope_parameters={
                "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            },
            sliding_window=8,
            layer_types=["full_attention", "sliding_attention"],
            num_attention_heads_per_layer=[8, 8],
            gating="per-head",
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=64,
            mlp_only_layers=[],
            decoder_sparse_step=1,
            moe_routed_scaling_factor=2.5,
        )
        model = laguna.Model(args)
        mx.eval(model.parameters())

        predicate = build_laguna_quant_predicate(expert_bits=4, attention_bits=8)
        quantized_model, _ = quantize_model(
            model,
            config={"model_type": "laguna"},
            group_size=64,
            bits=8,
            quant_predicate=predicate,
        )

        router = quantized_model.model.layers[0].mlp.gate
        self.assertIsInstance(router, laguna.Router)
        self.assertEqual(router.weight.dtype, mx.float32)

        norm = quantized_model.model.layers[0].input_layernorm
        self.assertIsInstance(norm, nn.RMSNorm)

        expert_proj = quantized_model.model.layers[0].mlp.switch_mlp.gate_proj
        self.assertIsInstance(expert_proj, QuantizedSwitchLinear)

        attn_proj = quantized_model.model.layers[0].self_attn.q_proj
        self.assertIsInstance(attn_proj, nn.QuantizedLinear)


if __name__ == "__main__":
    unittest.main()
