import math
import unittest

import mlx.core as mx

from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.laguna import (
    LagunaMLP,
    LagunaSparseMoE,
    Model,
    ModelArgs,
    apply_attention_gate,
    router_select,
)


def tiny_args(**overrides):
    values = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_attention_heads_per_layer": [2, 4],
        "head_dim": 4,
        "layer_types": ["full_attention", "sliding_attention"],
        "sliding_window": 4,
        "num_experts": 3,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 8,
        "shared_expert_intermediate_size": 8,
        "mlp_layer_types": ["dense", "sparse"],
        "rope_parameters": {
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 500000.0,
                "factor": 2.0,
                "original_max_position_embeddings": 8,
                "beta_slow": 1.0,
                "beta_fast": 32.0,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            },
        },
    }
    values.update(overrides)
    return ModelArgs(**values)


class TestLaguna(unittest.TestCase):
    def setUp(self):
        mx.random.seed(7)

    def test_tiny_forward_and_layer_topology(self):
        model = Model(tiny_args())
        self.assertEqual(model.layers[0].self_attn.num_heads, 2)
        self.assertEqual(model.layers[1].self_attn.num_heads, 4)
        self.assertEqual(model.layers[0].self_attn.num_kv_heads, 2)
        self.assertEqual(model.layers[1].self_attn.num_kv_heads, 2)
        self.assertIsInstance(model.layers[0].mlp, LagunaMLP)
        self.assertIsInstance(model.layers[1].mlp, LagunaSparseMoE)
        self.assertEqual(model.layers[0].self_attn.g_proj.weight.shape, (2, 16))
        self.assertEqual(model.layers[1].self_attn.g_proj.weight.shape, (4, 16))

        logits = model(mx.array([[1, 2, 3]]))
        mx.eval(logits)
        self.assertEqual(logits.shape, (1, 3, 32))
        self.assertTrue(bool(mx.all(mx.isfinite(logits))))

    def test_router_correction_bias_changes_selection_only(self):
        probabilities = mx.array([[0.9, 0.8, 0.1]], dtype=mx.float32)
        logits = mx.log(probabilities / (1 - probabilities))
        correction = mx.array([0.0, 0.0, 1.0], dtype=mx.float32)
        indices, weights = router_select(
            logits, correction, top_k=2, normalize=True, output_dtype=mx.float16
        )
        mx.eval(indices, weights)

        selected = indices.tolist()[0]
        selected_weights = dict(zip(selected, weights.tolist()[0]))
        self.assertEqual(set(selected), {0, 2})
        self.assertEqual(weights.dtype, mx.float16)
        self.assertAlmostEqual(selected_weights[0], 0.9, places=3)
        self.assertAlmostEqual(selected_weights[2], 0.1, places=3)

    def test_attention_gate_is_per_head_float32_softplus(self):
        output = mx.ones((1, 1, 6), dtype=mx.float16)
        gate_logits = mx.array(
            [[[0.0, math.log(math.exp(2.0) - 1.0)]]], dtype=mx.float16
        )
        gated = apply_attention_gate(
            output, gate_logits, num_heads=2, head_dim=3, per_head=True
        )
        mx.eval(gated)

        expected = mx.array([[[math.log(2.0)] * 3 + [2.0] * 3]], dtype=mx.float16)
        self.assertEqual(gated.shape, output.shape)
        self.assertEqual(gated.dtype, output.dtype)
        self.assertTrue(bool(mx.allclose(gated, expected, atol=1e-3, rtol=1e-3)))

    def test_cache_mix_and_growth(self):
        model = Model(tiny_args())
        cache = model.make_cache()
        self.assertIsInstance(cache[0], KVCache)
        self.assertIsInstance(cache[1], RotatingKVCache)
        self.assertEqual(cache[1].max_size, 4)
        self.assertEqual(cache[1].keep, 0)

        model(mx.array([[1, 2, 3]]), cache=cache)
        for token in (4, 5, 6):
            model(mx.array([[token]]), cache=cache)
        mx.eval(cache[0].state, cache[1].state)
        self.assertEqual(cache[0].offset, 6)
        self.assertEqual(cache[0].size(), 6)
        self.assertEqual(cache[1].offset, 6)
        self.assertEqual(cache[1].size(), 4)

    def test_sanitizer_stacks_experts_in_numeric_order_and_maps_keys(self):
        model = Model(tiny_args())
        prefix = "model.layers.1.mlp"
        weights = {
            f"{prefix}.gate.weight": mx.ones((3, 16)),
            f"{prefix}.experts.e_score_correction_bias": mx.arange(3, dtype=mx.float32),
            f"{prefix}.shared_expert.gate_proj.weight": mx.ones((8, 16)),
        }
        # Insert out of order to prove stacking uses expert indices, not dict order.
        for expert in (2, 0, 1):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                shape = (16, 8) if projection == "down_proj" else (8, 16)
                weights[f"{prefix}.experts.{expert}.{projection}.weight"] = mx.full(
                    shape, expert
                )

        sanitized = model.sanitize(weights)
        stacked = sanitized[f"{prefix}.switch_mlp.gate_proj.weight"]
        mx.eval(stacked)
        self.assertEqual(stacked[:, 0, 0].tolist(), [0, 1, 2])
        self.assertEqual(stacked.shape, (3, 8, 16))
        self.assertEqual(
            sanitized[f"{prefix}.switch_mlp.down_proj.weight"].shape,
            (3, 16, 8),
        )
        self.assertIn(f"{prefix}.gate.gate.weight", sanitized)
        self.assertIn(f"{prefix}.gate.e_score_correction_bias", sanitized)
        self.assertIn(f"{prefix}.share_expert.gate_proj.weight", sanitized)
        self.assertNotIn(f"{prefix}.experts.0.gate_proj.weight", sanitized)
        # Exercise the real fused SwitchGLU parameter contract, while allowing
        # unrelated model parameters to remain absent from this focused test.
        model.load_weights(list(sanitized.items()), strict=False)

    def test_sanitizer_rejects_partial_expert_sets_with_diagnostics(self):
        model = Model(tiny_args())
        prefix = "model.layers.1.mlp"

        partial_ids = {
            f"{prefix}.experts.{expert}.gate_proj.weight": mx.ones((1, 1))
            for expert in (0, 1)
        }
        with self.assertRaisesRegex(
            ValueError,
            r"layer 1: found 2 individual tensors.*gate_proj\.weight: 2/3"
            r".*experts\.2\.gate_proj\.weight.*up_proj\.weight: 0/3",
        ):
            model.sanitize(partial_ids)

        missing_projection = {}
        for expert in range(3):
            for projection in ("gate_proj", "up_proj"):
                missing_projection[f"{prefix}.experts.{expert}.{projection}.weight"] = (
                    mx.ones((1, 1))
                )
        with self.assertRaisesRegex(
            ValueError, r"found 6 individual tensors.*down_proj\.weight: 0/3"
        ):
            model.sanitize(missing_projection)

        partial_quantization = {}
        for expert in range(3):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                partial_quantization[
                    f"{prefix}.experts.{expert}.{projection}.weight"
                ] = mx.ones((1, 1))
        partial_quantization[f"{prefix}.experts.0.gate_proj.scales"] = mx.ones((1, 1))
        with self.assertRaisesRegex(ValueError, r"gate_proj\.scales: 1/3"):
            model.sanitize(partial_quantization)

        wrong_layer = {"model.layers.0.mlp.experts.0.gate_proj.weight": mx.ones((1, 1))}
        with self.assertRaisesRegex(
            ValueError, r"Unconsumed Laguna.*layers\.0.*experts\.0"
        ):
            model.sanitize(wrong_layer)

    def test_sanitizer_rejects_ambiguous_remaps(self):
        model = Model(tiny_args())
        prefix = "model.layers.1.mlp"
        weights = {
            f"{prefix}.gate.weight": mx.ones((3, 16)),
            f"{prefix}.gate.gate.weight": mx.ones((3, 16)),
        }
        with self.assertRaisesRegex(
            ValueError, r"both .*mlp\.gate\.weight.*mlp\.gate\.gate\.weight"
        ):
            model.sanitize(weights)

        shared = {
            f"{prefix}.shared_expert.gate_proj.weight": mx.ones((8, 16)),
            f"{prefix}.share_expert.gate_proj.weight": mx.ones((8, 16)),
        }
        with self.assertRaisesRegex(ValueError, r"both .*shared_expert.*share_expert"):
            model.sanitize(shared)

    def test_cached_decode_matches_full_forward_inside_window(self):
        model = Model(tiny_args())
        tokens = mx.array([[1, 2, 3, 4]])
        full = model(tokens)[:, -1]

        cache = model.make_cache()
        model(tokens[:, :3], cache=cache)
        cached = model(tokens[:, 3:], cache=cache)[:, -1]
        mx.eval(full, cached)
        self.assertTrue(bool(mx.allclose(full, cached, atol=2e-4, rtol=2e-4)))

    def test_cached_decode_matches_full_forward_after_window_evictions(self):
        model = Model(tiny_args())
        tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])

        cache = model.make_cache()
        model(tokens[:, :3], cache=cache)
        for end in (4, 5, 6, 7):
            full = model(tokens[:, :end])[:, -1]
            cached = model(tokens[:, end - 1 : end], cache=cache)[:, -1]
            mx.eval(full, cached)
            self.assertTrue(bool(mx.allclose(full, cached, atol=2e-4, rtol=2e-4)))

        full_chunk = model(tokens)[:, -2:]
        cached_chunk = model(tokens[:, 7:9], cache=cache)
        mx.eval(full_chunk, cached_chunk)
        self.assertEqual(cache[1].offset, 9)
        self.assertEqual(cache[1].size(), 4)
        self.assertTrue(
            bool(mx.allclose(full_chunk, cached_chunk, atol=2e-4, rtol=2e-4))
        )

    def test_config_guards_and_quantization_predicate(self):
        with self.assertRaisesRegex(ValueError, "one entry per hidden layer"):
            tiny_args(layer_types=["full_attention"])
        with self.assertRaisesRegex(ValueError, "divisible"):
            tiny_args(num_attention_heads_per_layer=[2, 3])
        with self.assertRaisesRegex(ValueError, "num_experts_per_tok"):
            tiny_args(num_experts_per_tok=4)
        with self.assertRaisesRegex(ValueError, "sliding_window"):
            tiny_args(sliding_window=0)
        with self.assertRaisesRegex(ValueError, "bias-free"):
            tiny_args(qkv_bias=True)
        with self.assertRaisesRegex(ValueError, "Rotary dimensions"):
            tiny_args(
                rope_parameters={
                    "full_attention": {
                        "rope_type": "default",
                        "rope_theta": 10000.0,
                        "partial_rotary_factor": 0.75,
                    },
                    "sliding_attention": {
                        "rope_type": "default",
                        "rope_theta": 10000.0,
                    },
                }
            )

        predicate = Model(tiny_args()).quant_predicate
        self.assertFalse(predicate("model.layers.1.mlp.gate.gate", None))
        self.assertFalse(predicate("model.layers.0.input_layernorm", None))
        self.assertFalse(predicate("model.layers.1.mlp.gate.router_bias", None))
        self.assertTrue(predicate("model.layers.0.mlp.gate_proj", None))
        self.assertTrue(predicate("model.layers.0.self_attn.q_proj", None))


if __name__ == "__main__":
    unittest.main()
