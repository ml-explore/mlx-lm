# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

import mlx_lm.models.mimo_v2_attention as mimo_attention
from mlx_lm.models import mimo_v2_flash
from mlx_lm.models.mimo_v2_attention import (
    decode_supported,
    mimo_decode_attention,
    mimo_prefill_attention,
    prefill_supported,
)

SCALE = 192**-0.5


def _rel_l2(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    return (mx.linalg.norm(a - b) / mx.maximum(mx.linalg.norm(b), 1e-9)).item()


def _inputs(batch, heads, kv_heads, length, keys):
    q = mx.random.normal((batch, heads, length, 192)).astype(mx.bfloat16)
    k = mx.random.normal((batch, kv_heads, keys, 192)).astype(mx.bfloat16)
    v = mx.random.normal((batch, kv_heads, keys, 128)).astype(mx.bfloat16)
    return q, k, v


def _reference(q, k, v, key_mask=None):
    """Float32 attention; the L queries are the last L of the S positions (causal)."""
    groups = q.shape[1] // k.shape[1]
    k = mx.repeat(k.astype(mx.float32), groups, axis=1)
    v = mx.repeat(v.astype(mx.float32), groups, axis=1)
    scores = (q.astype(mx.float32) @ k.transpose(0, 1, 3, 2)) * SCALE
    length, keys = q.shape[2], k.shape[2]
    rows = mx.arange(keys - length, keys)[:, None]
    visible = mx.arange(keys)[None, :] <= rows
    if key_mask is not None:
        visible = visible & key_mask
    scores = mx.where(visible, scores, -mx.inf)
    return mx.softmax(scores, axis=-1) @ v


@unittest.skipIf(
    not mx.metal.is_available(), "MiMo-V2 attention kernels are Metal only"
)
class TestMiMoV2Attention(unittest.TestCase):
    def test_prefill_matches_reference(self):
        mx.random.seed(0)
        for length, keys in [(77, 77), (130, 400), (256, 1000)]:
            q, k, v = _inputs(1, 32, 2, length, keys)
            self.assertTrue(prefill_supported(q, k, v))
            out = mimo_prefill_attention(q, k, v, SCALE)
            self.assertEqual(out.shape, (1, 32, length, 128))
            self.assertLess(_rel_l2(out, _reference(q, k, v)), 1e-2)

    def test_prefill_split_launches_match_one_launch(self):
        mx.random.seed(1)
        q, k, v = _inputs(1, 16, 1, 200, 500)
        whole = mimo_prefill_attention(q, k, v, SCALE)
        limit = mimo_attention._MAX_SCORES_PER_DISPATCH
        mimo_attention._MAX_SCORES_PER_DISPATCH = 1
        try:
            split = mimo_prefill_attention(q, k, v, SCALE)
        finally:
            mimo_attention._MAX_SCORES_PER_DISPATCH = limit
        self.assertTrue(mx.array_equal(whole, split))

    def test_decode_matches_reference(self):
        mx.random.seed(2)
        for keys in [1, 100, 3000]:
            q, k, v = _inputs(2, 32, 2, 1, keys)
            self.assertTrue(decode_supported(q, k, v))
            out = mimo_decode_attention(q, k, v, SCALE)
            self.assertEqual(out.shape, (2, 32, 1, 128))
            self.assertLess(_rel_l2(out, _reference(q, k, v)), 1e-2)

    def test_decode_honours_boolean_mask(self):
        mx.random.seed(3)
        keys = 700
        q, k, v = _inputs(2, 16, 1, 1, keys)
        # Left padding: the second sequence starts 300 keys later.
        mask = mx.stack([mx.ones((keys,), mx.bool_), mx.arange(keys) >= 300]).reshape(
            2, 1, 1, keys
        )
        out = mimo_decode_attention(q, k, v, SCALE, mask)
        self.assertLess(_rel_l2(out, _reference(q, k, v, mask)), 1e-2)

    def test_decode_reads_a_strided_cache_view(self):
        mx.random.seed(4)
        q, k_buffer, v_buffer = _inputs(1, 16, 1, 1, 1024)
        keys = 600
        view = mimo_decode_attention(
            q, k_buffer[:, :, :keys], v_buffer[:, :, :keys], SCALE
        )
        copy = mimo_decode_attention(
            q,
            mx.contiguous(k_buffer[:, :, :keys]),
            mx.contiguous(v_buffer[:, :, :keys]),
            SCALE,
        )
        self.assertTrue(mx.array_equal(view, copy))

    def test_unsupported_calls_are_declined(self):
        q, k, v = _inputs(1, 16, 1, 1, 64)
        self.assertFalse(decode_supported(q.astype(mx.float16), k, v))
        self.assertFalse(decode_supported(q[..., :128], k[..., :128], v))
        q8, k8, v8 = _inputs(1, 8, 1, 1, 64)  # 8 query heads per KV head
        self.assertFalse(decode_supported(q8, k8, v8))
        self.assertFalse(decode_supported(q, k, v, mx.ones((1, 1, 1, 64))))

    def test_model_matches_the_unfused_path(self):
        config = {
            "model_type": "mimo_v2_flash",
            "num_experts_per_tok": 2,
            "hybrid_layer_pattern": [0, 1],
            "moe_layer_freq": [0, 1],
            "add_swa_attention_sink_bias": True,
            "add_full_attention_sink_bias": False,
            "sliding_window_size": 32,
            "vocab_size": 256,
            "hidden_size": 256,
            "intermediate_size": 256,
            "moe_intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 1,
            "n_shared_experts": 1,
            "n_routed_experts": 4,
            "routed_scaling_factor": None,
            "topk_method": "noaux_tc",
            "scoring_func": "sigmoid",
            "norm_topk_prob": True,
            "n_group": 1,
            "topk_group": 1,
            "max_position_embeddings": 1000,
            "layernorm_epsilon": 1e-5,
            "rope_theta": 1000.0,
            "swa_rope_theta": 1000.0,
            "swa_num_attention_heads": 16,
            "swa_num_key_value_heads": 1,
            "head_dim": 192,
            "v_head_dim": 128,
            "swa_head_dim": 192,
            "swa_v_head_dim": 128,
            "partial_rotary_factor": 0.334,
        }
        mx.random.seed(5)
        model = mimo_v2_flash.Model(mimo_v2_flash.ModelArgs.from_dict(config))
        model.set_dtype(mx.bfloat16)
        model.eval()
        prompt = mx.random.randint(0, 256, (1, 50))
        steps = [mx.array([[7]]), mx.array([[11]]), mx.array([[13]])]

        calls = {"prefill": 0, "decode": 0}
        prefill, decode = (
            mimo_v2_flash.mimo_prefill_attention,
            mimo_v2_flash.mimo_decode_attention,
        )

        def counting_prefill(*args):
            calls["prefill"] += 1
            return prefill(*args)

        def counting_decode(*args):
            calls["decode"] += 1
            return decode(*args)

        def run():
            cache = model.make_cache()
            outputs = [model(prompt, cache=cache)[:, -1]]
            for token in steps:
                outputs.append(model(token, cache=cache)[:, -1])
            return mx.concatenate(outputs)

        mimo_v2_flash.mimo_prefill_attention = counting_prefill
        mimo_v2_flash.mimo_decode_attention = counting_decode
        try:
            fused = run()
        finally:
            mimo_v2_flash.mimo_prefill_attention = prefill
            mimo_v2_flash.mimo_decode_attention = decode
        self.assertEqual(calls, {"prefill": 1, "decode": len(steps)})

        supports = mimo_v2_flash.prefill_supported, mimo_v2_flash.decode_supported
        mimo_v2_flash.prefill_supported = lambda *args: False
        mimo_v2_flash.decode_supported = lambda *args: False
        try:
            stock = run()
        finally:
            mimo_v2_flash.prefill_supported, mimo_v2_flash.decode_supported = supports
        self.assertLess(_rel_l2(fused, stock), 2e-2)


if __name__ == "__main__":
    unittest.main()
