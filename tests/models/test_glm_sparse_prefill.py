# Copyright © 2026 Apple Inc.

import unittest
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.models import glm_moe_dsa as G
from mlx_lm.models.cache import make_prompt_cache

CONFIG = {
    "model_type": "glm_moe_dsa",
    "vocab_size": 256,
    "hidden_size": 64,
    "index_head_dim": 16,
    "index_n_heads": 8,
    "index_topk": 16,
    "intermediate_size": 128,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "n_shared_experts": 1,
    "n_routed_experts": 4,
    "routed_scaling_factor": 1.0,
    "kv_lora_rank": 32,
    "q_lora_rank": 32,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "qk_nope_head_dim": 16,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
    "n_group": 1,
    "topk_group": 1,
    "num_experts_per_tok": 2,
    "moe_layer_freq": 1,
    "first_k_dense_replace": 1,
    "max_position_embeddings": 1024,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    "attention_bias": False,
    "index_topk_pattern": "FSFSFS",
}


def _args():
    return G.ModelArgs.from_dict({**CONFIG, "sparse_prefill": False})


class TestGlmSparsePrefill(unittest.TestCase):
    def setUp(self):
        # Tiny random models: GPU fp32 matmul rounding is larger than the dense/sparse difference we check.
        self._device = mx.default_device()
        mx.set_default_device(mx.cpu)

    def tearDown(self):
        mx.set_default_device(self._device)

    def _prefill(self, model, tokens, chunk):
        cache = make_prompt_cache(model)
        outs = []
        for s in range(0, tokens.shape[1], chunk):
            outs.append(model(tokens[:, s : s + chunk], cache=cache))
        return mx.concatenate(outs, axis=1)

    def test_default_is_on(self):
        layers = G.Model(G.ModelArgs.from_dict(CONFIG)).model.layers
        self.assertTrue(all(layer.self_attn.sparse_prefill for layer in layers))

    def test_sparse_prefill_matches_dense(self):
        mx.random.seed(0)
        model = G.Model(_args())
        mx.eval(model.parameters())
        tokens = mx.random.randint(
            0, 256, (1, 128)
        )  # longer than index_topk, so sparse path is used
        dense = self._prefill(model, tokens, 32)
        for layer in model.model.layers:
            layer.self_attn.sparse_prefill = True
            layer.self_attn.sparse_prefill_min_context = 0
        sparse = self._prefill(model, tokens, 32)
        mx.eval(dense, sparse)
        self.assertLess(float(mx.abs(dense - sparse).max()), 1e-4)
        self.assertTrue(bool(mx.all(mx.argmax(dense, -1) == mx.argmax(sparse, -1))))

    def test_blocked_topk_matches_indexer(self):
        from mlx_lm.models.base import create_causal_mask

        model = G.Model(_args())
        mx.eval(model.parameters())
        indexer = model.model.layers[0].self_attn.indexer
        x = mx.random.normal((1, 64, 64))
        qr = mx.random.normal((1, 64, 32))
        mask = create_causal_mask(64)
        stock = indexer(x, qr, mask)
        blocked = G._blocked_topk(indexer, x, qr, mask, None, block=16)
        # Rows with fewer than index_topk causal keys also select masked keys.
        rows = slice(indexer.index_topk - 1, None)
        self.assertTrue(
            bool(
                mx.array_equal(
                    mx.sort(stock[..., rows, :], -1), mx.sort(blocked[..., rows, :], -1)
                )
            )
        )

    def test_short_context_is_unchanged(self):
        # With at most index_topk keys every key is selected, so the dense path runs.
        mx.random.seed(0)
        model = G.Model(_args())
        mx.eval(model.parameters())
        tokens = mx.random.randint(0, 256, (1, 16))
        dense = self._prefill(model, tokens, 16)
        for layer in model.model.layers:
            layer.self_attn.sparse_prefill = True
            layer.self_attn.sparse_prefill_min_context = 0
        sparse = self._prefill(model, tokens, 16)
        self.assertTrue(bool(mx.array_equal(dense, sparse)))

    def test_batch_uses_dense_path(self):
        mx.random.seed(0)
        model = G.Model(_args())
        mx.eval(model.parameters())
        tokens = mx.random.randint(0, 256, (2, 96))
        dense = self._prefill(model, tokens, 32)
        for layer in model.model.layers:
            layer.self_attn.sparse_prefill = True
            layer.self_attn.sparse_prefill_min_context = 0
        sparse = self._prefill(model, tokens, 32)
        self.assertTrue(bool(mx.array_equal(dense, sparse)))

    def test_min_context_uses_dense_path(self):
        mx.random.seed(0)
        model = G.Model(_args())
        mx.eval(model.parameters())
        tokens = mx.random.randint(0, 256, (1, 128))
        dense = self._prefill(model, tokens, 32)
        for layer in model.model.layers:
            layer.self_attn.sparse_prefill = True
            layer.self_attn.sparse_prefill_min_context = 129
        gated = self._prefill(model, tokens, 32)
        self.assertTrue(bool(mx.array_equal(dense, gated)))

    def test_query_blocks_match(self):
        mx.random.seed(0)
        model = G.Model(_args())
        mx.eval(model.parameters())
        for layer in model.model.layers:
            layer.self_attn.sparse_prefill = True
            layer.self_attn.sparse_prefill_min_context = 0
        tokens = mx.random.randint(0, 256, (1, 128))
        whole = self._prefill(model, tokens, 64)
        with patch.object(G, "_SPARSE_QUERY_BLOCK", 24):
            blocked = self._prefill(model, tokens, 64)
        self.assertTrue(bool(mx.array_equal(whole, blocked)))

    def test_decode_after_sparse_prefill(self):
        mx.random.seed(0)
        model = G.Model(_args())
        mx.eval(model.parameters())
        for layer in model.model.layers:
            layer.self_attn.sparse_prefill = True
            layer.self_attn.sparse_prefill_min_context = 0
        cache = make_prompt_cache(model)
        tokens = mx.random.randint(0, 256, (1, 150))
        for s in range(0, 150, 64):
            mx.eval(model(tokens[:, s : s + 64], cache=cache))
        out = model(mx.array([[5]]), cache=cache)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 1, 256))


if __name__ == "__main__":
    unittest.main()
