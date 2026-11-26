# Copyright © 2026 Apple Inc.

import unittest
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.layers import attention
from mlx_lm.layers.attention import create_causal_mask
from mlx_lm.models import kimi_k31
from mlx_lm.models.cache import make_prompt_cache

# Layers 4 and 6 (1-based) use DSA, the others KDA. index_topk is small, so the
# indexer selects keys in short prompts.
CONFIG = {
    "model_type": "kimi_k31",
    "text_config": {
        "model_type": "kimi_linear",
        "vocab_size": 256,
        "hidden_size": 64,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 96,
        "rms_norm_eps": 1e-5,
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "linear_attn_config": {
            "kda_layers": [1, 2, 3, 5],
            "full_attn_layers": [4, 6],
            "num_heads": 2,
            "head_dim": 32,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "use_full_rank_gate": True,
        },
        "num_experts": 8,
        "moe_intermediate_size": 32,
        "q_lora_rank": 32,
        "kv_lora_rank": 32,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "num_experts_per_token": 2,
        "num_shared_experts": 1,
        "first_k_dense_replace": 1,
        "routed_expert_hidden_size": 32,
        "latent_moe_use_norm": True,
        "attn_res_block_size": 2,
        "sparse_prefill": False,
        "dsa_config": {
            "dsa_layers": [4, 6],
            "index_topk": 8,
            "index_head_dim": 16,
            "index_num_heads": 4,
            "index_rope_dim": 0,
            "index_k_norm_type": "layernorm",
            "index_topk_freq": 1,
            "index_skip_topk_offset": 1,
        },
    },
}


def make_model(**text):
    config = {**CONFIG, "text_config": {**CONFIG["text_config"], **text}}
    model = kimi_k31.Model(kimi_k31.ModelArgs.from_dict(config))
    mx.eval(model.parameters())
    return model


def prefill(model, tokens, chunk):
    cache = make_prompt_cache(model)
    outs = []
    for s in range(0, tokens.shape[1], chunk):
        outs.append(model(tokens[:, s : s + chunk], cache=cache))
    return mx.concatenate(outs, axis=1), cache


def set_sparse(model):
    for layer in model.layers:
        if layer.is_dsa:
            layer.self_attn.sparse_prefill = True
            layer.self_attn.sparse_prefill_min_context = 0


class TestKimiK31(unittest.TestCase):
    def setUp(self):
        # Tiny random models: GPU fp32 matmul rounding is larger than the differences we check.
        self._device = mx.default_device()
        mx.set_default_device(mx.cpu)
        mx.random.seed(0)

    def tearDown(self):
        mx.set_default_device(self._device)

    def test_dsa_layers_are_one_based(self):
        model = make_model()
        dsa = [i for i, layer in enumerate(model.layers) if layer.is_dsa]
        self.assertEqual(dsa, [3, 5])
        for i, layer in enumerate(model.layers):
            self.assertEqual(hasattr(layer.self_attn, "indexer"), i in dsa)

    def test_chunked_prefill_matches_whole(self):
        model = make_model()
        tokens = mx.random.randint(0, 256, (1, 48))
        whole, _ = prefill(model, tokens, 48)
        chunked, _ = prefill(model, tokens, 16)
        self.assertLess(float(mx.abs(whole - chunked).max()), 1e-4)

    def test_sparse_prefill_matches_dense(self):
        model = make_model()
        tokens = mx.random.randint(0, 256, (1, 64))
        dense, _ = prefill(model, tokens, 16)
        set_sparse(model)
        layers = kimi_k31.layers
        with patch.object(
            layers, "sparse_attention", wraps=layers.sparse_attention
        ) as ran:
            sparse, _ = prefill(model, tokens, 16)
        self.assertTrue(ran.called)
        self.assertLess(float(mx.abs(dense - sparse).max()), 1e-4)

    def test_query_blocks_match(self):
        # The query-block loop must slice the output gate with the queries.
        model = make_model()
        set_sparse(model)
        tokens = mx.random.randint(0, 256, (1, 64))
        whole, _ = prefill(model, tokens, 32)
        with patch.object(attention, "SPARSE_QUERY_BLOCK", 12):
            blocked, _ = prefill(model, tokens, 32)
        self.assertLess(float(mx.abs(whole - blocked).max()), 1e-4)

    def test_decode_matches_prefill(self):
        model = make_model()
        tokens = mx.random.randint(0, 256, (1, 40))
        whole, _ = prefill(model, tokens, 40)
        _, cache = prefill(model, tokens[:, :24], 24)
        steps = [model(tokens[:, i : i + 1], cache=cache) for i in range(24, 40)]
        decoded = mx.concatenate(steps, axis=1)
        self.assertLess(float(mx.abs(whole[:, 24:] - decoded).max()), 1e-4)

    def test_indexer_matches_reference(self):
        # Reference: sum over heads of relu(q . k) * weight, with no RoPE.
        model = make_model()
        indexer = model.layers[3].self_attn.indexer
        n, topk = 32, indexer.index_topk
        x = mx.random.normal((1, n, 64))
        qr = mx.random.normal((1, n, 32))
        mask = create_causal_mask(n)

        q = indexer.wq_b(qr).reshape(n, indexer.n_heads, indexer.head_dim)
        k = indexer.k_norm(indexer.wk(x))[0]
        w = indexer.weights_proj(x)[0]
        scores = (mx.maximum(mx.einsum("shd,td->sht", q, k), 0) * w[..., None]).sum(1)
        scores = mx.where(mask, scores, -mx.inf)
        ref = mx.sort(mx.argpartition(scores, kth=-topk, axis=-1)[:, -topk:], -1)

        out = mx.sort(indexer(x, qr, mask)[0, 0], -1)
        # Rows with fewer than index_topk causal keys also select masked keys.
        self.assertTrue(bool(mx.array_equal(out[topk - 1 :], ref[topk - 1 :])))

    @unittest.skipIf(not mx.metal.is_available(), "The kernel needs the Metal backend")
    def test_fused_matches_gather_path(self):
        # The fused kernel must apply the output gate too. One layer is compared, as
        # bf16 rounding through the whole model is larger than a layer's difference.
        mx.set_default_device(mx.gpu)
        model = make_model(
            num_attention_heads=32,
            num_key_value_heads=32,
            kv_lora_rank=64,
            qk_rope_head_dim=16,
        )
        model.set_dtype(mx.bfloat16)
        model.eval()
        set_sparse(model)
        attn = model.layers[3].self_attn
        x = mx.random.normal((1, 48, 64)).astype(mx.bfloat16)
        mask = create_causal_mask(48)

        ran = []
        kernel = attention.fused_sparse_attention

        def record(*args):
            out = kernel(*args)
            ran.append(out is not None)
            return out

        with patch.object(attention, "fused_sparse_attention", record):
            fused = attn(x, mask=mask)
        self.assertTrue(ran and all(ran))
        attn.sparse_prefill_fused = False
        gather = attn(x, mask=mask)
        diff = mx.abs(fused - gather).astype(mx.float32)
        self.assertLess(float(diff.max()), 0.01)

    def test_unsupported_dsa_config(self):
        for key, value in (
            ("index_rope_dim", 64),
            ("index_topk_freq", 4),
            ("index_k_norm_type", "rmsnorm"),
        ):
            with self.subTest(key=key):
                dsa = {**CONFIG["text_config"]["dsa_config"], key: value}
                with self.assertRaises(ValueError):
                    make_model(dsa_config=dsa)

    def test_dsa_on_kda_layer_rejected(self):
        # Layer 3 (1-based) is a KDA layer.
        dsa = {**CONFIG["text_config"]["dsa_config"], "dsa_layers": [3, 6]}
        with self.assertRaisesRegex(ValueError, "full attention layers"):
            make_model(dsa_config=dsa)

    def test_missing_q_lora_rank_rejected(self):
        with self.assertRaisesRegex(ValueError, "q_lora_rank"):
            make_model(q_lora_rank=None)


if __name__ == "__main__":
    unittest.main()
