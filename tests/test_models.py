# Copyright © 2024 Apple Inc.
import copy
import importlib
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from mlx_lm.models import rope_utils
from mlx_lm.models.base import create_causal_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache
from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_ops
from mlx_lm.models.ssm import ssm_attn, ssm_update


class TestModels(unittest.TestCase):
    def test_kv_cache(self):
        cache = KVCache()

        k = mx.ones((1, 4, 1, 32), mx.float16)
        v = mx.ones((1, 4, 1, 32), mx.float16)

        k_up, v_up = cache.update_and_fetch(k, v)
        self.assertTrue(mx.array_equal(k_up, k))
        self.assertTrue(mx.array_equal(v_up, v))
        self.assertEqual(cache.offset, 1)

        k = mx.ones((1, 4, cache.step, 32), mx.float16)
        v = mx.ones((1, 4, cache.step, 32), mx.float16)
        k_up, v_up = cache.update_and_fetch(k, v)

        expected = mx.ones((1, 4, cache.step + 1, 32), mx.float16)
        self.assertTrue(mx.array_equal(k_up, expected))
        self.assertTrue(mx.array_equal(v_up, expected))
        self.assertEqual(cache.offset, cache.step + 1)

    def test_rotating_kv_cache(self):
        b, h, d = 1, 2, 32
        cache = RotatingKVCache(max_size=8)

        k = mx.random.uniform(shape=(b, h, 2, d))
        v = mx.random.uniform(shape=(b, h, 2, d))

        k_up, v_up = cache.update_and_fetch(k, v)
        self.assertTrue(mx.array_equal(k_up, k))
        self.assertTrue(mx.array_equal(v_up, v))
        self.assertEqual(cache.offset, 2)

        k = mx.random.uniform(shape=(b, h, 5, d))
        v = mx.random.uniform(shape=(b, h, 5, d))
        k_up, v_up = cache.update_and_fetch(k, v)
        self.assertTrue(mx.array_equal(k_up[..., 2:, :], k))
        self.assertTrue(mx.array_equal(v_up[..., 2:, :], v))

        k = mx.random.uniform(shape=(b, h, 4, d))
        v = mx.random.uniform(shape=(b, h, 4, d))
        k_up, v_up = cache.update_and_fetch(k, v)
        self.assertTrue(mx.array_equal(k_up[..., -4:, :], k))
        self.assertTrue(mx.array_equal(v_up[..., -4:, :], v))

        idx = 0
        for _ in range(10):
            k = mx.random.uniform(shape=(b, h, 1, d))
            v = mx.random.uniform(shape=(b, h, 1, d))
            k_up, v_up = cache.update_and_fetch(k, v)
            self.assertTrue(mx.array_equal(k_up[..., idx : idx + 1, :], k))
            self.assertTrue(mx.array_equal(v_up[..., idx : idx + 1, :], v))
            idx += 1
            idx %= 8

        # Try with nonzero keep
        cache = RotatingKVCache(max_size=8, keep=2)

        # Check a large update
        k = mx.random.uniform(shape=(b, h, 20, d))
        v = mx.random.uniform(shape=(b, h, 20, d))
        k_up, v_up = cache.update_and_fetch(k, v)
        self.assertTrue(mx.array_equal(k_up, k))
        self.assertTrue(mx.array_equal(v_up, v))

        # A bunch of small updates
        self.assertEqual(cache.offset, 20)
        idx = 2
        for i in range(10):
            k = mx.random.uniform(shape=(b, h, 1, d))
            v = mx.random.uniform(shape=(b, h, 1, d))
            k_up, v_up = cache.update_and_fetch(k, v)
            self.assertTrue(mx.array_equal(k_up[..., idx : idx + 1, :], k))
            self.assertTrue(mx.array_equal(v_up[..., idx : idx + 1, :], v))
            self.assertEqual(cache.offset, 21 + i)
            idx += 1
            if idx >= 8:
                idx = 2

    def test_rotating_kv_cache_chat_mode(self):
        # Test that the rotating kv cache can handle
        # alternating prompt/prefill with generation
        d = 4
        h = 2
        cache = RotatingKVCache(max_size=18)

        x = mx.random.uniform(shape=(1, h, 8, d))
        k, v = cache.update_and_fetch(x, x)
        self.assertEqual(k.shape[2], 8)
        self.assertEqual(cache.offset, 8)

        x = mx.random.uniform(shape=(1, h, 1, d))
        k, v = cache.update_and_fetch(x, x)
        self.assertEqual(k.shape[2], 9)
        self.assertEqual(cache.offset, 9)
        self.assertTrue(mx.allclose(x, k[..., 8:9, :]))

        x = mx.random.uniform(shape=(1, h, 2, d))
        k, v = cache.update_and_fetch(x, x)
        self.assertEqual(k.shape[2], 11)
        self.assertEqual(cache.offset, 11)
        self.assertTrue(mx.allclose(x, k[..., 9:11, :]))

        x = mx.random.uniform(shape=(1, h, 3, d))
        k, v = cache.update_and_fetch(x, x)
        self.assertEqual(k.shape[2], 14)
        self.assertEqual(cache.offset, 14)
        self.assertTrue(mx.allclose(x, k[..., 11:14, :]))

        x = mx.random.uniform(shape=(1, h, 6, d))
        k, v = cache.update_and_fetch(x, x)
        self.assertEqual(cache.offset, 20)
        self.assertTrue(mx.allclose(x, k[..., -6:, :]))

        x = mx.random.uniform(shape=(1, h, 2, d))
        k, v = cache.update_and_fetch(x, x)
        self.assertEqual(cache.offset, 22)
        self.assertTrue(mx.allclose(x, k[..., -2:, :]))

    def test_causal_mask_padding(self):
        right_padding = mx.array([2, 1, 0])
        mask = create_causal_mask(3, right_padding=right_padding)

        causal_mask = create_causal_mask(3)
        self.assertTrue(
            mx.array_equal(mask[0, 0], causal_mask & mx.array([True, False, False]))
        )
        self.assertTrue(
            mx.array_equal(mask[1, 0], causal_mask & mx.array([True, True, False]))
        )
        self.assertTrue(mx.array_equal(mask[2, 0], causal_mask))

        left_padding = mx.array([2, 1, 0])
        mask = create_causal_mask(3, left_padding=left_padding)

        self.assertTrue(
            mx.array_equal(mask[0, 0], causal_mask & mx.array([False, False, True]))
        )
        self.assertTrue(
            mx.array_equal(mask[1, 0], causal_mask & mx.array([False, True, True]))
        )
        self.assertTrue(mx.array_equal(mask[2, 0], causal_mask))

    def test_mask_with_window(self):
        mask = create_causal_mask(5, 0, window_size=3)
        expected_sums = mx.array([1, 2, 3, 3, 3])
        sums = mask.sum(axis=1)
        self.assertTrue(mx.array_equal(sums, expected_sums))

        mask = create_causal_mask(5, 1, window_size=3)
        self.assertEqual(mask.shape, (5, 6))
        expected_sums = mx.array([2, 3, 3, 3, 3])
        sums = mask.sum(axis=1)
        self.assertTrue(mx.array_equal(sums, expected_sums))

        mask = create_causal_mask(5, 2, window_size=3)
        self.assertEqual(mask.shape, (5, 7))
        expected_sums = mx.array([3, 3, 3, 3, 3])
        sums = mask.sum(axis=1)
        self.assertTrue(mx.array_equal(sums, expected_sums))

    def test_llama_model_sliding_attention(self):
        from mlx_lm.models import llama

        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=64,
            num_hidden_layers=4,
            intermediate_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=128,
            sliding_window=4,
            layer_types=[
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            tie_word_embeddings=False,
            rope_theta=10000.0,
        )
        model = llama.Model(args)

        tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
        out = model(tokens)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 5, args.vocab_size))

        caches = model.make_cache()
        self.assertIsInstance(caches[0], KVCache)
        self.assertIsInstance(caches[1], RotatingKVCache)
        self.assertIsInstance(caches[2], RotatingKVCache)
        self.assertIsInstance(caches[3], KVCache)

        caches = model.make_cache()
        step = model(tokens[:, :2], cache=caches)
        mx.eval(step)
        step = model(tokens[:, 2:3], cache=caches)
        mx.eval(step)
        self.assertEqual(caches[0].offset, 3)
        self.assertEqual(caches[1].offset, 3)

    def test_rope(self):
        rope = rope_utils.initialize_rope(32, base=100, traditional=False)
        self.assertTrue(isinstance(rope, nn.RoPE))

        rope = rope_utils.initialize_rope(
            32,
            base=100,
            traditional=False,
            scaling_config={"rope_type": "linear", "factor": 10.0},
        )
        self.assertTrue(isinstance(rope, nn.RoPE))

        rope = rope_utils.initialize_rope(
            32,
            base=100,
            traditional=False,
            scaling_config={"rope_type": "llama3", "factor": 2.0},
        )
        self.assertTrue(isinstance(rope, rope_utils.Llama3RoPE))

    def test_quantized_sdpa(self):
        cache = KVCache()

        k = 1e-1 * mx.random.normal(shape=(1, 1, 256, 32))
        v = 1e-1 * mx.random.normal(shape=(1, 1, 256, 32))

        cache.update_and_fetch(k, v)
        quant_cache = cache.to_quantized(group_size=32, bits=8)

        k = 1e-1 * mx.random.normal(shape=(1, 1, 1, 32))
        v = 1e-1 * mx.random.normal(shape=(1, 1, 1, 32))

        k_up, v_up = cache.update_and_fetch(k, v)
        qk_up, qv_up = quant_cache.update_and_fetch(k, v)

        q = 1e-1 * mx.random.normal(shape=(1, 4, 257, 32))

        mask = "causal"
        out = scaled_dot_product_attention(
            q,
            k_up,
            v_up,
            cache=cache,
            mask=mask,
            scale=1.0,
        )
        qout = scaled_dot_product_attention(
            q,
            qk_up,
            qv_up,
            cache=quant_cache,
            mask=mask,
            scale=1.0,
        )
        self.assertTrue(mx.allclose(out, qout, rtol=1e-2, atol=1e-2))

    def model_test_runner(self, model, model_type, vocab_size, num_layers):
        self.assertEqual(len(model.layers), num_layers)
        self.assertEqual(model.model_type, model_type)

        for t in [mx.float32, mx.float16]:
            model.update(tree_map(lambda p: p.astype(t), model.parameters()))

            inputs = mx.array([[0, 1]])
            outputs = model(inputs)
            self.assertEqual(outputs.shape, (1, 2, vocab_size))
            self.assertEqual(outputs.dtype, t)

            cache = make_prompt_cache(model)
            outputs = model(inputs, cache=cache)
            self.assertEqual(outputs.shape, (1, 2, vocab_size))
            self.assertEqual(outputs.dtype, t)

            outputs = model(mx.argmax(outputs[0, -1:, :], keepdims=True), cache=cache)
            self.assertEqual(outputs.shape, (1, 1, vocab_size))
            self.assertEqual(outputs.dtype, t)

        # Test batch size > 1
        inputs = mx.array([[0, 1], [2, 3]])
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (2, 2, vocab_size))

        # Make sure the model can be copied / pickled
        copy.deepcopy(model)

    def test_llama(self):
        from mlx_lm.models import llama

        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
        )
        model = llama.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_lfm2(self):
        from mlx_lm.models import lfm2

        args = lfm2.ModelArgs(
            model_type="lfm2",
            hidden_size=1024,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            norm_eps=1e-5,
            vocab_size=10_000,
            full_attn_idxs=[0, 1, 2],
            rope_theta=10000,
            block_dim=1024,
            block_ffn_dim_multiplier=1.5,
            block_auto_adjust_ff_dim=True,
            block_ff_dim=2048,
            block_multiple_of=256,
            max_position_embeddings=1000,
            conv_bias=True,
            conv_L_cache=3,
        )
        model = lfm2.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_lfm2_moe(self):
        from mlx_lm.models import lfm2_moe

        args = lfm2_moe.ModelArgs(
            model_type="lfm2_moe",
            hidden_size=1024,
            intermediate_size=7168,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            norm_eps=1e-5,
            vocab_size=10_000,
            full_attn_idxs=[0, 1, 2],
            rope_theta=10000,
            max_position_embeddings=1000,
            conv_bias=True,
            conv_L_cache=3,
            moe_intermediate_size=1792,
            num_dense_layers=2,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            use_expert_bias=True,
        )
        model = lfm2_moe.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_bitnet(self):
        from mlx_lm.models import bitnet

        args = bitnet.ModelArgs(
            model_type="bitnet",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            num_key_value_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
        )
        model = bitnet.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_phi2(self):
        from mlx_lm.models import phi

        args = phi.ModelArgs()
        model = phi.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_phi3(self):
        from mlx_lm.models import phi3

        args = phi3.ModelArgs(
            model_type="phi3",
            hidden_size=3072,
            num_hidden_layers=32,
            intermediate_size=8192,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=32064,
        )
        model = phi3.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_gemma(self):
        from mlx_lm.models import gemma

        args = gemma.ModelArgs(
            model_type="gemma",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            head_dim=128,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
            num_key_value_heads=4,
        )
        model = gemma.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_mixtral(self):
        from mlx_lm.models import mixtral

        # Make a baby mixtral, because it will actually do the
        # eval
        args = mixtral.ModelArgs(
            model_type="mixtral",
            vocab_size=100,
            hidden_size=32,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts_per_tok=2,
            num_key_value_heads=2,
            num_local_experts=4,
        )
        model = mixtral.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    @unittest.skip("requires ai2-olmo")
    def test_olmo(self):
        from mlx_lm.models import olmo

        args = olmo.ModelArgs(
            model_type="olmo",
            d_model=1024,
            n_layers=4,
            mlp_hidden_size=2048,
            n_heads=2,
            vocab_size=10_000,
            embedding_size=10_000,
        )
        model = olmo.Model(args)
        self.model_test_runner(
            model,
            args.model_type,
            args.vocab_size,
            args.n_layers,
        )

    def test_qwen3_moe(self):
        from mlx_lm.models import qwen3_moe

        args = qwen3_moe.ModelArgs(
            model_type="qwen3_moe",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            num_key_value_heads=4,
            rms_norm_eps=1e-5,
            head_dim=128,
            vocab_size=10_000,
            decoder_sparse_step=1,
            mlp_only_layers=[],
            num_experts_per_tok=4,
            num_experts=16,
            moe_intermediate_size=1024,
            rope_theta=1000,
            max_position_embeddings=4096,
            tie_word_embeddings=False,
            norm_topk_prob=True,
        )
        model = qwen3_moe.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_qwen3(self):
        from mlx_lm.models import qwen3

        args = qwen3.ModelArgs(
            model_type="qwen3",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            num_key_value_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
            head_dim=128,
            max_position_embeddings=4096,
            tie_word_embeddings=False,
            rope_theta=1000,
        )
        model = qwen3.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_qwen2_moe(self):
        from mlx_lm.models import qwen2_moe

        args = qwen2_moe.ModelArgs(
            model_type="qwen2_moe",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
            num_experts_per_tok=4,
            num_experts=16,
            moe_intermediate_size=1024,
            shared_expert_intermediate_size=2048,
        )
        model = qwen2_moe.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_qwen2(self):
        from mlx_lm.models import qwen2

        args = qwen2.ModelArgs(
            model_type="qwen2",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            num_key_value_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
        )
        model = qwen2.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_qwen(self):
        from mlx_lm.models import qwen

        args = qwen.ModelArgs(
            model_type="qwen",
        )
        model = qwen.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_plamo(self):
        from mlx_lm.models import plamo

        args = plamo.ModelArgs(
            model_type="plamo",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=8,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
        )
        model = plamo.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_plamo2(self):
        from mlx_lm.models import plamo2

        args = plamo2.ModelArgs(
            model_type="plamo2",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=8,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
        )
        model = plamo2.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_stablelm(self):
        from mlx_lm.models import stablelm

        args = stablelm.ModelArgs(
            model_type="stablelm",
            vocab_size=10_000,
            hidden_size=1024,
            num_attention_heads=4,
            num_hidden_layers=4,
            num_key_value_heads=2,
            partial_rotary_factor=1.0,
            intermediate_size=2048,
            layer_norm_eps=1e-2,
            rope_theta=10_000,
            use_qkv_bias=False,
        )
        model = stablelm.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

        # StableLM 2
        args = stablelm.ModelArgs(
            model_type="stablelm",
            vocab_size=10000,
            hidden_size=512,
            num_attention_heads=8,
            num_hidden_layers=4,
            num_key_value_heads=2,
            partial_rotary_factor=0.25,
            intermediate_size=1024,
            layer_norm_eps=1e-5,
            rope_theta=10000,
            use_qkv_bias=True,
        )
        model = stablelm.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_starcoder2(self):
        from mlx_lm.models import starcoder2

        args = starcoder2.ModelArgs(
            model_type="starcoder2",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
        model = starcoder2.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_cohere(self):
        from mlx_lm.models import cohere

        args = cohere.ModelArgs(
            model_type="cohere",
        )
        model = cohere.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_dbrx(self):
        from mlx_lm.models import dbrx

        args = dbrx.ModelArgs(
            model_type="dbrx",
            d_model=1024,
            ffn_config={"ffn_hidden_size": 2048, "moe_num_experts": 4, "moe_top_k": 2},
            attn_config={"kv_n_heads": 2, "clip_qkv": True, "rope_theta": 10000},
            n_layers=4,
            n_heads=4,
            vocab_size=10_000,
        )
        model = dbrx.Model(args)
        self.model_test_runner(model, args.model_type, args.vocab_size, args.n_layers)

    def test_minicpm(self):
        from mlx_lm.models import minicpm

        args = minicpm.ModelArgs(
            model_type="minicpm",
            hidden_size=1024,
            dim_model_base=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            rms_norm_eps=1e-4,
            vocab_size=10000,
            num_key_value_heads=2,
            scale_depth=1.0,
            scale_emb=1.0,
        )
        model = minicpm.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_mamba(self):
        from mlx_lm.models import mamba

        args = mamba.ModelArgs(
            model_type="mamba",
            vocab_size=10000,
            use_bias=False,
            use_conv_bias=True,
            conv_kernel=4,
            hidden_size=768,
            num_hidden_layers=24,
            state_size=16,
            intermediate_size=1536,
            time_step_rank=48,
        )
        model = mamba.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_falcon_h1(self):
        from mlx_lm.models import falcon_h1

        args = falcon_h1.ModelArgs(
            model_type="falcon_h1",
            num_hidden_layers=12,
            vocab_size=10000,
        )
        model = falcon_h1.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_gpt2(self):
        from mlx_lm.models import gpt2

        args = gpt2.ModelArgs(
            model_type="gpt2",
            n_ctx=1024,
            n_embd=768,
            n_head=12,
            n_layer=12,
            n_positions=1024,
            layer_norm_epsilon=1e-5,
            vocab_size=50256,
        )
        model = gpt2.Model(args)
        self.model_test_runner(model, args.model_type, args.vocab_size, args.n_layer)

    def test_gpt_neox(self):
        from mlx_lm.models import gpt_neox

        args = gpt_neox.ModelArgs(
            model_type="gpt_neox",
            max_position_embeddings=2048,
            hidden_size=6144,
            num_attention_heads=64,
            num_hidden_layers=44,
            layer_norm_eps=1e-5,
            vocab_size=50432,
            rotary_emb_base=10_000,
            rotary_pct=0.25,
        )
        model = gpt_neox.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_openelm(self):
        from mlx_lm.models import openelm

        args = openelm.ModelArgs(
            model_type="openelm",
            ffn_dim_divisor=256,
            ffn_multipliers=[
                0.5,
                0.73,
                0.97,
                1.2,
                1.43,
                1.67,
                1.9,
                2.13,
                2.37,
                2.6,
                2.83,
                3.07,
                3.3,
                3.53,
                3.77,
                4.0,
            ],
            head_dim=64,
            model_dim=1280,
            normalize_qk_projections=True,
            num_kv_heads=[3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5],
            num_query_heads=[
                12,
                12,
                12,
                12,
                12,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                20,
                20,
                20,
                20,
            ],
            num_transformer_layers=16,
            vocab_size=32000,
        )

        model = openelm.Model(args)
        self.model_test_runner(
            model,
            args.model_type,
            args.vocab_size,
            len(args.ffn_multipliers),
        )

    def test_internlm2(self):
        from mlx_lm.models import internlm2

        args = internlm2.ModelArgs(
            model_type="internlm2",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=10000,
        )
        model = internlm2.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_llama3_1(self):
        from mlx_lm.models import llama

        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
            max_position_embeddings=128,
            mlp_bias=False,
            num_key_value_heads=2,
            rope_scaling={
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
        )
        model = llama.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_deepseek(self):
        from mlx_lm.models import deepseek

        args = deepseek.ModelArgs(
            model_type="deepseek",
            vocab_size=1024,
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
        )
        model = deepseek.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_deepseek_v2(self):
        from mlx_lm.models import deepseek_v2

        args = deepseek_v2.ModelArgs(
            model_type="deepseek_v2",
            vocab_size=1024,
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            kv_lora_rank=4,
            q_lora_rank=4,
            qk_rope_head_dim=32,
            v_head_dim=16,
            qk_nope_head_dim=32,
            rope_scaling={
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 40,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 4096,
                "type": "yarn",
            },
        )
        model = deepseek_v2.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_deepseek_v3(self):
        from mlx_lm.models import deepseek_v3

        args = deepseek_v3.ModelArgs(
            model_type="deepseek_v3",
            vocab_size=1024,
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            n_routed_experts=4,
            n_group=2,
            topk_group=1,
            num_experts_per_tok=2,
            n_shared_experts=1,
            kv_lora_rank=4,
            q_lora_rank=4,
            qk_rope_head_dim=32,
            v_head_dim=16,
            qk_nope_head_dim=32,
            rope_scaling={
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 40,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 4096,
                "type": "yarn",
            },
        )
        model = deepseek_v3.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_gemma2(self):
        from mlx_lm.models import gemma2

        args = gemma2.ModelArgs(
            model_type="gemma2",
            hidden_size=128,
            num_hidden_layers=4,
            intermediate_size=256,
            num_attention_heads=2,
            head_dim=32,
            rms_norm_eps=1e-4,
            vocab_size=1024,
            num_key_value_heads=2,
        )
        model = gemma2.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_gemma3_text(self):
        from mlx_lm.models import gemma3_text

        args = gemma3_text.ModelArgs(
            model_type="gemma3_text",
            hidden_size=128,
            num_hidden_layers=12,
            intermediate_size=256,
            num_attention_heads=4,
            head_dim=32,
            rms_norm_eps=1e-4,
            num_key_value_heads=1,
            sliding_window=1024,
            sliding_window_pattern=6,
        )
        model = gemma3_text.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_gpt_bigcode(self):
        from mlx_lm.models import gpt_bigcode

        args = gpt_bigcode.ModelArgs(
            model_type="gpt_bigcode",
            n_embd=128,
            n_layer=128,
            n_inner=256,
            n_head=4,
            n_positions=1000,
            layer_norm_epsilon=1e-5,
            vocab_size=1024,
        )
        model = gpt_bigcode.Model(args)
        self.model_test_runner(model, args.model_type, args.vocab_size, args.n_layer)

    def test_nemotron(self):
        from mlx_lm.models import nemotron

        args = nemotron.ModelArgs(
            model_type="nemotron",
            hidden_size=128,
            hidden_act="gelu",
            num_hidden_layers=4,
            intermediate_size=256,
            num_attention_heads=4,
            norm_eps=1e-5,
            vocab_size=1024,
            num_key_value_heads=2,
        )
        model = nemotron.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_phi3small(self):
        from mlx_lm.models import phi3small

        args = phi3small.ModelArgs(
            model_type="phi3small",
            hidden_size=128,
            dense_attention_every_n_layers=2,
            ff_intermediate_size=256,
            gegelu_limit=1.0,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            layer_norm_epsilon=1e-4,
            vocab_size=1000,
        )
        model = phi3small.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_phimoe(self):
        from mlx_lm.models import phimoe

        args = phimoe.ModelArgs(
            model_type="phimoe",
            vocab_size=320,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            rope_scaling={
                "long_factor": [1.0] * 16,
                "long_mscale": 1.243163121016122,
                "original_max_position_embeddings": 4096,
                "short_factor": [1.0] * 16,
                "short_mscale": 1.243163121016122,
                "type": "longrope",
            },
        )
        model = phimoe.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_recurrent_gemma(self):
        from mlx_lm.models import recurrent_gemma

        args = recurrent_gemma.ModelArgs(
            model_type="recurrent_gemma",
            hidden_size=128,
            attention_bias=False,
            conv1d_width=3,
            intermediate_size=256,
            logits_soft_cap=1.0,
            num_attention_heads=4,
            num_hidden_layers=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-4,
            rope_theta=1000,
            attention_window_size=1024,
            vocab_size=1000,
            block_types=["recurrent", "recurrent", "attention"],
        )
        model = recurrent_gemma.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_hunyuan(self):
        from mlx_lm.models import hunyuan

        args = hunyuan.ModelArgs(
            model_type="hunyuan",
            hidden_size=128,
            attention_bias=False,
            intermediate_size=256,
            num_attention_heads=4,
            num_hidden_layers=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-4,
            rope_theta=1000,
            vocab_size=1000,
            moe_topk=2,
            num_experts=2,
            num_shared_expert=1,
            use_mixed_mlp_moe=True,
            use_qk_norm=True,
            rope_scaling={
                "alpha": 1000.0,
                "factor": 1.0,
                "type": "dynamic",
            },
            use_cla=True,
            cla_share_factor=2,
        )
        model = hunyuan.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_hunyuan_v1_dense(self):
        from mlx_lm.models import hunyuan_v1_dense

        args = hunyuan_v1_dense.ModelArgs(
            model_type="hunyuan_v1_dense",
            hidden_size=128,
            attention_bias=False,
            intermediate_size=256,
            num_attention_heads=4,
            num_hidden_layers=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-4,
            rope_theta=1000,
            vocab_size=1000,
            use_qk_norm=True,
            rope_scaling={
                "alpha": 1000.0,
                "factor": 1.0,
                "type": "dynamic",
                "beta_fast": 32,
                "beta_slow": 1,
                "mscale": 1.0,
                "mscale_all_dim": 0.0,
                "original_max_position_embeddings": 8192,
            },
            max_position_embeddings=32768,
        )
        model = hunyuan_v1_dense.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_olmo2(self):
        from mlx_lm.models import olmo2

        args = olmo2.ModelArgs(
            model_type="olmo2",
            hidden_size=128,
            attention_bias=False,
            intermediate_size=256,
            num_attention_heads=4,
            num_hidden_layers=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-4,
            rope_theta=1000,
            vocab_size=1000,
        )
        model = olmo2.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_exaone(self):
        from mlx_lm.models import exaone

        args = exaone.ModelArgs(
            model_type="exaone",
            hidden_size=128,
            num_layers=4,
            intermediate_size=256,
            num_attention_heads=8,
            num_key_value_heads=2,
            vocab_size=1000,
            layer_norm_epsilon=1e-4,
            rope_theta=10000,
        )
        model = exaone.Model(args)
        self.model_test_runner(model, args.model_type, args.vocab_size, args.num_layers)

    def test_cohere2(self):
        from mlx_lm.models import cohere2

        args = cohere2.ModelArgs(
            model_type="cohere2",
            hidden_size=4096,
            head_dim=128,
            num_hidden_layers=40,
            sliding_window=4096,
            sliding_window_pattern=4,
        )
        model = cohere2.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_internlm3(self):
        from mlx_lm.models import internlm3

        args = internlm3.ModelArgs(
            model_type="internlm3",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
        )
        model = internlm3.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_smollm3(self):
        from mlx_lm.models import smollm3

        args = smollm3.ModelArgs(
            model_type="smollm3",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
        )
        model = smollm3.Model(args)
        self.model_test_runner(
            model, "smollm3", args.vocab_size, args.num_hidden_layers
        )

    def test_gpt_oss(self):
        from mlx_lm.models import gpt_oss

        args = gpt_oss.ModelArgs(
            model_type="gpt_oss",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=8,
            num_key_value_heads=2,
            num_local_experts=16,
            num_experts_per_tok=2,
            sliding_window=128,
            rope_theta=10000,
            vocab_size=10_000,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
        model = gpt_oss.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_all_models(self):
        test_configs = [
            {
                "model_type": "afm7",
                "vocab_size": 1000,
                "hidden_dim": 256,
                "num_layers": 16,
                "num_hidden_layers": 16,
                "num_kv_reuse_layers": 8,
                "num_heads": 8,
                "num_kv_heads": 4,
            },
            {
                "model_type": "apertus",
                "hidden_size": 128,
                "num_hidden_layers": 4,
                "intermediate_size": 256,
                "mlp_bias": True,
                "num_attention_heads": 8,
                "attention_bias": False,
                "rms_norm_eps": 1e-5,
                "vocab_size": 1000,
                "num_key_value_heads": 4,
                "max_position_embeddings": 1000,
                "rope_theta": 1000,
                "post_norm": True,
                "qk_norm": True,
                "tie_word_embeddings": False,
            },
            {
                "model_type": "baichuan_m1",
                "vocab_size": 1000,
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rope_theta": 1000,
                "sliding_window": 128,
                "sliding_window_layers": [0, 2],
                "conv_window": 2,
                "rms_norm_eps": 1e-5,
            },
            {
                "model_type": "bailing_moe",
                "hidden_size": 64,
                "intermediate_size": 128,
                "max_position_embeddings": 1000,
                "moe_intermediate_size": 128,
                "num_experts": 4,
                "num_shared_experts": 1,
                "norm_topk_prob": True,
                "num_attention_heads": 4,
                "num_experts_per_tok": 2,
                "num_hidden_layers": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-5,
                "rope_theta": 1000,
                "vocab_size": 1000,
                "first_k_dense_replace": 2,
            },
            {
                "model_type": "dots1",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "rms_norm_eps": 1e-5,
                "vocab_size": 1000,
                "max_position_embeddings": None,
                "num_key_value_heads": 2,
                "first_k_dense_replace": 1,
                "moe_intermediate_size": 64,
                "n_routed_experts": 4,
                "n_shared_experts": 1,
                "norm_topk_prob": True,
                "num_experts_per_tok": 1,
                "rope_theta": 1000,
                "routed_scaling_factor": 1.0,
            },
            {
                "hidden_size": 128,
                "intermediate_size": 128,
                "model_type": "ernie4_5",
                "max_position_embeddings": 1000,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": None,
                "num_hidden_layers": 4,
                "rms_norm_eps": 1e-5,
                "vocab_size": 1000,
                "rope_theta": 10000,
                "use_bias": False,
                "tie_word_embeddings": True,
            },
            {
                "hidden_size": 128,
                "intermediate_size": 128,
                "model_type": "ernie4_5_moe",
                "max_position_embeddings": 1000,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_hidden_layers": 4,
                "rms_norm_eps": 1e-5,
                "vocab_size": 1000,
                "rope_theta": 1000,
                "use_bias": False,
                "tie_word_embeddings": False,
                "moe_num_experts": 4,
            },
            {
                "model_type": "exaone4",
                "hidden_size": 128,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "vocab_size": 1000,
                "rms_norm_eps": 1e-5,
                "num_hidden_layers": 4,
                "max_position_embeddings": 1000,
                "rope_theta": 10000,
                "layer_norm_epsilon": 1e-5,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "tie_word_embeddings": False,
                "rope_scaling": None,
                "sliding_window": 8,
                "sliding_window_pattern": "LLGL",
            },
            {
                "model_type": "gemma3n",
                "num_hidden_layers": 4,
                "vocab_size": 1000,
                "text_config": {
                    "model_type": "gemma3n",
                    "hidden_size": 128,
                    "num_hidden_layers": 4,
                    "intermediate_size": 128,
                    "num_attention_heads": 4,
                    "head_dim": 32,
                    "rms_norm_eps": 1e-5,
                    "vocab_size": 1000,
                    "num_key_value_heads": 2,
                    "num_kv_shared_layers": 2,
                    "vocab_size_per_layer_input": 1000,
                    "sliding_window": 8,
                    "max_position_embeddings": 1000,
                    "rope_local_base_freq": 1.0,
                    "rope_theta": 1000.0,
                    "final_logit_softcapping": 1.0,
                    "layer_types": [
                        "sliding_attention",
                        "full_attention",
                        "sliding_attention",
                        "full_attention",
                    ],
                    "activation_sparsity_pattern": [0.5, 0.5, 0.5, 0.5],
                    "hidden_size_per_layer_input": 256,
                    "altup_num_inputs": 4,
                    "altup_coef_clip": 1.0,
                    "altup_correct_scale": True,
                    "altup_active_idx": 0,
                    "laurel_rank": 8,
                },
            },
            {
                "model_type": "glm4",
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "attention_bias": False,
                "head_dim": 64,
                "rms_norm_eps": 1e-5,
                "vocab_size": 1000,
                "num_key_value_heads": 2,
                "partial_rotary_factor": 0,
                "rope_theta": 1000,
            },
            {
                "model_type": "glm4_moe",
                "vocab_size": 1000,
                "hidden_size": 128,
                "intermediate_size": 128,
                "max_position_embeddings": 1000,
                "moe_intermediate_size": 128,
                "norm_topk_prob": True,
                "num_attention_heads": 4,
                "n_group": 2,
                "head_dim": 32,
                "topk_group": 1,
                "n_shared_experts": 1,
                "n_routed_experts": 4,
                "routed_scaling_factor": 1.0,
                "num_experts_per_tok": 2,
                "first_k_dense_replace": 1,
                "num_hidden_layers": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-5,
                "rope_theta": 1000,
                "rope_scaling": None,
                "use_qk_norm": True,
                "tie_word_embeddings": False,
                "attention_bias": False,
                "partial_rotary_factor": 0.0,
            },
            {
                "model_type": "granite",
                "hidden_size": 128,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "rms_norm_eps": 1e-5,
                "vocab_size": 1000,
                "logits_scaling": 1.0,
                "attention_multiplier": 1.0,
                "embedding_multiplier": 1.0,
                "residual_multiplier": 1.0,
                "max_position_embeddings": 1000,
                "num_key_value_heads": 2,
                "attention_bias": False,
                "mlp_bias": False,
                "rope_theta": 1000,
            },
            {
                "model_type": "granitemoe",
                "hidden_size": 128,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "rms_norm_eps": 1e-5,
                "vocab_size": 1000,
                "logits_scaling": 1.0,
                "attention_multiplier": 1.0,
                "embedding_multiplier": 1.0,
                "residual_multiplier": 1.0,
                "max_position_embeddings": 1000,
                "num_key_value_heads": 2,
                "attention_bias": False,
                "rope_theta": 1000,
                "num_local_experts": 4,
                "num_experts_per_tok": 2,
            },
            {
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "intermediate_size": 256,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-5,
                "vocab_size": 1000,
                "attention_bias": False,
                "head_dim": 64,
                "max_position_embeddings": 1000,
                "mlp_bias": False,
                "model_type": "helium",
                "rope_theta": 1000,
                "tie_word_embeddings": False,
            },
            {
                "text_config": {
                    "vocab_size": 1000,
                    "hidden_size": 128,
                    "intermediate_size": 128,
                    "moe_intermediate_size": 128,
                    "num_hidden_layers": 4,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "n_shared_experts": 1,
                    "n_routed_experts": 4,
                    "kv_lora_rank": 32,
                    "q_lora_rank": 32,
                    "qk_rope_head_dim": 8,
                    "v_head_dim": 16,
                    "qk_nope_head_dim": 16,
                },
                "model_type": "kimi_vl",
                "num_hidden_layers": 4,
                "vocab_size": 1000,
            },
            {
                "model_type": "lfm2-vl",
                "vocab_size": 1000,
                "num_hidden_layers": 4,
                "text_config": {
                    "model_type": "lfm2",
                    "vocab_size": 1000,
                    "hidden_size": 128,
                    "num_hidden_layers": 4,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "max_position_embeddings": 1000,
                    "norm_eps": 1e-5,
                    "conv_bias": False,
                    "conv_L_cache": 3,
                    "block_dim": 128,
                    "block_ff_dim": 128,
                    "block_multiple_of": 4,
                    "block_ffn_dim_multiplier": 2,
                    "block_auto_adjust_ff_dim": True,
                    "layer_types": ["full_attention", "", "full_attention", ""],
                    "rope_theta": 1000,
                },
            },
            {
                "model_type": "llama4",
                "text_config": {
                    "attention_bias": False,
                    "attention_chunk_size": 8,
                    "head_dim": 32,
                    "hidden_size": 128,
                    "interleave_moe_layer_step": 2,
                    "intermediate_size": 128,
                    "intermediate_size_mlp": 128,
                    "max_position_embeddings": 1000,
                    "model_type": "llama4",
                    "num_attention_heads": 4,
                    "num_experts_per_tok": 1,
                    "num_hidden_layers": 4,
                    "num_key_value_heads": 2,
                    "num_local_experts": 2,
                    "rms_norm_eps": 1e-4,
                    "rope_scaling": None,
                    "rope_theta": 1000,
                    "use_qk_norm": True,
                    "vocab_size": 1000,
                },
                "num_hidden_layers": 4,
                "vocab_size": 1000,
            },
            {
                "model_type": "longcat_flash",
                "attention_method": "MLA",
                "zero_expert_type": "identity",
                "hidden_size": 128,
                "ffn_hidden_size": 128,
                "moe_topk": 2,
                "expert_ffn_hidden_size": 128,
                "n_routed_experts": 2,
                "zero_expert_num": 2,
                "num_layers": 4,
                "num_hidden_layers": 4,
                "vocab_size": 1000,
                "max_position_embeddings": 1000,
                "num_attention_heads": 4,
                "kv_lora_rank": 16,
                "q_lora_rank": 16,
                "qk_rope_head_dim": 8,
                "qk_nope_head_dim": 8,
                "v_head_dim": 8,
                "routed_scaling_factor": 1.0,
                "rms_norm_eps": 1e-5,
                "rope_theta": 1000,
                "mla_scale_q_lora": True,
                "mla_scale_kv_lora": True,
                "attention_bias": False,
            },
            {
                "model_type": "mimo",
                "hidden_size": 128,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "rms_norm_eps": 1e-5,
                "vocab_size": 1000,
                "num_key_value_heads": 2,
            },
            {
                "model_type": "nemotron-nas",
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "rms_norm_eps": 1e-5,
                "vocab_size": 128256,
                "block_configs": [
                    {
                        "attention": {
                            "n_heads_in_group": 8,
                            "no_op": False,
                            "replace_with_linear": False,
                        },
                        "ffn": {
                            "ffn_mult": 1.3125,
                            "no_op": False,
                            "replace_with_linear": False,
                        },
                    },
                ]
                * 4,
            },
            {
                "model_type": "nemotron_h",
                "vocab_size": 1000,
                "hidden_size": 128,
                "intermediate_size": 128,
                "num_hidden_layers": 4,
                "max_position_embeddings": 1000,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "attention_bias": False,
                "mamba_num_heads": 8,
                "mamba_head_dim": 64,
                "mamba_proj_bias": False,
                "ssm_state_size": 128,
                "conv_kernel": 3,
                "n_groups": 4,
                "time_step_limit": [1.0, 2.0],
                "mlp_bias": False,
                "layer_norm_epsilon": 1e-4,
                "rms_norm_eps": 1e-5,
                "use_bias": True,
                "use_conv_bias": True,
                "residual_in_fp32": True,
                "hybrid_override_pattern": ["*", "M", "*", "M"],
            },
            {
                "model_type": "olmoe",
                "hidden_size": 128,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "rms_norm_eps": 1e-5,
                "vocab_size": 1000,
                "num_experts": 4,
                "num_experts_per_tok": 2,
            },
            {
                "model_type": "pixtral",
                "text_config": {
                    "model_type": "llama",
                    "hidden_size": 1024,
                    "num_hidden_layers": 4,
                    "intermediate_size": 2048,
                    "num_attention_heads": 4,
                    "rms_norm_eps": 1e-5,
                    "vocab_size": 1000,
                },
                "num_hidden_layers": 4,
                "vocab_size": 1000,
            },
            {
                "model_type": "qwen3_vl_moe",
                "text_config": {
                    "model_type": "qwen3_moe",
                    "hidden_size": 128,
                    "num_hidden_layers": 4,
                    "intermediate_size": 256,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "rms_norm_eps": 1e-5,
                    "head_dim": 32,
                    "vocab_size": 1000,
                    "decoder_sparse_step": 1,
                    "mlp_only_layers": [],
                    "num_experts_per_tok": 2,
                    "num_experts": 4,
                    "moe_intermediate_size": 128,
                    "rope_theta": 1000,
                    "max_position_embeddings": 1000,
                    "tie_word_embeddings": False,
                    "norm_topk_prob": True,
                },
                "num_hidden_layers": 4,
                "vocab_size": 1000,
            },
            {
                "model_type": "qwen3_vl",
                "text_config": {
                    "model_type": "qwen3",
                    "hidden_size": 128,
                    "num_hidden_layers": 4,
                    "intermediate_size": 256,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "rms_norm_eps": 1e-5,
                    "vocab_size": 1000,
                    "head_dim": 32,
                    "max_position_embeddings": 1000,
                    "tie_word_embeddings": False,
                    "rope_theta": 1000,
                },
                "num_hidden_layers": 4,
                "vocab_size": 1000,
            },
            {
                "model_type": "seed_oss",
                "hidden_size": 128,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "rms_norm_eps": 1e-5,
                "vocab_size": 1000,
                "num_key_value_heads": 2,
                "head_dim": 64,
            },
            {
                "model_type": "Klear",
                "hidden_size": 128,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "attention_bias": False,
                "mlp_only_layers": [0],
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 2,
                "n_shared_experts": 1,
                "moe_intermediate_size": 128,
                "rms_norm_eps": 1e-5,
                "vocab_size": 1000,
                "num_key_value_heads": 4,
                "rope_theta": 1000.0,
                "max_position_embeddings": 1000,
                "norm_topk_prob": True,
            },
            {
                "model_type": "lille-130m",
                "block_size": 128,
                "num_hidden_layers": 4,
                "n_layer": 4,
                "n_head": 4,
                "n_kv_heads": 4,
                "n_embd": 128,
                "vocab_size": 1000,
                "rope_theta": 1000,
                "layer_norm_eps": 1e-5,
            },
            {
                "model_type": "granitemoehybrid",
                "vocab_size": 1000,
                "hidden_size": 128,
                "intermediate_size": 128,
                "num_hidden_layers": 4,
                "max_position_embeddings": 1000,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "attention_bias": False,
                "embedding_multiplier": 1.0,
                "attention_multiplier": 1.0,
                "logits_scaling": 1.0,
                "residual_multiplier": 1.0,
                "num_local_experts": 8,
                "num_experts_per_tok": 2,
                "shared_intermediate_size": 128,
                "mamba_n_heads": 8,
                "mamba_d_head": 16,
                "mamba_proj_bias": False,
                "mamba_d_state": 128,
                "mamba_d_conv": 4,
                "mamba_n_groups": 1,
                "mamba_conv_bias": False,
                "layer_types": ["mamba", "attention", "mamba", "attention"],
                "rms_norm_eps": 1e-5,
                "rope_theta": 1000.0,
            },
            {
                "model_type": "glm",
                "hidden_size": 128,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "rms_norm_eps": 1e-5,
                "vocab_size": 1000,
                "head_dim": 32,
                "num_key_value_heads": 2,
            },
            {
                "model_type": "llama4_text",
                "hidden_size": 128,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "rms_norm_eps": 1e-5,
                "vocab_size": 1000,
                "head_dim": 32,
                "num_key_value_heads": 2,
                "intermediate_size_mlp": 128,
                "rope_theta": 1000.0,
                "head_dim": 8,
                "tie_word_embeddings": False,
                "no_rope_layers": [0, 0, 1, 1],
                "use_qk_norm": True,
            },
            {
                "model_type": "mamba2",
                "num_heads": 8,
                "head_dim": 16,
                "vocab_size": 1000,
                "hidden_size": 128,
                "intermediate_size": 128,
                "state_size": 32,
                "num_hidden_layers": 4,
                "layer_norm_epsilon": 1e-4,
                "conv_kernel": 3,
                "n_groups": 4,
                "use_bias": False,
                "use_conv_bias": False,
                "chunk_size": 32,
                "tie_word_embeddings": True,
                "time_step_limit": (0.01, 10),
                "time_step_rank": "auto",
            },
            {
                "model_type": "olmo3",
                "num_heads": 8,
                "head_dim": 16,
                "vocab_size": 1000,
                "hidden_size": 128,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "rope_theta": 1000,
                "num_hidden_layers": 8,
                "rms_norm_eps": 1e-4,
                "sliding_window": 128,
                "tie_word_embeddings": True,
                "max_position_embeddings": 1000,
            },
            {
                "model_type": "jamba",
                "hidden_size": 128,
                "intermediate_size": 128,
                "num_hidden_layers": 8,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "attn_layer_offset": 1,
                "attn_layer_period": 2,
                "expert_layer_offset": 1,
                "expert_layer_period": 2,
                "mamba_d_conv": 4,
                "mamba_d_state": 128,
                "mamba_expand": 128,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "rms_norm_eps": 1e-5,
                "max_position_embeddings": 1000,
                "vocab_size": 1000,
            },
            {
                "model_type": "nanochat",
                "hidden_size": 1280,
                "num_hidden_layers": 20,
                "vocab_size": 32,
                "intermediate_size": 128,
            },
            {
                "model_type": "minimax",
                "hidden_size": 128,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "max_position_embeddings": 1000,
                "num_experts_per_tok": 2,
                "num_local_experts": 8,
                "shared_intermediate_size": 128,
                "num_hidden_layers": 4,
                "rms_norm_eps": 1e-4,
                "rope_theta": 1000,
                "rotary_dim": 16,
                "vocab_size": 1000,
            },
            {
                "model_type": "bailing_moe_linear",
                "hidden_size": 1024,
                "num_hidden_layers": 4,
                "intermediate_size": 2048,
                "moe_intermediate_size": 1024,
                "num_experts_per_tok": 2,
                "num_experts": 4,
                "norm_topk_prob": True,
                "num_shared_experts": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "rms_norm_eps": 1e-5,
                "vocab_size": 10_000,
                "rope_theta": 1000,
                "first_k_dense_replace": 0,
                "layer_group_size": 2,
                "group_norm_size": 1,
                "max_position_embeddings": 1000,
            },
            {
                "model_type": "kimi_linear",
                "vocab_size": 1000,
                "hidden_size": 128,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "intermediate_size": 128,
                "head_dim": 32,
                "rope_theta": 100.0,
                "rms_norm_eps": 1e-6,
                "linear_attn_config": {
                    "num_heads": 8,
                    "head_dim": 32,
                    "kda_layers": [1],
                },
                "model_max_length": 1000,
                "num_experts": 2,
                "moe_intermediate_size": 128,
                "kv_lora_rank": 8,
            },
        ]
        for config in test_configs:
            model_type = config["model_type"]
            with self.subTest(f"Testing {model_type}", model_type=model_type):
                arch = importlib.import_module(f"mlx_lm.models.{model_type}")
                args = arch.ModelArgs.from_dict(config)
                model = arch.Model(args)
                self.model_test_runner(
                    model,
                    args.model_type,
                    config["vocab_size"],
                    config["num_hidden_layers"],
                )

    def test_ssm(self):
        for batch_size in [1, 2]:
            for n_group in [1, 4]:
                num_heads = 48
                head_dim = 64
                state_dim = 128

                hidden_states = mx.random.normal(
                    shape=(batch_size, 1, num_heads, head_dim)
                )
                B = mx.random.normal(shape=(batch_size, 1, n_group, state_dim))
                C = mx.random.normal(shape=(batch_size, 1, n_group, state_dim))
                dt = mx.random.normal(shape=(batch_size, 1, num_heads))
                dt_bias = mx.random.normal(shape=(num_heads,))
                A_log = mx.random.normal(shape=(num_heads,))
                D = mx.random.normal(shape=(num_heads,))
                state = mx.random.normal(
                    shape=(batch_size, num_heads, head_dim, state_dim)
                )

                out, out_state = ssm_attn(
                    hidden_states, A_log, B, C, D, dt, dt_bias, state
                )
                out_c, out_state_c = ssm_update(
                    hidden_states, A_log, B, C, D, dt, dt_bias, state
                )
                self.assertTrue(mx.allclose(out, out_c, atol=1e-4, rtol=1e-4))
                self.assertTrue(
                    mx.allclose(out_state, out_state_c, atol=1e-4, rtol=1e-4)
                )

    def test_ssm_masked(self):
        batch_size = 1
        n_group = 1
        num_heads = 48
        head_dim = 64
        state_dim = 128
        seq_len = 4
        pad = 2

        hidden_states = mx.random.normal(
            shape=(batch_size, seq_len + pad, num_heads, head_dim)
        )
        B = mx.random.normal(shape=(batch_size, seq_len + pad, n_group, state_dim))
        C = mx.random.normal(shape=(batch_size, seq_len + pad, n_group, state_dim))
        dt = mx.random.normal(shape=(batch_size, seq_len + pad, num_heads))
        dt_bias = mx.random.normal(shape=(num_heads,))
        A_log = mx.random.normal(shape=(num_heads,))
        D = mx.random.normal(shape=(num_heads,))
        out, out_state = ssm_attn(
            hidden_states[:, pad:],
            A_log,
            B[:, pad:],
            C[:, pad:],
            D,
            dt[:, pad:],
            dt_bias,
        )
        mask = mx.array([[False] * pad + [True] * seq_len])
        out_m, out_state_m = ssm_attn(
            hidden_states, A_log, B, C, D, dt, dt_bias, mask=mask
        )
        out_m = out_m[:, pad:]
        self.assertTrue(mx.allclose(out, out_m, atol=1e-4, rtol=1e-4))
        self.assertTrue(mx.allclose(out_state, out_state_m, atol=1e-4, rtol=1e-4))

    def test_gated_delta(self):
        mx.random.seed(0)
        for B in [1, 2]:
            for T in [1, 2]:
                Hk = 16
                Hv = 32
                Dk = 128
                Dv = 128

                q = mx.random.normal(shape=(B, T, Hk, Dk))
                k = mx.random.normal(shape=(B, T, Hk, Dk))
                v = mx.random.normal(shape=(B, T, Hv, Dv))
                g = mx.random.uniform(shape=(B, T, Hv))
                beta = mx.random.uniform(shape=(B, T, Hv))
                state = mx.random.normal(shape=(B, Hv, Dk, Dv))

                y_op, st_op = gated_delta_ops(q, k, v, g, beta, state)
                y_c, st_c = gated_delta_kernel(q, k, v, g, beta, state)
                self.assertTrue(mx.allclose(y_op, y_c, rtol=1e-4, atol=1e-4))
                self.assertTrue(mx.allclose(st_op, st_c, rtol=1e-4, atol=1e-4))

    def test_gated_delta_masked(self):
        B = 1
        T = 3
        Hk = 16
        Hv = 32
        Dk = 128
        Dv = 128

        mx.random.seed(0)
        q = mx.random.normal(shape=(B, T, Hk, Dk))
        k = mx.random.normal(shape=(B, T, Hk, Dk))
        v = mx.random.normal(shape=(B, T, Hv, Dv))
        g = mx.random.normal(shape=(B, T, Hv))
        mask = mx.array([[False, True, True]])
        beta = mx.random.normal(shape=(B, T, Hv))
        state = mx.random.normal(shape=(B, Hv, Dk, Dv))

        y_gt, st_gt = gated_delta_ops(
            q[:, 1:],
            k[:, 1:],
            v[:, 1:],
            g[:, 1:],
            beta[:, 1:],
            state,
        )
        for fn in [gated_delta_ops, gated_delta_kernel]:
            y, st = fn(q, k, v, g, beta, state, mask)
            y = y[:, 1:]
            self.assertTrue(mx.allclose(y, y_gt, rtol=1e-4, atol=1e-4))
            self.assertTrue(mx.allclose(st, st_gt, rtol=1e-4, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
