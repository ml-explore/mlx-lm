# Copyright © 2025 Apple Inc.

import importlib
import itertools
import unittest
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from mlx_lm.models.cache import KVCache
from mlx_lm.models.switch_layers import SwitchGLU, SwitchMLP
from mlx_lm.tuner.trainer import iterate_batches


class MockDistributedGroup:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class TestTunerTrainer(unittest.TestCase):
    def test_iterate_batches_ddp(self):
        group = MockDistributedGroup(0, 1)

        def run(rank, size, batch):
            group._rank = rank
            group._size = size

            data = mx.arange(128).reshape(-1, 1).tolist()
            data = [(d, 0) for d in data]

            samples = set()
            for b, _ in iterate_batches(data, batch, 1, comm_group=group):
                samples.add(tuple(mx.flatten(b).tolist()))

            ref_batches = mx.arange(128).reshape(-1, batch).tolist()
            for b in ref_batches:
                self.assertTrue(tuple(b[rank::size]) in samples)

        run(0, 1, 4)
        run(0, 1, 8)
        run(0, 2, 8)
        run(1, 2, 8)
        run(0, 4, 8)
        run(1, 4, 8)
        run(2, 4, 8)
        run(3, 4, 8)

    def test_iterate_batches_seed(self):
        # One distinct token id per row so the batch order is observable.
        data = [([i + 1] * 8, 0) for i in range(64)]

        def order(seed, consume=0):
            np.random.seed(1)
            if consume:
                # Stand-in for anything else drawing from numpy in between.
                np.random.rand(consume)
            batches = iterate_batches(data, 4, 8, loop=True, seed=seed)
            return [b[0].tolist()[0] for b, _ in zip(batches, range(5))]

        # seed=0 must be honored like any other seed. It is also the default
        # in mlx_lm.lora's CONFIG_DEFAULTS, so `if seed:` silently dropped it.
        for seed in (0, 42):
            with self.subTest(seed=seed):
                self.assertEqual(order(seed), order(seed, consume=3))

        self.assertNotEqual(order(0), order(42))

    def _assert_backward(self, model):
        tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

        def loss_fn(current_model, inputs):
            return current_model(inputs).mean()

        loss, gradients = nn.value_and_grad(model, loss_fn)(model, tokens)
        mx.eval(loss, gradients)
        self.assertTrue(mx.isfinite(loss).item())

    def test_qwen3_moe_backward(self):
        from mlx_lm.models import qwen3_moe

        model = qwen3_moe.Model(
            qwen3_moe.ModelArgs(
                model_type="qwen3_moe",
                hidden_size=16,
                num_hidden_layers=1,
                intermediate_size=32,
                num_attention_heads=4,
                num_experts=4,
                num_experts_per_tok=2,
                decoder_sparse_step=1,
                mlp_only_layers=[],
                moe_intermediate_size=16,
                rms_norm_eps=1e-6,
                vocab_size=32,
                num_key_value_heads=2,
                head_dim=4,
                rope_theta=10_000.0,
                tie_word_embeddings=False,
                max_position_embeddings=128,
                norm_topk_prob=True,
            )
        )
        self._assert_backward(model)

    def test_granitemoe_backward(self):
        from mlx_lm.models import granitemoe

        model = granitemoe.Model(
            granitemoe.ModelArgs(
                model_type="granitemoe",
                hidden_size=16,
                num_hidden_layers=1,
                intermediate_size=32,
                num_attention_heads=4,
                rms_norm_eps=1e-6,
                vocab_size=32,
                logits_scaling=1.0,
                attention_multiplier=1.0,
                embedding_multiplier=1.0,
                residual_multiplier=1.0,
                max_position_embeddings=128,
                num_key_value_heads=2,
                attention_bias=False,
                rope_theta=10_000.0,
                num_local_experts=4,
                num_experts_per_tok=2,
            )
        )
        self._assert_backward(model)

    def test_granitemoehybrid_backward(self):
        from mlx_lm.models import granitemoehybrid

        model = granitemoehybrid.Model(
            granitemoehybrid.ModelArgs(
                model_type="granitemoehybrid",
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                max_position_embeddings=128,
                num_attention_heads=4,
                num_key_value_heads=2,
                attention_bias=False,
                embedding_multiplier=1.0,
                attention_multiplier=1.0,
                logits_scaling=1.0,
                residual_multiplier=1.0,
                layer_types=["attention"],
                rms_norm_eps=1e-6,
                rope_theta=10_000.0,
                num_local_experts=4,
                num_experts_per_tok=2,
                shared_intermediate_size=32,
            )
        )
        self._assert_backward(model)

    def test_lfm2_moe_backward(self):
        from mlx_lm.models import lfm2_moe

        model = lfm2_moe.Model(
            lfm2_moe.ModelArgs(
                model_type="lfm2_moe",
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                moe_intermediate_size=16,
                num_hidden_layers=2,
                num_experts=4,
                num_experts_per_tok=2,
                norm_topk_prob=True,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
                use_expert_bias=False,
                num_dense_layers=1,
                norm_eps=1e-6,
                conv_bias=False,
                conv_L_cache=3,
                layer_types=["conv", "full_attention"],
            )
        )
        self._assert_backward(model)


def _routing_loss(output):
    arrays = output if isinstance(output, tuple) else (output,)
    loss = mx.array(0.0)
    for value in arrays:
        if mx.issubdtype(value.dtype, mx.floating):
            # A plain sum gives zero gradients for normalized routing weights.
            weights = (mx.arange(value.size).reshape(value.shape) % 7 + 1) / 7
            loss = loss + (value.astype(mx.float32) * weights).mean()
    return loss


class TestRoutingGradients(unittest.TestCase):
    def setUp(self):
        mx.random.seed(73)

    def _assert_finite_nonzero(self, value):
        self.assertTrue(mx.all(mx.isfinite(value)).item())
        self.assertTrue(mx.any(value != 0).item())

    def _assert_backward(self, call, x):
        expected = _routing_loss(call(x))
        value, grad = mx.value_and_grad(lambda z: _routing_loss(call(z)))(x)
        self.assertTrue(mx.allclose(expected, value).item())
        self._assert_finite_nonzero(grad)

    def _assert_selection(self, call, logits):
        self._assert_backward(lambda x: call(x)[1], logits)
        fixed = mx.stop_gradient(call(logits)[0])
        # Returned indices must also be safe for other gather operations.
        actual = mx.grad(
            lambda x: _routing_loss(mx.take_along_axis(x, call(x)[0], axis=-1))
        )(logits)
        expected = mx.grad(
            lambda x: _routing_loss(mx.take_along_axis(x, fixed, axis=-1))
        )(logits)
        self.assertTrue(mx.array_equal(actual, expected).item())

    def test_group_routers(self):
        routers = [
            ("deepseek_v3", "group_expert_select", ()),
            ("deepseek_v32", "group_expert_select", ()),
            ("glm4_moe", "group_expert_select", ()),
            ("glm4_moe_lite", "group_expert_select", ()),
            ("exaone_moe", "group_expert_select", ()),
            ("mimo_v2_flash", "group_expert_select", ()),
            ("nemotron_h", "group_expert_select", ()),
            ("bailing_moe", "group_expert_select", ("sigmoid",)),
            ("bailing_moe", "group_expert_select", ("softmax",)),
            ("bailing_moe_linear", "group_expert_select", ("sigmoid",)),
            ("bailing_moe_linear", "group_expert_select", ("softmax",)),
            ("kimi_linear", "_group_expert_select", ("sigmoid",)),
            ("kimi_linear", "_group_expert_select", ("softmax",)),
            ("dots1", "group_expert_select", ()),
            ("kimi_k3", "_group_expert_select", ()),
        ]
        for (name, function, extra), groups in itertools.product(routers, (1, 2)):
            with self.subTest(model=name, score=extra, groups=groups):
                module = importlib.import_module(f"mlx_lm.models.{name}")
                select = getattr(module, function)
                logits = mx.random.normal((1, 6, 8))
                bias = mx.array([0.0, 0.01, -0.03, 0.02, 0.06, -0.02, 0.04, -0.01])
                self._assert_selection(
                    lambda x: select(x, bias, 2, groups, 1, 1.5, True, *extra), logits
                )

    def test_bailing_moe_v3_router(self):
        from mlx_lm.models.bailing_moe_v3 import _group_expert_select

        self._assert_selection(
            lambda x: _group_expert_select(x, mx.zeros((8,)), 2, 2, 1, 1.5),
            mx.random.normal((1, 6, 8)),
        )

    def test_step3p5_router(self):
        from mlx_lm.models.step3p5 import moe_gate_select

        self._assert_selection(
            lambda x: moe_gate_select(x, mx.zeros((8,)), 2, 1.5, True),
            mx.random.normal((1, 6, 8)),
        )

    def test_moe_blocks(self):
        blocks = [
            ("Klear", "KlearSparseMoeBlock", {}, "gate.weight"),
            ("afmoe", "AfmoeMoE", {}, "router.gate.weight"),
            ("mellum", "MellumSparseMoeBlock", {}, "gate.weight"),
            ("qwen3_next", "Qwen3NextSparseMoeBlock", {}, "gate.weight"),
            ("minimax", "MiniMaxSparseMoeBlock", {}, "gate.weight"),
            ("llama4", "MoE", {"num_experts_per_tok": 1}, "router.weight"),
            ("gemma4_text", "Router", {}, "proj.weight"),
            ("longcat_flash", "LongcatFlashTopkRouter", {}, "classifier.weight"),
            ("laguna", "MoEGate", {}, "weight"),
            ("deepseek_v2", "MoEGate", {"topk_method": "greedy"}, "weight"),
            (
                "deepseek_v2",
                "MoEGate",
                {"topk_method": "group_limited_greedy"},
                "weight",
            ),
        ]
        args = dict(
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=24,
            shared_expert_intermediate_size=24,
            num_experts=8,
            num_local_experts=8,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            num_shared_experts=1,
            norm_topk_prob=True,
            routed_scaling_factor=1.5,
            n_group=2,
            topk_group=1,
            rms_norm_eps=1e-5,
            top_k_experts=2,
            moe_topk=2,
            zero_expert_num=0,
            router_bias=False,
            route_norm=True,
            route_scale=1.5,
            score_func="sigmoid",
            moe_router_logit_softcapping=10.0,
            moe_router_score_func="sigmoid",
        )
        for (name, cls, overrides, router_key), length in itertools.product(
            blocks, (6, 40)
        ):
            with self.subTest(model=name, config=overrides, tokens=length):
                module = importlib.import_module(f"mlx_lm.models.{name}")
                block = getattr(module, cls)(SimpleNamespace(**{**args, **overrides}))
                block.train()
                if router_key == "weight":
                    block.weight = mx.random.normal(block.weight.shape) * 0.1
                x = mx.random.normal((1, length, args["hidden_size"]))
                self._assert_backward(block, x)
                _, gradients = nn.value_and_grad(block, lambda m: _routing_loss(m(x)))(
                    block
                )
                flat = dict(tree_flatten(gradients))
                for key, grad in flat.items():
                    self.assertTrue(mx.all(mx.isfinite(grad)).item(), key)
                self._assert_finite_nonzero(flat[router_key])

    def test_switch_sort_boundary(self):
        for cls, top_k, over_boundary, training, quantized in itertools.product(
            (SwitchGLU, SwitchMLP), (1, 2), (False, True), (False, True), (False, True)
        ):
            length = 64 // top_k - (not over_boundary)
            with self.subTest(
                layer=cls.__name__,
                top_k=top_k,
                tokens=length,
                training=training,
                quantized=quantized,
            ):
                block = cls(32, 32, 8)
                if quantized:
                    nn.quantize(block, group_size=32, bits=4)
                block.train(training)
                x = mx.random.normal((1, length, 32))
                router = mx.random.normal((32, 8))

                def select(z):
                    return mx.argpartition(z @ router, kth=-top_k, axis=-1)[
                        ..., -top_k:
                    ]

                fixed = mx.stop_gradient(select(x))
                self.assertTrue(
                    mx.array_equal(block(x, select(x)), block(x, fixed)).item()
                )
                actual = mx.grad(lambda z: _routing_loss(block(z, select(z))))(x)
                expected = mx.grad(lambda z: _routing_loss(block(z, fixed)))(x)
                self.assertTrue(mx.allclose(actual, expected, atol=1e-6).item())
                self._assert_finite_nonzero(actual)

    def test_sparse_attention(self):
        from mlx_lm.models import deepseek_v32

        args = deepseek_v32.ModelArgs(
            hidden_size=16,
            num_attention_heads=2,
            q_lora_rank=8,
            kv_lora_rank=8,
            qk_rope_head_dim=4,
            qk_nope_head_dim=4,
            v_head_dim=8,
            index_n_heads=2,
            index_head_dim=8,
            index_topk=2,
        )
        for length, decode, masked in (
            (2, False, False),
            (6, False, False),
            (1, True, False),
            (1, True, True),
        ):
            with self.subTest(tokens=length, decode=decode, masked=masked):
                attention = deepseek_v32.DeepseekV32Attention(args)
                attention.train()
                caches = None
                if decode:
                    caches = [KVCache(), KVCache()]
                    mx.eval(attention(mx.random.normal((1, 4, 16)), cache=caches))
                    saved = [(c.keys, c.values, c.offset) for c in caches]

                def call(x):
                    if decode:
                        for cache, state in zip(caches, saved):
                            cache.keys, cache.values, cache.offset = state
                    mask = (
                        mx.array([[[[True, False, True, True, True]]]])
                        if masked
                        else None
                    )
                    return attention(x, mask=mask, cache=caches)

                self._assert_backward(call, mx.random.normal((1, length, 16)))

    def test_qwen3_next_lora_checkpointing(self):
        from mlx_lm.models import qwen3_next
        from mlx_lm.tuner.trainer import grad_checkpoint
        from mlx_lm.tuner.utils import linear_to_lora_layers

        args = qwen3_next.ModelArgs(
            model_type="qwen3_next",
            hidden_size=16,
            num_hidden_layers=2,
            intermediate_size=32,
            num_attention_heads=2,
            linear_num_value_heads=2,
            linear_num_key_heads=1,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_conv_kernel_dim=3,
            num_experts=8,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            shared_expert_intermediate_size=24,
            mlp_only_layers=[],
            moe_intermediate_size=24,
            rms_norm_eps=1e-5,
            vocab_size=32,
            num_key_value_heads=1,
            rope_theta=10000,
            partial_rotary_factor=1.0,
            max_position_embeddings=128,
            head_dim=8,
            full_attention_interval=1,
            norm_topk_prob=True,
        )
        model = qwen3_next.Model(args)
        model.train()
        model.freeze()
        linear_to_lora_layers(model, 2, {"rank": 2, "scale": 1.0, "dropout": 0.0})
        cls = type(model.layers[0])
        self.addCleanup(setattr, cls, "__call__", cls.__call__)
        tokens = mx.array([[1, 2, 3, 4, 5, 6]])
        targets = mx.array([[2, 3, 4, 5, 6, 7]])

        def backward():
            value, gradients = nn.value_and_grad(
                model, lambda m: nn.losses.cross_entropy(m(tokens), targets).mean()
            )(model)
            flat = dict(tree_flatten(gradients))
            mx.eval(value, list(flat.values()))
            self.assertTrue(mx.isfinite(value).item())
            for key, grad in flat.items():
                self.assertTrue(mx.all(mx.isfinite(grad)).item(), key)
            self.assertTrue(any(mx.any(g != 0).item() for g in flat.values()))
            return value, flat

        expected_loss, expected_grads = backward()
        grad_checkpoint(model.layers[0])
        loss, grads = backward()
        self.assertTrue(mx.allclose(loss, expected_loss).item())
        self.assertEqual(grads.keys(), expected_grads.keys())
        for key in grads:
            self.assertTrue(
                mx.allclose(
                    grads[key], expected_grads[key], rtol=1e-5, atol=1e-6
                ).item(),
                key,
            )


if __name__ == "__main__":
    unittest.main()
