# Copyright © 2024 Apple Inc.

import math
import sys
import unittest
from contextlib import contextmanager
from io import StringIO
from unittest.mock import ANY, MagicMock

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten

from mlx_lm import lora, tuner
from mlx_lm.tuner.dora import DoRAEmbedding, DoRALinear
from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear
from mlx_lm.tuner.trainer import evaluate
from mlx_lm.tuner.utils import build_schedule


@contextmanager
def swapped_with_identity(obj, func):
    old_func = getattr(obj, func)
    setattr(obj, func, lambda x, **kwargs: x)
    yield
    setattr(obj, func, old_func)


class TestLora(unittest.TestCase):
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
            tie_word_embeddings=False,
        )
        lora_layers = 4

        def check_config(params, expected_trainable_parameters=None):
            n_keys = 2
            if "keys" in params:
                n_keys = len(params["keys"])
            model = llama.Model(args)
            model.freeze()
            tuner.utils.linear_to_lora_layers(model, lora_layers, params)
            trainable_params = sum(
                v.size for _, v in tree_flatten(model.trainable_parameters())
            )

            expected_trainable_parameters = expected_trainable_parameters or (
                lora_layers * params["rank"] * args.hidden_size * 2 * n_keys
            )
            self.assertEqual(trainable_params, expected_trainable_parameters)

        params = {"rank": 8, "dropout": 0.0, "scale": 10.0}
        nparams = (
            args.hidden_size * 2 * 4 + (args.intermediate_size + args.hidden_size) * 3
        ) * lora_layers
        check_config(params, expected_trainable_parameters=nparams * params["rank"])

        params["rank"] = 1
        check_config(params, expected_trainable_parameters=nparams * params["rank"])

        params["keys"] = ["self_attn.k_proj"]
        check_config(params)

        params["keys"] = ["lm_head"]
        check_config(
            params,
            expected_trainable_parameters=(
                params["rank"] * (args.hidden_size + args.vocab_size)
            ),
        )

        params["keys"] = ["model.embed_tokens"]
        check_config(
            params,
            expected_trainable_parameters=(
                params["rank"] * (args.hidden_size + args.vocab_size)
            ),
        )

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

        num_lora_layers = 4
        params = {"rank": 8, "dropout": 0.0, "scale": 10.0}

        model = gpt_neox.Model(args)
        model.freeze()
        tuner.utils.linear_to_lora_layers(model, num_lora_layers, params)

    def test_lora_embedding(self):
        num_embeddings = 256
        dims = 512
        tokens = mx.array([1, 2, 3])

        embedding = nn.QuantizedEmbedding(num_embeddings, dims)
        dequantized_weight = mx.dequantize(
            embedding.weight,
            embedding.scales,
            embedding.biases,
            embedding.group_size,
            embedding.bits,
        )
        lora_emb = LoRAEmbedding.from_base(embedding, r=8, dropout=0, scale=10)
        new_embedding = lora_emb.fuse(dequantize=True)
        self.assertTrue(mx.array_equal(dequantized_weight, new_embedding.weight))
        self.assertTrue(mx.array_equal(embedding(tokens), lora_emb(tokens)))

        # as_linear
        attn_output = mx.random.uniform(shape=(dims,))
        embedding_lin_out = lora_emb.as_linear(attn_output)
        self.assertEqual(embedding_lin_out.shape, (num_embeddings,))
        self.assertTrue(
            mx.array_equal(embedding_lin_out, embedding.as_linear(attn_output))
        )

        # change the value of lora_b and the embeddings will no longer be equal
        lora_emb.lora_b = mx.random.uniform(shape=lora_emb.lora_b.shape)
        new_embedding = lora_emb.fuse(dequantize=True)
        self.assertFalse(mx.array_equal(dequantized_weight, new_embedding.weight))
        self.assertFalse(mx.array_equal(embedding(tokens), lora_emb(tokens)))


class TestDora(unittest.TestCase):
    def test_dora_embedding(self):
        num_embeddings = 256
        dims = 512
        tokens = mx.array([1, 2, 3])

        embedding = nn.Embedding(num_embeddings, dims)

        dora_emb = DoRAEmbedding.from_base(embedding, r=8, dropout=0, scale=10)
        new_embedding = dora_emb.fuse()
        self.assertTrue(mx.array_equal(embedding.weight, new_embedding.weight))
        self.assertTrue(mx.array_equal(embedding(tokens), dora_emb(tokens)))

        # as_linear
        attn_output = mx.random.uniform(shape=(dims,))
        embedding_lin_out = dora_emb.as_linear(attn_output)
        self.assertEqual(embedding_lin_out.shape, (num_embeddings,))
        self.assertTrue(
            mx.array_equal(embedding_lin_out, embedding.as_linear(attn_output))
        )

        # change the value of lora_b and the embeddings will no longer be equal
        dora_emb.lora_b = mx.random.uniform(shape=dora_emb.lora_b.shape)
        new_embedding = dora_emb.fuse()
        self.assertFalse(mx.array_equal(embedding.weight, new_embedding.weight))
        self.assertFalse(mx.array_equal(embedding(tokens), dora_emb(tokens)))

    def test_llama(self):
        from mlx_lm.models import llama

        hidden_size = 1024
        intermediate_size = 2048
        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=hidden_size,
            num_hidden_layers=4,
            intermediate_size=intermediate_size,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
        )

        dora_layers = 4

        def check_config(params, expected_params=None):
            n_keys = 2
            if "keys" in params:
                n_keys = len(params["keys"])
            model = llama.Model(args)
            model.freeze()
            tuner.utils.linear_to_lora_layers(model, dora_layers, params, use_dora=True)
            trainable_params = sum(
                v.size for _, v in tree_flatten(model.trainable_parameters())
            )
            expected_params = expected_params or (
                dora_layers
                * (params["rank"] * hidden_size * 2 * n_keys + n_keys * hidden_size)
            )
            self.assertEqual(trainable_params, expected_params)

        params = {"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 10.0}
        nparams = (
            args.hidden_size * 2 * 4 + (args.intermediate_size + args.hidden_size) * 3
        )
        nparams = (
            nparams * params["rank"] + 5 * args.hidden_size + 2 * args.intermediate_size
        ) * dora_layers
        check_config(params, expected_params=nparams)

        params["rank"] = 1
        nparams = (
            args.hidden_size * 2 * 4 + (args.intermediate_size + args.hidden_size) * 3
        )
        nparams = (
            nparams * params["rank"] + 5 * args.hidden_size + 2 * args.intermediate_size
        ) * dora_layers
        check_config(params, expected_params=nparams * params["rank"])

        params["keys"] = ["self_attn.k_proj"]
        check_config(params)

    def test_dora_m_parameter(self):
        dora_lin = DoRALinear(input_dims=100, output_dims=100)
        self.assertTrue(
            mx.allclose(dora_lin.m, mx.linalg.norm(dora_lin.linear.weight, axis=1))
        )

        # Recomputes m when changing Linear
        inital_m = dora_lin.m
        lin = nn.Linear(10, 10)
        dora_lin.set_linear(lin)
        self.assertTrue(mx.allclose(dora_lin.m, mx.linalg.norm(lin.weight, axis=1)))

        # Works with quantized weights
        quantized_linear = nn.QuantizedLinear(512, 512)
        dora_lin.set_linear(quantized_linear)
        dequantized_weight = mx.dequantize(
            quantized_linear.weight,
            quantized_linear.scales,
            quantized_linear.biases,
            quantized_linear.group_size,
            quantized_linear.bits,
        )
        self.assertTrue(
            mx.allclose(dora_lin.m, mx.linalg.norm(dequantized_weight, axis=1))
        )

    def test_dora_from_linear(self):
        in_dims = 256
        out_dims = 256
        r = 4

        linear = nn.Linear(in_dims, out_dims)
        dora_lin = DoRALinear.from_base(linear, r)
        self.assertTrue(mx.allclose(dora_lin.m, mx.linalg.norm(linear.weight, axis=1)))
        self.assertEqual(dora_lin.lora_a.shape, (in_dims, r))
        self.assertEqual(dora_lin.lora_b.shape, (r, out_dims))
        self.assertEqual(dora_lin.m.shape, (out_dims,))

        quantized_linear = nn.QuantizedLinear(in_dims, out_dims)
        dequantized_weight = mx.dequantize(
            quantized_linear.weight,
            quantized_linear.scales,
            quantized_linear.biases,
            quantized_linear.group_size,
            quantized_linear.bits,
        )
        dora_quant_lin = DoRALinear.from_base(quantized_linear, r)
        self.assertTrue(
            mx.allclose(dora_quant_lin.m, mx.linalg.norm(dequantized_weight, axis=1))
        )
        self.assertEqual(dora_quant_lin.lora_a.shape, (in_dims, r))
        self.assertEqual(dora_quant_lin.lora_b.shape, (r, out_dims))
        self.assertEqual(dora_quant_lin.m.shape, (out_dims,))

    def test_dora_to_linear(self):
        in_dims = 256
        out_dims = 256
        r = 4

        linear = nn.Linear(in_dims, out_dims, bias=True)
        dora_lin = DoRALinear.from_base(linear, r)
        to_linear = dora_lin.fuse()
        self.assertTrue(mx.allclose(linear.weight, to_linear.weight))
        self.assertTrue(mx.allclose(linear.bias, to_linear.bias))

        def dequantize_weight(quantized_linear):
            return mx.dequantize(
                quantized_linear.weight,
                quantized_linear.scales,
                quantized_linear.biases,
                quantized_linear.group_size,
                quantized_linear.bits,
            )

        quantized_linear = nn.QuantizedLinear(in_dims, out_dims, bias=True)
        dora_quantized_linear = DoRALinear.from_base(quantized_linear, r)
        # Dequantize
        to_linear_from_quantized = dora_quantized_linear.fuse(dequantize=True)
        self.assertTrue(
            mx.allclose(quantized_linear.bias, to_linear_from_quantized.bias)
        )
        self.assertTrue(
            mx.allclose(
                dequantize_weight(quantized_linear), to_linear_from_quantized.weight
            )
        )

    def test_dora_dtype(self):
        in_dims = 256
        out_dims = 256
        r = 4

        linear = nn.Linear(in_dims, out_dims, bias=True)
        linear.set_dtype(mx.float16)
        dora_lin = DoRALinear.from_base(linear, r)

        x = mx.random.uniform(shape=(2, 256)).astype(mx.float16)
        self.assertEqual(dora_lin(x).dtype, mx.float16)


class TestScheduleConfig(unittest.TestCase):
    def test_join(self):
        config = {"name": "cosine_decay", "warmup": 100, "arguments": [1e-5, 100]}
        cos_with_warmup = build_schedule(config)
        self.assertIsNotNone(cos_with_warmup)

        self.assertEqual(cos_with_warmup(0), 0.0)
        self.assertAlmostEqual(cos_with_warmup(101), 1e-5, delta=1e-1)
        optimizer = opt.Adam(learning_rate=cos_with_warmup)
        for _ in range(100):
            optimizer.update({}, {})
        self.assertAlmostEqual(optimizer.learning_rate.item(), 1e-5, delta=1e-1)
        for _ in range(100):
            optimizer.update({}, {})
        expected_lr = 1e-5 * 0.5 * (1.0 + math.cos(math.pi * 200 / 10))
        self.assertAlmostEqual(optimizer.learning_rate.item(), expected_lr, delta=1e-1)

    def test_single_schedule(self):
        config = {
            "name": "cosine_decay",
            "arguments": [0.1, 10],
        }
        lr_schedule = build_schedule(config)
        lr = lr_schedule(4)
        expected_lr = 0.1 * 0.5 * (1.0 + math.cos(math.pi * 4 / 10))
        self.assertAlmostEqual(lr, expected_lr, delta=1e-7)

    def test_non_zero_warmup(self):
        config = {
            "name": "cosine_decay",
            "warmup": 10,
            "warmup_init": 1e-6,
            "arguments": [1e-5, 20],
        }
        lr_schedule = build_schedule(config)
        lr = lr_schedule(0)
        self.assertAlmostEqual(lr, 1e-6, delta=1e-7)

    def test_malformed_config(self):
        config = {"warmup": 100}
        self.assertRaises(KeyError, build_schedule, config)

        config = {"cosine_decay": None}
        self.assertRaises(KeyError, build_schedule, config)

    def test_evaluate_calls(self):
        mock_model = MagicMock()
        mock_dataset = MagicMock()
        mock_default_loss = MagicMock()
        mock_iterate_batches = MagicMock()

        mock_iterate_batches.return_value = [
            (MagicMock(), MagicMock()),
            (MagicMock(), MagicMock()),
            (MagicMock(), MagicMock()),
            (MagicMock(), MagicMock()),
            (MagicMock(), MagicMock()),
        ]

        mock_default_loss.side_effect = [
            (MagicMock(return_value=0.5), MagicMock(return_value=100)),
            (MagicMock(return_value=0.3), MagicMock(return_value=200)),
            (MagicMock(return_value=0.2), MagicMock(return_value=150)),
            (MagicMock(return_value=0.4), MagicMock(return_value=180)),
            (MagicMock(return_value=0.6), MagicMock(return_value=120)),
        ]
        with swapped_with_identity(mx.distributed, "all_sum"):
            evaluate(
                model=mock_model,
                dataset=mock_dataset,
                batch_size=2,
                num_batches=2,
                max_seq_length=2048,
                loss=mock_default_loss,
                iterate_batches=mock_iterate_batches,
            )

        mock_iterate_batches.assert_called_once_with(
            dataset=mock_dataset,
            batch_size=2,
            max_seq_length=2048,
            comm_group=ANY,
        )
        self.assertEqual(mock_default_loss.call_count, 2)

    def test_evaluate_infinite_batches(self):
        mock_model = MagicMock()
        mock_dataset = MagicMock()
        mock_default_loss = MagicMock()
        mock_iterate_batches = MagicMock()

        mock_iterate_batches.return_value = [
            (MagicMock(), MagicMock()),
            (MagicMock(), MagicMock()),
            (MagicMock(), MagicMock()),
        ]

        mock_default_loss.side_effect = [
            (MagicMock(return_value=0.5), MagicMock(return_value=100)),
            (MagicMock(return_value=0.3), MagicMock(return_value=200)),
            (MagicMock(return_value=0.2), MagicMock(return_value=150)),
        ]

        with swapped_with_identity(mx.distributed, "all_sum"):
            evaluate(
                model=mock_model,
                dataset=mock_dataset,
                batch_size=2,
                num_batches=-1,
                max_seq_length=2048,
                loss=mock_default_loss,
                iterate_batches=mock_iterate_batches,
            )

        mock_iterate_batches.assert_called_once_with(
            dataset=mock_dataset,
            batch_size=2,
            max_seq_length=2048,
            comm_group=ANY,
        )
        self.assertEqual(mock_default_loss.call_count, 3)


class TestGradientAccumulation(unittest.TestCase):
    def _build(self):
        import numpy as np

        vocab, dim = 64, 16

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab, dim)
                self.lm_head = nn.Linear(dim, vocab, bias=False)

            def __call__(self, x):
                return self.lm_head(self.embed(x))

        def fresh_model():
            mx.random.seed(0)
            model = TinyLM()
            mx.eval(model.parameters())
            return model

        def make_mb(ntok, seq_len=33):
            np.random.seed(ntok)
            arr = np.zeros((1, seq_len), np.int32)
            arr[0, :ntok] = np.random.randint(1, vocab, ntok)
            return mx.array(arr), ntok

        def lengths(ls):
            return mx.array([[0, n] for n in ls])

        return fresh_model, make_mb, lengths

    def _run(self, model, batches, iters, grad_accum, loss=None):
        import contextlib
        import os
        import tempfile

        from mlx_lm.tuner.trainer import TrainingArgs, default_loss, train

        def batch_iterator(bs):
            def iterate(
                dataset, batch_size, max_seq_length, loop=False, comm_group=None
            ):
                while True:
                    yield from bs
                    if not loop:
                        break

            return iterate

        with (
            tempfile.TemporaryDirectory() as td,
            contextlib.redirect_stdout(StringIO()),
        ):
            train(
                model,
                opt.SGD(learning_rate=0.1),
                train_dataset=list(range(len(batches))),
                args=TrainingArgs(
                    iters=iters,
                    batch_size=int(batches[0][0].shape[0]),
                    grad_accumulation_steps=grad_accum,
                    max_seq_length=64,
                    steps_per_report=1000,
                    steps_per_eval=1000,
                    steps_per_save=1000,
                    adapter_file=os.path.join(td, "adapters.safetensors"),
                ),
                loss=loss or default_loss,
                iterate_batches=batch_iterator(batches),
            )
        return dict(tree_flatten(model.parameters()))

    def _rel_diff(self, pa, pb):
        diff = sum(mx.abs(pa[k] - pb[k]).sum().item() for k in pa)
        norm = sum(mx.abs(pb[k]).sum().item() for k in pb)
        return diff / norm

    def test_unequal_token_microbatches_match_combined_batch(self):
        # Unequal-token micro-batches must give the same update as one combined batch.
        fresh_model, make_mb, lengths = self._build()
        a1, n1 = make_mb(12)
        a2, n2 = make_mb(4)
        both = mx.concatenate([a1, a2], axis=0)
        micro = [(a1, lengths([n1])), (a2, lengths([n2]))]
        combined = [(both, lengths([n1, n2]))]
        pa = self._run(fresh_model(), micro, iters=2, grad_accum=2)
        pb = self._run(fresh_model(), combined, iters=1, grad_accum=1)
        self.assertLess(self._rel_diff(pa, pb), 1e-4)

    def test_final_partial_window_is_applied(self):
        # A trailing partial window (iters not a multiple of grad_accum) must
        # still be applied: 3 micro-batches at grad_accum=2 should match the two
        # windows run explicitly.
        fresh_model, make_mb, lengths = self._build()
        a1, n1 = make_mb(12)
        a2, n2 = make_mb(4)
        a3, n3 = make_mb(7)
        w1 = mx.concatenate([a1, a2], axis=0)
        micro = [(a1, lengths([n1])), (a2, lengths([n2])), (a3, lengths([n3]))]
        windows = [(w1, lengths([n1, n2])), (a3, lengths([n3]))]
        pa = self._run(fresh_model(), micro, iters=3, grad_accum=2)
        pb = self._run(fresh_model(), windows, iters=2, grad_accum=1)
        self.assertLess(self._rel_diff(pa, pb), 1e-4)

    def test_zero_token_window_keeps_params_finite(self):
        # An all-zero-token window must not add a step-side divide-by-zero (the
        # zero-token loss NaN itself lives in default_loss). Use a guarded loss.
        fresh_model, make_mb, lengths = self._build()

        def guarded_loss(model, batch, length):
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(inputs)
            steps = mx.arange(1, targets.shape[1] + 1)
            mask = mx.logical_and(steps >= length[:, 0:1], steps <= length[:, 1:])
            ce = nn.losses.cross_entropy(logits, targets) * mask
            ntoks = mask.sum()
            ce = ce.astype(mx.float32).sum() / mx.maximum(ntoks, 1)
            return ce, ntoks

        a0, n0 = make_mb(0)
        zero_window = [(a0, lengths([n0])), (a0, lengths([n0]))]
        params = self._run(
            fresh_model(), zero_window, iters=2, grad_accum=2, loss=guarded_loss
        )
        for name, p in params.items():
            self.assertFalse(bool(mx.isnan(p.astype(mx.float32)).any().item()), name)

    def test_mixed_zero_and_nonzero_window_ignores_zero_token(self):
        # A zero-token micro-batch contributes nothing, so mixing it with a normal
        # one (either order) must match the normal micro-batch alone.
        fresh_model, make_mb, lengths = self._build()

        def guarded_loss(model, batch, length):
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(inputs)
            steps = mx.arange(1, targets.shape[1] + 1)
            mask = mx.logical_and(steps >= length[:, 0:1], steps <= length[:, 1:])
            ce = nn.losses.cross_entropy(logits, targets) * mask
            ntoks = mask.sum()
            ce = ce.astype(mx.float32).sum() / mx.maximum(ntoks, 1)
            return ce, ntoks

        a0, n0 = make_mb(0)
        a1, n1 = make_mb(9)
        zero_first = [(a0, lengths([n0])), (a1, lengths([n1]))]
        nonzero_first = [(a1, lengths([n1])), (a0, lengths([n0]))]
        only = [(a1, lengths([n1]))]
        p_zero_first = self._run(
            fresh_model(), zero_first, iters=2, grad_accum=2, loss=guarded_loss
        )
        p_nonzero_first = self._run(
            fresh_model(), nonzero_first, iters=2, grad_accum=2, loss=guarded_loss
        )
        p_only = self._run(
            fresh_model(), only, iters=1, grad_accum=1, loss=guarded_loss
        )
        self.assertLess(self._rel_diff(p_zero_first, p_only), 1e-4)
        self.assertLess(self._rel_diff(p_nonzero_first, p_only), 1e-4)


if __name__ == "__main__":
    unittest.main()
