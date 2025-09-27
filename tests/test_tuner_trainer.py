# Copyright © 2025 Apple Inc.

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from mlx_lm.tuner.trainer import TrainingArgs, iterate_batches, train


class MockDistributedGroup:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class MockDistributed:
    def __init__(self):
        self.rank = 0
        self.size = 1

    def init(self):
        return MockDistributedGroup(self.rank, self.size)


def _fake_compile(fn=None, inputs=None, outputs=None):
    if fn is None:
        return lambda f: f
    return fn


def _zeros_like_safe(value):
    try:
        return mx.zeros_like(value)
    except TypeError:
        return value


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self.layers = [self.layer]

    def __call__(self, inputs):  # pragma: no cover - not used in tests
        return self.layer(inputs)


class DummyOptimizer:
    def __init__(self, learning_rate=1e-3):
        self.state = {}
        self.learning_rate = mx.array(learning_rate)
        self.update_calls = 0

    def update(self, model, grad):
        self.update_calls += 1


class TestTunerTrainer(unittest.TestCase):
    def test_iterate_batches_ddp(self):
        olddist = mx.distributed
        try:
            mx.distributed = MockDistributed()

            def run(rank, size, batch):
                mx.distributed.rank = rank
                mx.distributed.size = size

                data = mx.arange(128).reshape(-1, 1).tolist()
                data = [(d, 0) for d in data]

                samples = set()
                for i, (b, l) in enumerate(iterate_batches(data, batch, 1)):
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

        finally:
            mx.distributed = olddist


if __name__ == "__main__":
    unittest.main()


class TestEarlyStopping(unittest.TestCase):
    def _run_training(self, eval_losses, args: TrainingArgs):
        model = DummyModel()
        optimizer = DummyOptimizer()
        grad_template = tree_map(_zeros_like_safe, model.trainable_parameters())

        eval_iter = iter(eval_losses)
        eval_len = len(eval_losses)
        eval_call_count = 0

        def fake_evaluate(*_, **__):
            nonlocal eval_call_count
            eval_call_count += 1
            try:
                return next(eval_iter)
            except StopIteration as exc:  # pragma: no cover
                raise AssertionError(
                    "evaluate called more times than expected"
                ) from exc

        def fake_value_and_grad(model_arg, loss_fn):
            def inner(*inner_args, **inner_kwargs):
                return ((mx.array(1.0), mx.array(4)), grad_template)

            return inner

        def fake_iterate_batches(dataset, batch_size, max_seq_length, train=False):
            del dataset, batch_size, max_seq_length, train
            for _ in range(args.iters):
                yield (mx.array([[0]]), mx.array([[0, 0]]))

        with ExitStack() as stack:
            stack.enter_context(
                patch("mlx_lm.tuner.trainer.evaluate", new=fake_evaluate)
            )
            stack.enter_context(
                patch("mlx_lm.tuner.trainer.nn.value_and_grad", new=fake_value_and_grad)
            )
            stack.enter_context(
                patch("mlx_lm.tuner.trainer.average_gradients", new=lambda grad: grad)
            )
            stack.enter_context(
                patch("mlx_lm.tuner.trainer.mx.compile", new=_fake_compile)
            )
            stack.enter_context(
                patch("mlx_lm.tuner.trainer.mx.eval", new=lambda *a, **k: None)
            )
            stack.enter_context(
                patch(
                    "mlx_lm.tuner.trainer.mx.distributed.init",
                    new=lambda: MockDistributedGroup(0, 1),
                )
            )
            stack.enter_context(
                patch(
                    "mlx_lm.tuner.trainer.mx.distributed.all_sum",
                    new=lambda value, stream=None: value,
                )
            )
            stack.enter_context(
                patch("mlx_lm.tuner.trainer.mx.metal.is_available", new=lambda: False)
            )
            stack.enter_context(
                patch("mlx_lm.tuner.trainer.mx.get_peak_memory", new=lambda: 0)
            )
            stack.enter_context(
                patch(
                    "mlx_lm.tuner.trainer.mx.save_safetensors", new=lambda *a, **k: None
                )
            )

            train(
                model=model,
                optimizer=optimizer,
                train_dataset=[],
                val_dataset=[],
                args=args,
                iterate_batches=fake_iterate_batches,
            )

        self.assertEqual(eval_call_count, eval_len)
        return optimizer.update_calls

    def _base_args(self, **overrides):
        defaults = dict(
            batch_size=1,
            iters=3,
            val_batches=1,
            steps_per_report=1,
            steps_per_eval=1,
            steps_per_save=1,
            adapter_file="dummy.safetensors",
            max_seq_length=8,
            grad_checkpoint=False,
            early_stopping=False,
            early_stopping_patience=0,
            early_stopping_min_delta=0.0,
        )
        defaults.update(overrides)
        return TrainingArgs(**defaults)

    def test_early_stopping_disabled_runs_all_iters(self):
        args = self._base_args(iters=3, early_stopping=False)
        eval_losses = [1.0, 1.1, 1.2]
        updates = self._run_training(eval_losses, args)
        self.assertEqual(updates, args.iters)

    def test_early_stopping_stops_when_patience_zero(self):
        args = self._base_args(
            iters=3,
            early_stopping=True,
            early_stopping_patience=0,
            early_stopping_min_delta=0.0,
        )
        eval_losses = [1.0, 1.05]
        updates = self._run_training(eval_losses, args)
        self.assertEqual(updates, 1)

    def test_early_stopping_respects_min_delta(self):
        args = self._base_args(
            iters=4,
            early_stopping=True,
            early_stopping_patience=1,
            early_stopping_min_delta=0.05,
        )
        eval_losses = [1.0, 0.97, 0.96]
        updates = self._run_training(eval_losses, args)
        self.assertEqual(updates, 2)

    def test_early_stopping_resets_patience_on_sufficient_improvement(self):
        args = self._base_args(
            iters=4,
            early_stopping=True,
            early_stopping_patience=1,
            early_stopping_min_delta=0.05,
        )
        eval_losses = [1.0, 0.97, 0.9, 0.84]
        updates = self._run_training(eval_losses, args)
        self.assertEqual(updates, args.iters)
