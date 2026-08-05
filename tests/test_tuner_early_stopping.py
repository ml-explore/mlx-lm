# Copyright © 2025 Apple Inc.

import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import numpy as np

from mlx_lm.tuner.trainer import TrainingArgs, TrainingCallback, train


class Tiny(nn.Module):
    def __init__(self, vocab_size=32, dims=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dims)
        self.out = nn.Linear(dims, vocab_size, bias=False)

    def __call__(self, x):
        return self.out(self.embed(x))


class Recorder(TrainingCallback):
    def __init__(self):
        self.val = []

    def on_train_loss_report(self, info):
        pass

    def on_val_loss_report(self, info):
        self.val.append(info)


def batches(
    dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None
):
    while True:
        tokens = mx.full((batch_size, 8), 1, dtype=mx.int32)
        yield tokens, mx.array([[0, 8]] * batch_size)
        if not loop:
            break


def scripted_loss(val_losses):
    """Drives validation loss through a fixed sequence so the improvement logic
    can be tested without depending on real optimization dynamics.

    train() calls the loss for training steps and for validation; only the
    validation calls happen while model.training is False.
    """
    seq = list(val_losses)
    state = {"i": 0}

    def loss(model, batch, lengths):
        ntoks = mx.array(batch.size)
        if not model.training:
            i = min(state["i"], len(seq) - 1)
            state["i"] += 1
            return mx.array(seq[i]), ntoks
        logits = model(batch[:, :-1])
        ce = nn.losses.cross_entropy(logits, batch[:, 1:]).astype(mx.float32).mean()
        return ce, ntoks

    return loss


class TestEarlyStopping(unittest.TestCase):
    def setUp(self):
        mx.random.seed(0)
        self.data = [([1] * 8, 0)] * 8
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.adapter = Path(self.tmp.name) / "adapters.safetensors"
        self.best = Path(self.tmp.name) / "best_adapters.safetensors"

    def _train(self, iters=6, steps_per_eval=1, loss=None, **kwargs):
        cb = Recorder()
        args = TrainingArgs(
            batch_size=2,
            iters=iters,
            val_batches=1,
            steps_per_report=10**9,
            steps_per_eval=steps_per_eval,
            steps_per_save=10**9,
            max_seq_length=8,
            adapter_file=self.adapter,
            **kwargs,
        )
        train(
            model=Tiny(),
            optimizer=opt.Adam(learning_rate=1e-3),
            train_dataset=self.data,
            val_dataset=self.data,
            args=args,
            iterate_batches=batches,
            training_callback=cb,
            **({"loss": loss} if loss else {}),
        )
        return cb

    def test_best_adapter_saved_on_improvement(self):
        self._train(loss=scripted_loss([1.0, 0.5, 0.4]))
        self.assertTrue(self.best.exists())
        self.assertTrue(self.adapter.exists())

    def test_best_adapter_not_written_when_disabled(self):
        self._train(loss=scripted_loss([1.0, 0.5]), save_best=False)
        self.assertFalse(self.best.exists())
        self.assertTrue(self.adapter.exists())

    def test_best_differs_from_final_when_val_loss_rises(self):
        """The motivating case: validation bottoms out and then degrades, so the
        final adapter is not the best one."""
        self._train(iters=6, loss=scripted_loss([1.0, 0.3, 2.0, 3.0, 4.0, 5.0]))
        best = mx.load(str(self.best))
        final = mx.load(str(self.adapter))
        differs = any(
            not np.array_equal(np.asarray(best[k]), np.asarray(final[k])) for k in best
        )
        self.assertTrue(differs, "best checkpoint should differ from the final one")

    def test_patience_stops_early(self):
        cb = self._train(iters=20, loss=scripted_loss([1.0, 2.0, 3.0, 4.0]), patience=2)
        # Validations at iter 1 (1.0), 2 (2.0), 3 (3.0): two non-improving in a
        # row trips patience, so training ends well before iters=20.
        self.assertLess(len(cb.val), 20)
        self.assertTrue(self.adapter.exists())

    def test_patience_not_tripped_while_improving(self):
        cb = self._train(
            iters=5, loss=scripted_loss([1.0, 0.9, 0.8, 0.7, 0.6]), patience=2
        )
        self.assertEqual(len(cb.val), 5)

    def test_min_delta_requires_meaningful_improvement(self):
        """Tiny improvements should not reset the patience counter."""
        cb = self._train(
            iters=20,
            loss=scripted_loss([1.0, 0.9999, 0.9998, 0.9997]),
            patience=2,
            min_delta=0.01,
        )
        self.assertLess(len(cb.val), 20)

    def test_short_run_without_periodic_save(self):
        """Regression: iters < steps_per_save must still save, and must not
        raise for an unbound adapter_weights."""
        self._train(iters=2, steps_per_eval=1, loss=scripted_loss([1.0, 0.5]))
        self.assertTrue(self.adapter.exists())
        self.assertTrue(self.best.exists())

    def test_best_tracked_when_eval_and_save_cadences_differ(self):
        """Regression: the best checkpoint must come from a validated iteration,
        not from whenever the last periodic save happened."""
        cb = Recorder()
        args = TrainingArgs(
            batch_size=2,
            iters=6,
            val_batches=1,
            steps_per_report=10**9,
            steps_per_eval=2,
            steps_per_save=3,  # deliberately not a multiple of steps_per_eval
            max_seq_length=8,
            adapter_file=self.adapter,
        )
        train(
            model=Tiny(),
            optimizer=opt.Adam(learning_rate=1e-3),
            train_dataset=self.data,
            val_dataset=self.data,
            args=args,
            iterate_batches=batches,
            training_callback=cb,
            loss=scripted_loss([1.0, 0.2, 5.0, 6.0]),
        )
        self.assertTrue(self.best.exists())

    def test_rejects_bad_patience(self):
        for bad in (0, -1):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    self._train(patience=bad)

    def test_patience_requires_validation_set(self):
        args = TrainingArgs(
            batch_size=2,
            iters=2,
            adapter_file=self.adapter,
            max_seq_length=8,
            patience=2,
        )
        with self.assertRaises(ValueError):
            train(
                model=Tiny(),
                optimizer=opt.Adam(learning_rate=1e-3),
                train_dataset=self.data,
                val_dataset=None,
                args=args,
                iterate_batches=batches,
            )

    def test_save_best_without_validation_set_is_a_noop(self):
        args = TrainingArgs(
            batch_size=2,
            iters=2,
            adapter_file=self.adapter,
            max_seq_length=8,
            steps_per_report=10**9,
            steps_per_save=10**9,
        )
        train(
            model=Tiny(),
            optimizer=opt.Adam(learning_rate=1e-3),
            train_dataset=self.data,
            val_dataset=None,
            args=args,
            iterate_batches=batches,
        )
        self.assertTrue(self.adapter.exists())
        self.assertFalse(self.best.exists())


if __name__ == "__main__":
    unittest.main()
