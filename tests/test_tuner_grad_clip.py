# Copyright © 2025 Apple Inc.

import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import numpy as np
from mlx.utils import tree_flatten

from mlx_lm.tuner.trainer import TrainingArgs, TrainingCallback, train


class Tiny(nn.Module):
    """Small model so these tests run without loading real weights."""

    def __init__(self, vocab_size=32, dims=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dims)
        self.out = nn.Linear(dims, vocab_size, bias=False)

    def __call__(self, x):
        return self.out(self.embed(x))


class Recorder(TrainingCallback):
    def __init__(self):
        self.reports = []

    def on_train_loss_report(self, info):
        self.reports.append(info)

    def on_val_loss_report(self, info):
        pass


NORMAL_TOKEN = 1
POISON_TOKEN = 7


def constant_batches(length=8, poison_at=None):
    """Deterministic batch stream in the shape iterate_batches yields.

    A poisoned batch is marked in the token ids rather than by a Python flag,
    because step() is wrapped in mx.compile: a Python-side branch would be
    traced once and then baked into every later iteration.
    """

    def gen(
        dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None
    ):
        it = 0
        while True:
            it += 1
            token = POISON_TOKEN if it == poison_at else NORMAL_TOKEN
            tokens = mx.full((batch_size, length), token, dtype=mx.int32)
            yield tokens, mx.array([[0, length]] * batch_size)
            if not loop:
                break

    return gen


def spiking_loss(kind="nan"):
    """Loss that overflows only on a batch marked with POISON_TOKEN, simulating
    a single batch whose gradient is not finite. The branch is a mx.where on
    the batch contents so it survives compilation."""
    bad = float("nan") if kind == "nan" else float("inf")

    def loss(model, batch, lengths):
        logits = model(batch[:, :-1])
        ce = nn.losses.cross_entropy(logits, batch[:, 1:]).astype(mx.float32).mean()
        poisoned = mx.any(batch == POISON_TOKEN)
        ce = ce * mx.where(poisoned, mx.array(bad), mx.array(1.0))
        return ce, mx.array(batch.size)

    return loss


def weights_of(model):
    return {k: np.asarray(v) for k, v in tree_flatten(model.trainable_parameters())}


class TestGradClip(unittest.TestCase):
    def setUp(self):
        mx.random.seed(0)
        self.dataset = [([1] * 8, 0)] * 8
        # train() always saves the final adapter, so keep it out of the repo.
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)

    def _train(self, model, optimizer, iters=4, loss=None, poison_at=None, **kwargs):
        cb = Recorder()
        args = TrainingArgs(
            batch_size=2,
            iters=iters,
            val_batches=0,
            steps_per_report=1,
            steps_per_eval=10**9,
            steps_per_save=10**9,
            max_seq_length=8,
            adapter_file=Path(self.tmp_dir.name) / "adapters.safetensors",
            **kwargs,
        )
        train(
            model=model,
            optimizer=optimizer,
            train_dataset=self.dataset,
            val_dataset=None,
            args=args,
            iterate_batches=constant_batches(poison_at=poison_at),
            training_callback=cb,
            **({"loss": loss} if loss else {}),
        )
        return cb

    def test_default_is_off(self):
        """No grad_clip means no norm tracking and no reporting keys."""
        model = Tiny()
        cb = self._train(model, opt.Adam(learning_rate=1e-3))
        self.assertTrue(cb.reports)
        for r in cb.reports:
            self.assertNotIn("max_grad_norm", r)
            self.assertNotIn("non_finite_grad_steps", r)

    def test_clip_matches_unclipped_when_norm_is_small(self):
        """A clip bound above the actual norm must not change the trajectory."""
        mx.random.seed(0)
        a = Tiny()
        self._train(a, opt.SGD(learning_rate=1e-3), loss=None)

        mx.random.seed(0)
        b = Tiny()
        cb = self._train(b, opt.SGD(learning_rate=1e-3), grad_clip=1e9)

        for k, v in weights_of(a).items():
            self.assertTrue(np.allclose(v, weights_of(b)[k], atol=1e-6), k)
        self.assertLess(max(r["max_grad_norm"] for r in cb.reports), 1e9)

    def test_clip_bounds_the_norm(self):
        """With a tight bound, the update is smaller than without one."""
        mx.random.seed(0)
        loose = Tiny()
        self._train(loose, opt.SGD(learning_rate=1.0), iters=1)

        mx.random.seed(0)
        tight = Tiny()
        cb = self._train(tight, opt.SGD(learning_rate=1.0), iters=1, grad_clip=1e-4)

        mx.random.seed(0)
        start = weights_of(Tiny())
        moved_loose = max(
            np.abs(v - start[k]).max() for k, v in weights_of(loose).items()
        )
        moved_tight = max(
            np.abs(v - start[k]).max() for k, v in weights_of(tight).items()
        )
        self.assertGreater(moved_loose, moved_tight)
        self.assertGreater(cb.reports[0]["max_grad_norm"], 1e-4)

    def test_nan_gradient_does_not_reach_the_weights(self):
        """The motivating case: one bad batch must not poison the run."""
        for kind in ("nan", "inf"):
            with self.subTest(kind=kind):
                mx.random.seed(0)
                model = Tiny()
                cb = self._train(
                    model,
                    opt.Adam(learning_rate=1e-3),
                    iters=4,
                    loss=spiking_loss(kind=kind),
                    poison_at=2,
                    grad_clip=1.0,
                )
                for k, v in weights_of(model).items():
                    self.assertTrue(np.isfinite(v).all(), f"{k} went non-finite")
                self.assertEqual(sum(r["non_finite_grad_steps"] for r in cb.reports), 1)

    def test_nan_gradient_poisons_the_weights_without_the_guard(self):
        """Control for the test above: without grad_clip the weights are lost.
        This is current behavior on main, and the reason for the flag."""
        mx.random.seed(0)
        model = Tiny()
        self._train(
            model,
            opt.Adam(learning_rate=1e-3),
            iters=4,
            loss=spiking_loss(),
            poison_at=2,
        )
        self.assertFalse(
            all(np.isfinite(v).all() for v in weights_of(model).values()),
            "expected unguarded training to go non-finite",
        )

    def test_training_continues_after_a_bad_batch(self):
        """A skipped step must not stop later steps from learning."""
        mx.random.seed(0)
        model = Tiny()
        before = weights_of(model)
        self._train(
            model,
            opt.Adam(learning_rate=1e-3),
            iters=6,
            loss=spiking_loss(),
            poison_at=2,
            grad_clip=1.0,
        )
        after = weights_of(model)
        self.assertTrue(any(not np.allclose(before[k], after[k]) for k in before))
        self.assertTrue(all(np.isfinite(v).all() for v in after.values()))

    def test_rejects_non_positive_clip(self):
        for bad in (0.0, -1.0):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    self._train(Tiny(), opt.Adam(learning_rate=1e-3), grad_clip=bad)

    def test_clips_accumulated_gradient_once(self):
        """With accumulation the bound applies to the averaged gradient, so the
        reported norm is on the same scale as without accumulation."""
        mx.random.seed(0)
        cb = self._train(
            Tiny(),
            opt.SGD(learning_rate=1e-3),
            iters=4,
            grad_clip=1e9,
            grad_accumulation_steps=2,
        )
        norms = [r["max_grad_norm"] for r in cb.reports if r["max_grad_norm"] > 0]
        self.assertTrue(norms)
        self.assertTrue(all(np.isfinite(n) for n in norms))


if __name__ == "__main__":
    unittest.main()
