# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.steer import CommitSteerer, build_vector, steer_layer


class _Layer(nn.Module):
    """Identity decoder layer with a distinguishing attribute, matching the
    ``layer(x, ...) -> x`` calling convention taps rely on."""

    def __init__(self):
        super().__init__()
        self.is_linear = False

    def __call__(self, x, *args, **kwargs):
        return x


class _Inner(nn.Module):
    def __init__(self, n_layers=4, dim=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, dim)
        self.layers = [_Layer() for _ in range(n_layers)]

    def __call__(self, ids, cache=None):
        h = self.embed_tokens(ids)
        for layer in self.layers:
            h = layer(h)
        return h


class _Model(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.model = _Inner(dim=dim)
        self.args = type("A", (), {"hidden_size": dim})()

    def __call__(self, ids, cache=None):
        return self.model(ids, cache=cache)


class _Tok:
    """Minimal tokenizer: </think> and <think> are single ids 2 and 1."""

    def encode(self, text, add_special_tokens=False):
        return {"<think>": [1], "</think>": [2]}.get(text, [9, 9])


class TestBuildVector(unittest.TestCase):
    def test_scales(self):
        v = build_vector(np.array([1.0, 0.0, 0.0]), rms=10.0, alpha=0.2)
        self.assertAlmostEqual(float(mx.sum(v)), 2.0, places=2)

    def test_degenerate_is_none(self):
        self.assertIsNone(build_vector(np.zeros(3), 10.0, 0.2))          # zero norm
        self.assertIsNone(build_vector(np.array([np.nan, 0, 0]), 1.0, 0.2))  # nan
        self.assertIsNone(build_vector(np.array([1.0, 0, 0]), np.nan, 0.2))  # nan rms


class TestSteerLayer(unittest.TestCase):
    def test_injects_and_restores(self):
        model = _Model(dim=4)
        ids = mx.array([[3, 4, 5]])
        base = model(ids)
        holder = [mx.array([1.0, 1.0, 1.0, 1.0])]
        original = model.model.layers[2]
        with steer_layer(model, 2, holder):
            steered = model(ids)
        # every position shifted by exactly the vector (three identity layers
        # after the tap pass it through unchanged)
        self.assertTrue(
            mx.allclose(steered, base + holder[0], atol=1e-4).item()
        )
        # None holder is a no-op
        holder[0] = None
        with steer_layer(model, 2, holder):
            noop = model(ids)
        self.assertTrue(mx.allclose(noop, base, atol=1e-5).item())
        # layer object restored
        self.assertIs(model.model.layers[2], original)

    def test_delegates_attributes(self):
        model = _Model()
        holder = [None]
        with steer_layer(model, 1, holder):
            # the loop reads layer.is_linear on some hybrid archs; must resolve
            self.assertFalse(model.model.layers[1].is_linear)


class TestCommitSteererSchedule(unittest.TestCase):
    def _steerer(self, **kw):
        model = _Model()
        v = np.ones(8) / np.sqrt(8)
        return CommitSteerer(model, _Tok(), v, rms=10.0, layer=1, **kw)

    def test_bias_then_hammer_then_off(self):
        s = self._steerer(alpha_bias=0.2, alpha_hammer=0.8, hammer_budget=3)
        s.reset()
        # first call: prompt (pre-opened think). Holder stays off during prefill.
        s.logits_processor(mx.array([1, 5, 5]), mx.zeros((1, 32)))
        self.assertIsNone(s._holder[0])
        # decode tokens accumulate; under budget => bias
        toks = [1, 5, 5]
        for extra in (5, 5):  # 2 think tokens (count 1, 2)
            toks.append(extra)
            s.logits_processor(mx.array(toks), mx.zeros((1, 32)))
        self.assertTrue(mx.allclose(s._holder[0], s._bias).item())
        # cross the hammer budget (count reaches 3) => hammer
        toks.append(5)
        s.logits_processor(mx.array(toks), mx.zeros((1, 32)))
        self.assertTrue(mx.allclose(s._holder[0], s._hammer).item())
        # closing the think channel => steering off
        toks.append(2)  # </think>
        s.logits_processor(mx.array(toks), mx.zeros((1, 32)))
        self.assertIsNone(s._holder[0])

    def test_disabled_on_degenerate_axis(self):
        s = self._steerer()
        s._bias = None  # simulate a null direction
        s.disabled = True
        out = s.logits_processor(mx.array([1, 5]), mx.ones((1, 32)))
        self.assertTrue(mx.allclose(out, mx.ones((1, 32))).item())
        self.assertIsNone(s._holder[0])

    def test_no_close_tag_disables(self):
        model = _Model()

        class _NoThink:
            def encode(self, text, add_special_tokens=False):
                return [7, 8]  # </think> is multi-token => -1 => disabled

        s = CommitSteerer(model, _NoThink(), np.ones(8) / np.sqrt(8), 10.0, layer=1)
        self.assertTrue(s.disabled)

    def test_logits_never_modified(self):
        s = self._steerer()
        logits = mx.random.normal((1, 32))
        out = s.logits_processor(mx.array([1, 5, 5, 5]), logits)
        self.assertTrue(mx.array_equal(out, logits).item())


if __name__ == "__main__":
    unittest.main()
