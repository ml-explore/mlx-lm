# Copyright © 2026 Apple Inc.
#
# DiffusionGemma block-autoregressive generation. Tiny random-init config (no
# weights) so the test is fast: it exercises the OUTER-LOOP control flow (canvas
# chaining via cache-commit, max_tokens→block count, the max_canvases cap, and
# EOS early-stop), not output quality (quality is parity-verified separately).

import unittest

import mlx.core as mx

from mlx_lm.models import diffusion_gemma as dg


def _tiny_model(canvas_length=8):
    args = dg.ModelArgs(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=16,
        sliding_window=8,
        sliding_window_pattern=2,
        canvas_length=canvas_length,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=32,
        use_bidirectional_attention="all",
        final_logit_softcapping=30.0,
    )
    model = dg.Model(args)
    mx.eval(model.parameters())
    return model


class TestDiffusionGemmaGenerate(unittest.TestCase):
    PROMPT = mx.array([[1, 2, 3, 4, 5]])

    def test_single_canvas_default(self):
        # max_tokens unset → one canvas → (1, canvas_length); preserves slice-5 behaviour.
        m = _tiny_model()
        out = m.diffusion_generate(self.PROMPT, max_denoising_steps=4, key=mx.random.key(0))
        self.assertEqual(out.shape, (1, 8))

    def test_block_autoregressive_extends_beyond_one_canvas(self):
        # max_tokens=20, canvas=8 → ceil(20/8)=3 committed canvases → (1, 24).
        m = _tiny_model()
        out = m.diffusion_generate(
            self.PROMPT, max_tokens=20, max_denoising_steps=4, key=mx.random.key(0)
        )
        self.assertEqual(out.shape, (1, 24))

    def test_max_canvases_caps_block_count(self):
        m = _tiny_model()
        out = m.diffusion_generate(
            self.PROMPT, max_tokens=999, max_canvases=2,
            max_denoising_steps=3, key=mx.random.key(1),
        )
        self.assertEqual(out.shape, (1, 16))

    def test_eos_commit_stops_early(self):
        # Every vocab id marked EOS → the first committed canvas triggers the stop.
        m = _tiny_model()
        out = m.diffusion_generate(
            self.PROMPT, max_tokens=99, eos_token_ids=list(range(64)),
            max_denoising_steps=3, key=mx.random.key(2),
        )
        self.assertEqual(out.shape, (1, 8))

    def test_stream_yields_one_array_per_canvas(self):
        # The streaming generator yields one committed canvas per block; the collected
        # wrapper returns the same total. ceil(20/8) = 3 canvases.
        m = _tiny_model()
        chunks = list(
            m.diffusion_stream_generate(
                self.PROMPT, max_tokens=20, max_denoising_steps=3, key=mx.random.key(0)
            )
        )
        self.assertEqual(len(chunks), 3)
        for c in chunks:
            self.assertEqual(c.shape, (1, 8))
        full = m.diffusion_generate(
            self.PROMPT, max_tokens=20, max_denoising_steps=3, key=mx.random.key(0)
        )
        self.assertEqual(full.shape, (1, 24))


if __name__ == "__main__":
    unittest.main()
