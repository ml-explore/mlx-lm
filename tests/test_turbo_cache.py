# Copyright © 2024 Apple Inc.
"""Integration tests for TurboQuantKVCache.

Run with:
    python -m pytest tests/test_turbo_cache.py -v

These tests do NOT require a GPU: shape/API checks run on CPU.
The numerical accuracy test is skipped on CPU (TurboQuant SDPA needs Metal).
"""

import math
import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.turbo_cache import (
    TurboQuantKVCache,
    _hadamard_transform,
    _turbo_encode,
)
from mlx_lm.models.cache import make_turbo_cache, KVCache


# ── WHT correctness ───────────────────────────────────────────────────────────


class TestHadamardTransform(unittest.TestCase):
    def test_orthogonality_d64(self):
        """WHT * WHT^T = D * I  (before normalization)."""
        D = 64
        x = mx.random.normal(shape=(1, 1, 1, D))
        # Apply twice without normalization: should get D * x
        h = _hadamard_transform(x, scale=1.0)
        hh = _hadamard_transform(h, scale=1.0)
        mx.eval(hh)
        self.assertTrue(mx.allclose(hh, x * D, atol=1e-3).item())

    def test_normalized_dot_product(self):
        """dot(WHT(a)/√D, WHT(b)/√D) == dot(a, b)  (isometry)."""
        D = 128
        a = mx.random.normal(shape=(D,))
        b = mx.random.normal(shape=(D,))
        inv_d = 1.0 / math.sqrt(D)
        a_rot = _hadamard_transform(a, scale=inv_d)
        b_rot = _hadamard_transform(b, scale=inv_d)
        dot_rot = float(mx.sum(a_rot * b_rot).item())
        dot_orig = float(mx.sum(a * b).item())
        mx.eval(a_rot, b_rot)
        self.assertAlmostEqual(dot_rot, dot_orig, places=2)


# ── Encoding shape checks ─────────────────────────────────────────────────────


class TestTurboEncode(unittest.TestCase):
    def _check_encode(self, D: int, bits: int):
        from mlx_lm.models.turbo_cache import _CODEBOOKS

        B, H, L = 1, 2, 8
        x = mx.random.normal(shape=(B, H, L, D))
        cb = _CODEBOOKS[bits]
        packed, scales = _turbo_encode(x, cb, bits)
        mx.eval(packed, scales)

        expected_u32 = D * bits // 32
        self.assertEqual(packed.shape, (B, H, L, expected_u32))
        self.assertEqual(packed.dtype, mx.uint32)
        self.assertEqual(scales.shape, (B, H, L, 1))

    def test_encode_3bit_d64(self):
        self._check_encode(64, 3)

    def test_encode_3bit_d128(self):
        self._check_encode(128, 3)

    def test_encode_4bit_d64(self):
        self._check_encode(64, 4)

    def test_encode_4bit_d128(self):
        self._check_encode(128, 4)


# ── TurboQuantKVCache API ─────────────────────────────────────────────────────


class TestTurboQuantKVCache(unittest.TestCase):
    def _make_kv(self, B, H, L, D):
        k = 0.1 * mx.random.normal(shape=(B, H, L, D))
        v = 0.1 * mx.random.normal(shape=(B, H, L, D))
        return k, v

    def test_prefill_returns_float(self):
        """During prefill (L > 1), update_and_fetch returns float16."""
        c = TurboQuantKVCache(bits=3)
        k, v = self._make_kv(1, 2, 16, 64)
        rk, rv = c.update_and_fetch(k, v)
        mx.eval(rk, rv)
        self.assertIsInstance(rk, mx.array)
        self.assertIsInstance(rv, mx.array)
        self.assertEqual(rk.shape[-2], 16)

    def test_generation_returns_tuple(self):
        """During generation (L == 1, D in {64,128}), returns (packed, scales)."""
        c = TurboQuantKVCache(bits=3)
        k, v = self._make_kv(1, 2, 1, 64)
        rk, rv = c.update_and_fetch(k, v)
        mx.eval(rk[0], rk[1], rv[0], rv[1])
        self.assertIsInstance(rk, tuple)
        self.assertEqual(len(rk), 2)  # (packed, scales)
        self.assertEqual(rk[0].dtype, mx.uint32)

    def test_prefill_then_generation(self):
        """Prefill then generation: transition compresses float16 tokens."""
        c = TurboQuantKVCache(bits=3)
        B, H, D = 1, 2, 64

        # Prefill 32 tokens
        kp, vp = self._make_kv(B, H, 32, D)
        c.update_and_fetch(kp, vp)
        self.assertFalse(c._in_generation)

        # Generation step: triggers transition
        kg, vg = self._make_kv(B, H, 1, D)
        rk, rv = c.update_and_fetch(kg, vg)
        mx.eval(rk[0], rk[1])
        self.assertTrue(c._in_generation)
        # Cache now holds 33 tokens
        self.assertEqual(rk[0].shape[-2], 33)
        self.assertEqual(c.offset, 33)

    def test_unsupported_dim_falls_back(self):
        """head_dim not in {64,128} returns float (graceful fallback)."""
        c = TurboQuantKVCache(bits=3)
        k, v = self._make_kv(1, 2, 1, 32)  # D=32 unsupported
        rk, rv = c.update_and_fetch(k, v)
        mx.eval(rk, rv)
        # Should return float, not tuple
        self.assertIsInstance(rk, mx.array)

    def test_offset_tracking(self):
        c = TurboQuantKVCache(bits=4)
        for _ in range(5):
            k, v = self._make_kv(1, 1, 1, 64)
            c.update_and_fetch(k, v)
        self.assertEqual(c.offset, 5)


# ── make_turbo_cache ──────────────────────────────────────────────────────────


class TestMakeTurboCache(unittest.TestCase):
    def _make_mock_model(self, n_attn: int, n_linear: int):
        """Simple mock with mixed attention and linear-attention layers."""

        class AttnLayer(nn.Module):
            is_linear = False

        class LinearLayer(nn.Module):
            is_linear = True

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Interleave: 3 linear + 1 attn pattern
                self.layers = []
                for _ in range(n_attn):
                    for _ in range(n_linear):
                        self.layers.append(LinearLayer())
                    self.layers.append(AttnLayer())

            def make_cache(self):
                from mlx_lm.models.cache import ArraysCache

                return [
                    ArraysCache(size=2) if l.is_linear else KVCache()
                    for l in self.layers
                ]

        return MockModel()

    def test_turbo_cache_replaces_kvcache(self):
        model = self._make_mock_model(n_attn=4, n_linear=3)
        caches = make_turbo_cache(model, bits=3, fp16_layers=0)
        turbo_count = sum(isinstance(c, TurboQuantKVCache) for c in caches)
        kvcache_count = sum(isinstance(c, KVCache) for c in caches)
        self.assertEqual(turbo_count, 4)
        self.assertEqual(kvcache_count, 0)

    def test_fp16_layers_kept(self):
        """First and last fp16_layers attention layers remain as KVCache."""
        model = self._make_mock_model(n_attn=6, n_linear=3)
        caches = make_turbo_cache(model, bits=3, fp16_layers=1)
        turbo_count = sum(isinstance(c, TurboQuantKVCache) for c in caches)
        kvcache_count = sum(isinstance(c, KVCache) for c in caches)
        self.assertEqual(kvcache_count, 2)  # first + last kept
        self.assertEqual(turbo_count, 4)

    def test_fp16_layers_exact_boundary(self):
        """When fp16_layers*2 == n_attn, no layers remain as turbo."""
        # 4 attention layers, fp16_layers=2 → first 2 + last 2 = all 4 are kept fp16
        model = self._make_mock_model(n_attn=4, n_linear=0)
        caches = make_turbo_cache(model, bits=3, fp16_layers=2)
        turbo_count = sum(isinstance(c, TurboQuantKVCache) for c in caches)
        kvcache_count = sum(isinstance(c, KVCache) for c in caches)
        self.assertEqual(turbo_count, 0)
        self.assertEqual(kvcache_count, 4)

    def test_fp16_layers_zero(self):
        """fp16_layers=0 means all attention layers are compressed (no fp16 kept)."""
        model = self._make_mock_model(n_attn=4, n_linear=0)
        caches = make_turbo_cache(model, bits=3, fp16_layers=0)
        turbo_count = sum(isinstance(c, TurboQuantKVCache) for c in caches)
        self.assertEqual(turbo_count, 4)


# ── Numerical accuracy (GPU only) ─────────────────────────────────────────────


@unittest.skipUnless(mx.metal.is_available(), "TurboQuant SDPA requires Metal GPU")
class TestTurboNumericalAccuracy(unittest.TestCase):
    """Compare TurboQuant SDPA output against float16 reference.

    The tolerance is deliberately loose (3e-2) since we are testing a lossy
    compression scheme, not bit-exact arithmetic.
    """

    def _run_turbo_sdpa(self, B, Hq, Hkv, Lk, D, bits):
        import math

        from mlx_lm.models.turbo_cache import _CODEBOOKS, _hadamard_transform, _turbo_encode

        mx.random.seed(0)
        Lq = 1
        q = 0.1 * mx.random.normal(shape=(B, Hq, Lq, D))
        k = 0.1 * mx.random.normal(shape=(B, Hkv, Lk, D))
        v = 0.1 * mx.random.normal(shape=(B, Hkv, Lk, D))
        scale = 1.0 / math.sqrt(D)

        cb = _CODEBOOKS[bits]
        # K is WHT-rotated (for score computation); V is NOT rotated
        # so the kernel output is directly in the original V space.
        k_packed, k_scales = _turbo_encode(k, cb, bits, rotate=True)
        v_packed, v_scales = _turbo_encode(v, cb, bits, rotate=False)

        # Q is pre-rotated in Python (float32 precision) before the kernel.
        # The kernel receives Q_rot and K_rot (from encoding); the score
        # dot(Q_rot, K_rot) = dot(Q, K) by WHT isometry.
        q_rot = _hadamard_transform(q.astype(mx.float32), scale=1.0 / math.sqrt(D))
        q_rot = q_rot.astype(q.dtype)

        out_turbo = mx.fast.quantized_scaled_dot_product_attention(
            q_rot,
            k_packed,
            k_scales,
            v_packed,
            v_scales,
            scale=scale,
            mode=f"turbo{bits}",
            group_size=D,
        )
        mx.eval(out_turbo)

        # Reference: float SDPA with WHT-rotated Q/K and original V.
        k_rot = _hadamard_transform(k.astype(mx.float32), scale=1.0 / math.sqrt(D))
        k_rot = k_rot.astype(k.dtype)
        ref = mx.fast.scaled_dot_product_attention(q_rot, k_rot, v, scale=scale)
        mx.eval(ref)

        return out_turbo, ref

    def test_accuracy_3bit_d64(self):
        out, ref = self._run_turbo_sdpa(1, 4, 1, 64, 64, 3)
        err = float(mx.abs(out - ref).max().item())
        self.assertLess(err, 3e-2, f"3-bit D=64 max error {err:.4f} > 3e-2")

    def test_accuracy_3bit_d128(self):
        # GQA=4: matches Qwen3-14B / Qwen3.6-35B-A3B config
        out, ref = self._run_turbo_sdpa(1, 8, 2, 128, 128, 3)
        err = float(mx.abs(out - ref).max().item())
        self.assertLess(err, 3e-2, f"3-bit D=128 max error {err:.4f} > 3e-2")

    def test_accuracy_3bit_d256_gqa6(self):
        # GQA=6: matches Qwen3.6-27B (24Q/4KV heads, head_dim=256)
        out, ref = self._run_turbo_sdpa(1, 24, 4, 64, 256, 3)
        err = float(mx.abs(out - ref).max().item())
        self.assertLess(err, 3e-2, f"3-bit D=256 max error {err:.4f} > 3e-2")

    def test_accuracy_4bit_d128(self):
        out, ref = self._run_turbo_sdpa(1, 4, 1, 128, 128, 4)
        err = float(mx.abs(out - ref).max().item())
        self.assertLess(err, 2e-2, f"4-bit D=128 max error {err:.4f} > 2e-2")

    def test_output_shape(self):
        out, ref = self._run_turbo_sdpa(1, 4, 1, 64, 64, 3)
        self.assertEqual(out.shape, ref.shape)


@unittest.skipUnless(mx.metal.is_available(), "TurboQuant SDPA requires Metal GPU")
class TestQwen36_27B(unittest.TestCase):
    """End-to-end smoke test matching Qwen3.6-27B's exact attention config.

    Architecture: head_dim=256, 24Q/4KV heads (GQA=6), 64 layers.
    This test does NOT require the actual weights — it exercises the Metal
    kernel with the same tensor shapes and dtypes used during generation.
    """

    MODEL_ID = "mlx-community/Qwen3.6-27B-4bit"

    @classmethod
    def setUpClass(cls):
        import os

        if not os.path.exists(
            os.path.expanduser(
                "~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-4bit"
            )
        ):
            raise unittest.SkipTest("Qwen3.6-27B-4bit not downloaded")

    def test_turbo3_qwen36_27b_shapes(self):
        """Kernel correctness at Qwen3.6-27B attention geometry."""
        import math

        from mlx_lm.models.turbo_cache import (
            _CODEBOOKS,
            _hadamard_transform,
            _turbo_encode,
        )

        B, Hq, Hkv, Lk, D, bits = 1, 24, 4, 128, 256, 3
        scale = 1.0 / math.sqrt(D)
        cb = _CODEBOOKS[bits]
        mx.random.seed(0)

        q = 0.1 * mx.random.normal(shape=(B, Hq, 1, D)).astype(mx.bfloat16)
        k = 0.1 * mx.random.normal(shape=(B, Hkv, Lk, D)).astype(mx.bfloat16)
        v = 0.1 * mx.random.normal(shape=(B, Hkv, Lk, D)).astype(mx.bfloat16)

        k_packed, k_scales = _turbo_encode(k, cb, bits, rotate=True)
        v_packed, v_scales = _turbo_encode(v, cb, bits, rotate=False)

        q_rot = _hadamard_transform(q.astype(mx.float32), scale=1 / math.sqrt(D))
        q_rot = q_rot.astype(q.dtype)
        k_rot = _hadamard_transform(k.astype(mx.float32), scale=1 / math.sqrt(D))
        k_rot = k_rot.astype(k.dtype)

        ref = mx.fast.scaled_dot_product_attention(q_rot, k_rot, v, scale=scale)
        out = mx.fast.quantized_scaled_dot_product_attention(
            q_rot, k_packed, k_scales, v_packed, v_scales,
            scale=scale, mode="turbo3", group_size=D,
        )
        mx.eval(out, ref)

        self.assertEqual(out.shape, (B, Hq, 1, D))
        err = float(mx.abs(out - ref).max().item())
        self.assertLess(err, 5e-2, f"D=256 gqa=6 max error {err:.4f} > 5e-2")

    def test_turbo3_qwen36_27b_generation(self):
        """Full generation loop with TurboQuantKVCache on Qwen3.6-27B."""
        from mlx_lm import load
        from mlx_lm.generate import generate_step

        model, tokenizer = load(self.MODEL_ID)
        mx.eval(model.parameters())

        tokens = tokenizer.encode("Hello", return_tensors="mlx")[0]
        out_tokens = []
        for tok, _ in generate_step(
            tokens, model, max_tokens=10, kv_cache_type="turbo3"
        ):
            t = tok if isinstance(tok, int) else int(tok.item())
            out_tokens.append(t)
            mx.eval(tok)

        self.assertEqual(len(out_tokens), 10)
        text = tokenizer.decode(out_tokens)
        self.assertGreater(len(text.strip()), 0)


if __name__ == "__main__":
    unittest.main()
