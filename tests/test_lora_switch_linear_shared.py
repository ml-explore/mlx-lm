"""Tests: LoRASwitchLinear supports PEFT shared 2D lora_a/lora_b (target_parameters)."""
import unittest

import mlx.core as mx

from mlx_lm.models.switch_layers import SwitchLinear
from mlx_lm.tuner.lora import LoRASwitchLinear


class TestLoRASwitchLinearSharedAdapter(unittest.TestCase):
    E = 4
    D = 16
    H = 16
    r = 4

    def _base(self):
        return SwitchLinear(self.D, self.H, self.E, bias=False)

    def test_fuse_per_expert(self):
        """Original (E, r, D) / (E, out, r) layout still works."""
        base = self._base()
        lora = LoRASwitchLinear.from_base(base, r=self.r, scale=1.0)
        lora.lora_a = mx.random.normal(shape=(self.E, self.r, self.D))
        lora.lora_b = mx.random.normal(shape=(self.E, self.H, self.r))
        fused = lora.fuse()
        self.assertEqual(fused.weight.shape, (self.E, self.H, self.D))

    def test_fuse_shared_2d(self):
        """PEFT shared (r, D) / (out, r) layout fuses to same E experts."""
        base = self._base()
        lora = LoRASwitchLinear.from_base(base, r=self.r, scale=1.0)
        a = mx.random.normal(shape=(self.r, self.D))
        b = mx.random.normal(shape=(self.H, self.r))
        lora.lora_a = a
        lora.lora_b = b
        fused = lora.fuse()
        self.assertEqual(fused.weight.shape, (self.E, self.H, self.D))
        # Each expert receives the SAME (B @ A) update on top of its base weight.
        delta = (b @ a).astype(fused.weight.dtype)
        for i in range(self.E):
            self.assertTrue(
                mx.allclose(
                    fused.weight[i] - base.weight[i], delta, atol=1e-4
                ).item()
            )

    def test_call_shared_2d_matches_fuse_then_call(self):
        """Forward pass with shared adapter equals fuse-then-call equivalent."""
        base = self._base()
        lora = LoRASwitchLinear.from_base(base, r=self.r, scale=1.0)
        a = mx.random.normal(shape=(self.r, self.D))
        b = mx.random.normal(shape=(self.H, self.r))
        lora.lora_a = a
        lora.lora_b = b

        x = mx.random.normal(shape=(2, 1, self.D))
        indices = mx.array([[0], [1]], dtype=mx.uint32)

        y_lora = lora(x, indices)
        fused = lora.fuse()
        y_fused = fused(x, indices)
        self.assertTrue(mx.allclose(y_lora, y_fused, atol=1e-3).item())


    def test_mixed_ndim_raises(self):
        """Mixed 2D/3D adapters must raise a clear ValueError, not silently
        take a partial code path."""
        base = self._base()
        lora = LoRASwitchLinear.from_base(base, r=self.r, scale=1.0)
        # 2D lora_a paired with 3D lora_b -> invalid.
        lora.lora_a = mx.random.normal(shape=(self.r, self.D))
        lora.lora_b = mx.random.normal(shape=(self.E, self.H, self.r))
        x = mx.random.normal(shape=(2, 1, self.D))
        indices = mx.array([[0], [1]], dtype=mx.uint32)
        with self.assertRaises(ValueError):
            lora(x, indices)
        with self.assertRaises(ValueError):
            lora.fuse()


if __name__ == "__main__":
    unittest.main()
