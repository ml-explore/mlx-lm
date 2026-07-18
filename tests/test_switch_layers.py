import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.switch_layers import SwitchGLU


class TestSwitchLayers(unittest.TestCase):
    def test_quantized_switch_glu_ragged_sorted_tail(self):
        # ml-explore/mlx#3856: the sorted gather_qmm path corrupts expert
        # outputs for affine-quantized weights when the flattened row count
        # exceeds 32768 and is not a multiple of 64. 16401 tokens x top-2
        # gives n = 32802, which triggers the ragged tail; the pad guard in
        # _gather_sort must keep the output matching a dense fp32 reference.
        mx.random.seed(0)
        num_tokens, top_k = 16401, 2
        dims, hidden_dims, num_experts = 64, 64, 4

        glu = SwitchGLU(dims, hidden_dims, num_experts)
        nn.quantize(glu, group_size=64, bits=4)

        x = mx.random.normal((num_tokens, dims))
        indices = mx.random.randint(0, num_experts, (num_tokens, top_k))
        out = glu(x, indices)
        mx.eval(out)

        def dequant(proj):
            return mx.dequantize(
                proj.weight,
                proj.scales,
                proj.biases,
                group_size=proj.group_size,
                bits=proj.bits,
                mode=getattr(proj, "mode", "affine"),
            ).astype(mx.float32)

        w_gate = dequant(glu.gate_proj)
        w_up = dequant(glu.up_proj)
        w_down = dequant(glu.down_proj)

        xf = x.astype(mx.float32)
        per_expert = []
        for e in range(num_experts):
            h = nn.silu(xf @ w_gate[e].T) * (xf @ w_up[e].T)
            per_expert.append(h @ w_down[e].T)
        dense = mx.stack(per_expert, axis=1)
        ref = mx.take_along_axis(
            dense, indices[..., None], axis=1
        )
        mx.eval(ref)

        max_err = mx.abs(out.astype(mx.float32) - ref).max().item()
        # The gather_qmm bug produces errors ~0.5 on affected rows; a healthy
        # run agrees with the fp32 reference to well under 1e-3.
        self.assertLess(max_err, 1e-3)


if __name__ == "__main__":
    unittest.main()
