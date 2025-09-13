import unittest

import mlx.core as mx

from mlx_lm.models.gated_delta import (
    gated_delta_prefill_ops,
    gated_delta_step_ops,
)


class TestGatedDelta(unittest.TestCase):
    def test_step_matches_manual(self):
        B, H, Dk, Dv = 2, 3, 4, 5
        q = mx.random.uniform(shape=(B, H, Dk))
        k = mx.random.uniform(shape=(B, H, Dk))
        v = mx.random.uniform(shape=(B, H, Dv))
        g = mx.random.uniform(shape=(B, H))
        beta = mx.random.uniform(shape=(B, H))
        state = mx.random.uniform(shape=(B, H, Dk, Dv))

        y_op, st_op = gated_delta_step_ops(q, k, v, g, beta, state)

        st = state * g[..., None, None]
        kv_mem = (st * k[..., :, None]).sum(axis=-2)
        delta = (v - kv_mem) * beta[..., None]
        st = st + k[..., :, None] * delta[..., None, :]
        y = (st * q[..., :, None]).sum(axis=-2)

        self.assertTrue(mx.allclose(y_op, y))
        self.assertTrue(mx.allclose(st_op, st))

    def test_prefill_matches_step_loop(self):
        B, H, T, Dk, Dv = 1, 2, 6, 4, 3
        Q = mx.random.uniform(shape=(B, H, T, Dk))
        K = mx.random.uniform(shape=(B, H, T, Dk))
        V = mx.random.uniform(shape=(B, H, T, Dv))
        G = mx.random.uniform(shape=(B, H, T))
        BETA = mx.random.uniform(shape=(B, H, T))
        state = mx.zeros((B, H, Dk, Dv))

        Y1, S1 = gated_delta_prefill_ops(Q, K, V, G, BETA, state)

        S = state
        ys = []
        for t in range(T):
            y, S = gated_delta_step_ops(
                Q[..., t, :], K[..., t, :], V[..., t, :], G[..., t], BETA[..., t], S
            )
            ys.append(y)
        Y2 = mx.stack(ys, axis=2)

        self.assertTrue(mx.allclose(Y1, Y2))
        self.assertTrue(mx.allclose(S1, S))


if __name__ == "__main__":
    unittest.main()
