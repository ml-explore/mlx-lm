# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.batch_admission import AdmissionState, LinearStateCost, StateBudget
from mlx_lm.models import qwen3_next
from mlx_lm.models.cache import ArraysCache, KVCache


class TestStateBudget(unittest.TestCase):
    def test_real_hybrid_cache_maps_to_fixed_plus_growth(self):
        args = qwen3_next.ModelArgs(
            model_type="qwen3_next",
            hidden_size=128,
            num_hidden_layers=4,
            intermediate_size=128,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=1_000,
            linear_num_value_heads=4,
            linear_num_key_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=3,
            num_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            shared_expert_intermediate_size=128,
            mlp_only_layers=[0],
            moe_intermediate_size=128,
            rms_norm_eps=1e-5,
            head_dim=64,
            rope_theta=1_000.0,
            partial_rotary_factor=0.5,
            max_position_embeddings=2_000,
        )
        model = qwen3_next.Model(args)
        caches = model.make_cache()
        self.assertTrue(any(isinstance(c, ArraysCache) for c in caches))
        self.assertTrue(any(isinstance(c, KVCache) for c in caches))

        model(mx.ones((1, 256), dtype=mx.int32), cache=caches)
        mx.eval([c.state for c in caches])
        at_256 = sum(c.nbytes for c in caches)
        model(mx.ones((1, 256), dtype=mx.int32), cache=caches)
        mx.eval([c.state for c in caches])
        at_512 = sum(c.nbytes for c in caches)

        per_unit = (at_512 - at_256) / 256
        fixed = at_256 - per_unit * 256
        self.assertGreater(fixed, 0)
        self.assertGreater(per_unit, 0)
        cost = LinearStateCost(fixed, per_unit)
        self.assertEqual(cost(AdmissionState("hybrid", 512)), at_512)

    def test_hybrid_fixed_and_linear_state(self):
        cost = LinearStateCost(49_000_000, 6_144, max_units=32_768)
        policy = StateBudget(300_000_000, cost)

        short = AdmissionState("short", 4_096)
        long = AdmissionState("long", 65_536)
        self.assertEqual(policy.projected_bytes(short), 49_000_000 + 6_144 * 4_096)
        self.assertEqual(policy.projected_bytes(long), 49_000_000 + 6_144 * 32_768)

    def test_diffusion_timestep_dependent_peak(self):
        # A diffusion scheduler can project the largest remaining latent and
        # activation footprint. No token or KV-cache semantics are involved.
        activation_factors = (4, 3, 2, 1)

        def diffusion_peak(state):
            remaining = activation_factors[
                state.completed_units : state.projected_units
            ]
            factor = max(remaining, default=0)
            return state.metadata["latent_bytes"] * factor

        policy = StateBudget(850, diffusion_peak)
        early = AdmissionState(
            "diffusion-active",
            projected_units=4,
            completed_units=0,
            resident_bytes=100,
            metadata={"latent_bytes": 100},
        )
        late = AdmissionState(
            "diffusion-active",
            projected_units=4,
            completed_units=2,
            resident_bytes=100,
            metadata={"latent_bytes": 100},
        )
        candidate = AdmissionState(
            "candidate",
            projected_units=4,
            metadata={"latent_bytes": 150},
        )

        self.assertEqual(
            policy.admitted_prefix([candidate], live_bytes=100, active=[early]), 0
        )
        self.assertEqual(
            policy.admitted_prefix([candidate], live_bytes=100, active=[late]), 1
        )

    def test_exact_cohort_removal_and_liveness(self):
        policy = StateBudget(1_000, lambda state: state.metadata["peak_bytes"])
        old = AdmissionState("old", 1, metadata={"peak_bytes": 400})
        cheap = AdmissionState("cheap", 1, metadata={"peak_bytes": 200})
        expensive = AdmissionState("expensive", 1, metadata={"peak_bytes": 700})

        # The oldest request remains first, but the caller must account for
        # the exact reordered cohort rather than reusing a FIFO count.
        self.assertEqual(policy.admitted_prefix([old, cheap], live_bytes=0), 2)
        self.assertEqual(policy.admitted_prefix([old, expensive], live_bytes=0), 1)

        active = AdmissionState("active", 1, metadata={"peak_bytes": 500})
        self.assertEqual(
            policy.admitted_prefix([expensive], live_bytes=0, active=[active]), 0
        )
        # Removing the active request recomputes headroom from live state.
        self.assertEqual(policy.admitted_prefix([expensive], live_bytes=0), 1)

        oversized = AdmissionState("oversized", 1, metadata={"peak_bytes": 1_500})
        self.assertEqual(policy.admitted_prefix([oversized], live_bytes=0), 1)
        self.assertEqual(
            policy.admitted_prefix([oversized], live_bytes=0, active=[active]), 0
        )


if __name__ == "__main__":
    unittest.main()
