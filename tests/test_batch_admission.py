# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.batch_admission import (
    AdmissionState,
    LinearStateCost,
    StateBudget,
    StepStateCost,
)
from mlx_lm.models import qwen3_next
from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache


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
        arrays_256 = sum(c.nbytes for c in caches if isinstance(c, ArraysCache))
        kv_256 = sum(c.nbytes for c in caches if isinstance(c, KVCache))
        model(mx.ones((1, 1), dtype=mx.int32), cache=caches)
        mx.eval([c.state for c in caches])
        at_257 = sum(c.nbytes for c in caches)
        arrays_257 = sum(c.nbytes for c in caches if isinstance(c, ArraysCache))
        kv_257 = sum(c.nbytes for c in caches if isinstance(c, KVCache))
        model(mx.ones((1, 255), dtype=mx.int32), cache=caches)
        mx.eval([c.state for c in caches])
        at_512 = sum(c.nbytes for c in caches)
        arrays_512 = sum(c.nbytes for c in caches if isinstance(c, ArraysCache))
        kv_512 = sum(c.nbytes for c in caches if isinstance(c, KVCache))
        model(mx.ones((1, 256), dtype=mx.int32), cache=caches)
        mx.eval([c.state for c in caches])
        at_768 = sum(c.nbytes for c in caches)
        arrays_768 = sum(c.nbytes for c in caches if isinstance(c, ArraysCache))
        kv_768 = sum(c.nbytes for c in caches if isinstance(c, KVCache))

        per_unit = (at_512 - at_256) / 256
        fixed = at_256 - per_unit * 256
        self.assertGreater(fixed, 0)
        self.assertGreater(per_unit, 0)
        self.assertEqual(arrays_256, arrays_257)
        self.assertEqual(arrays_257, arrays_512)
        self.assertEqual(arrays_256, arrays_512)
        self.assertEqual(arrays_512, arrays_768)
        self.assertEqual(at_257, at_512)
        self.assertGreater(kv_257, kv_256)
        self.assertEqual(kv_257, kv_512)
        self.assertEqual(kv_512 - kv_256, kv_768 - kv_512)
        cost = LinearStateCost(fixed, per_unit, allocation_step_units=256)
        self.assertGreaterEqual(cost(AdmissionState("hybrid", 256)), at_256)
        self.assertEqual(cost(AdmissionState("hybrid", 257)), at_257)
        self.assertGreaterEqual(cost(AdmissionState("hybrid", 768)), at_768)

    def test_hybrid_fixed_and_linear_state(self):
        cost = LinearStateCost(49_000_000, 6_144, max_units=32_768)
        policy = StateBudget(300_000_000, cost)

        short = AdmissionState("short", 4_096)
        long = AdmissionState("long", 65_536)
        self.assertEqual(policy.projected_bytes(short), 49_000_000 + 6_144 * 4_096)
        self.assertEqual(policy.projected_bytes(long), 49_000_000 + 6_144 * 32_768)

    def test_stepped_projection_covers_allocation_boundaries(self):
        cost = LinearStateCost(7, 3, allocation_step_units=256)
        for units, allocated_upper_bound in (
            (0, 0),
            (255, 510),
            (256, 511),
            (257, 512),
            (511, 766),
            (512, 767),
            (513, 768),
        ):
            self.assertEqual(
                cost(AdmissionState(units, units)),
                7 + 3 * allocated_upper_bound,
            )

    def test_cohort_projection_uses_shared_rounded_maximum(self):
        cost = LinearStateCost(10, 2, allocation_step_units=256)
        short = AdmissionState("short", 1)
        long = AdmissionState("long", 1_000)

        self.assertEqual(cost.cohort_bytes([short, long]), 2 * (10 + 2 * 1_255))
        self.assertGreater(cost.cohort_bytes([short, long]), cost(short) + cost(long))
        self.assertEqual(cost.cohort_bytes([]), 0)

    def test_unaligned_continued_cache_uses_step_minus_one_envelope(self):
        history = KVCache()
        values = mx.zeros((1, 1, 252, 1), dtype=mx.float16)
        history.update_and_fetch(values, values)
        batch = BatchKVCache.merge([history])

        one = mx.zeros((1, 1, 1, 1), dtype=mx.float16)
        batch.update_and_fetch(one, one)
        mx.eval(batch.keys, batch.values)

        # Merging preserves logical width 252. Appending one token allocates
        # another 256-unit block, so logical 253 occupies capacity 508 rather
        # than round_up(253, 256) == 256.
        self.assertEqual(batch.keys.shape[2], 508)
        bytes_per_unit = 2 * mx.float16.size
        cost = LinearStateCost(0, bytes_per_unit, allocation_step_units=256)
        self.assertEqual(cost(AdmissionState("continued", 253)), batch.nbytes)
        self.assertGreater(batch.nbytes, bytes_per_unit * 256)

    def test_linear_cost_allocation_geometry_validation(self):
        for invalid in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                LinearStateCost(0, 1, allocation_step_units=invalid)
            with self.assertRaises(ValueError):
                LinearStateCost(0, 1, max_units=invalid)

    def test_diffusion_timestep_dependent_peak(self):
        # LLaDA-style block diffusion: each entry is the total peak for one
        # remaining block/denoising quantum, including the full logits canvas
        # and optional prefix/suffix cache. No token or KV-growth semantics are
        # imposed by the admission layer.
        policy = StateBudget(850, StepStateCost())
        schedule = (400, 300, 200, 100)
        early = AdmissionState(
            "diffusion-active",
            projected_units=4,
            completed_units=0,
            resident_bytes=100,
            metadata={"state_bytes_by_step": schedule},
        )
        late = AdmissionState(
            "diffusion-active",
            projected_units=4,
            completed_units=2,
            resident_bytes=100,
            metadata={"state_bytes_by_step": schedule},
        )
        candidate = AdmissionState(
            "candidate",
            projected_units=4,
            metadata={"state_bytes_by_step": (600, 450, 300, 150)},
        )

        self.assertEqual(
            policy.admitted_prefix([candidate], live_bytes=100, active=[early]), 0
        )
        self.assertEqual(
            policy.admitted_prefix([candidate], live_bytes=100, active=[late]), 1
        )

    def test_diffusion_schedule_validation(self):
        cost = StepStateCost()
        with self.assertRaisesRegex(ValueError, "requires 'state_bytes_by_step'"):
            cost(AdmissionState("missing", 1))
        with self.assertRaisesRegex(ValueError, "fewer than projected_units"):
            cost(
                AdmissionState(
                    "short-schedule", 2, metadata={"state_bytes_by_step": (10,)}
                )
            )
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            cost(
                AdmissionState(
                    "nonfinite",
                    2,
                    metadata={"state_bytes_by_step": (10, float("nan"))},
                )
            )

    def test_work_units_require_integers(self):
        for value in (True, 1.5):
            with self.assertRaises((TypeError, ValueError)):
                AdmissionState("invalid", value)

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

    def test_budget_mutation_remains_validated(self):
        policy = StateBudget(1_000, lambda state: 0)
        policy.budget_bytes = 2_000
        self.assertEqual(policy.budget_bytes, 2_000)
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                policy.budget_bytes = invalid
            self.assertEqual(policy.budget_bytes, 2_000)


if __name__ == "__main__":
    unittest.main()
