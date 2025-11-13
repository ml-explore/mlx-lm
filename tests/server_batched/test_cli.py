# ABOUTME: Verifies CLI exposes continuous batching configuration flags.
# ABOUTME: Ensures argument parser supports enabling the new scheduler path.

import importlib.util
import pathlib
import sys
import types
import unittest

from .util import ensure_mlx_stub

ensure_mlx_stub()

PACKAGE_NAME = "mlx_lm.server_batched"
if PACKAGE_NAME not in sys.modules:
    sys.modules[PACKAGE_NAME] = types.ModuleType(PACKAGE_NAME)

_CONFIG_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "mlx_lm"
    / "server_batched"
    / "config.py"
)
_SPEC = importlib.util.spec_from_file_location(f"{PACKAGE_NAME}.config", _CONFIG_PATH)
CONFIG_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = CONFIG_MODULE
_SPEC.loader.exec_module(CONFIG_MODULE)

create_arg_parser = CONFIG_MODULE.create_arg_parser


class ContinuousBatchingCLIArgsTests(unittest.TestCase):
    def test_parser_exposes_continuous_batching_flags(self):
        parser = create_arg_parser()
        args = parser.parse_args([])

        self.assertFalse(args.enable_continuous_batching)
        self.assertEqual(args.max_num_seqs, 16)
        self.assertEqual(args.max_tokens_per_step, 4096)
        self.assertEqual(args.decode_unroll, 4)
        self.assertTrue(args.decode_unroll_safe)
        self.assertEqual(args.prefill_chunk, 256)
        self.assertEqual(args.prefill_ramp_chunk, 64)
        self.assertEqual(args.prefill_hybrid_threshold, 0)
        self.assertIsNone(args.prefill_ramp_budget_ms)
        self.assertFalse(args.metal_profiling)
        self.assertFalse(args.force_legacy_generator)
        self.assertEqual(args.attn_backend, "auto")
        self.assertEqual(args.kv_block_size, 16)
        self.assertEqual(args.kv_pool_blocks, "auto")
        self.assertEqual(args.kv_quant, "none")
        self.assertEqual(args.kv_quant_group_size, 64)
        self.assertEqual(args.paged_vec_width, "auto")
        self.assertEqual(args.paged_threads_per_head, "auto")

    def test_enable_flag_turns_on_continuous_batching(self):
        parser = create_arg_parser()
        args = parser.parse_args(["--enable-continuous-batching"])
        self.assertTrue(args.enable_continuous_batching)

    def test_force_legacy_generator_flag(self):
        parser = create_arg_parser()
        args = parser.parse_args(["--force-legacy-generator"])
        self.assertTrue(args.force_legacy_generator)

    def test_metal_profiling_flag(self):
        parser = create_arg_parser()
        args = parser.parse_args(["--metal-profiling"])
        self.assertTrue(args.metal_profiling)

    def test_decode_unroll_unsafe_flag(self):
        parser = create_arg_parser()
        args = parser.parse_args(["--decode-unroll-unsafe"])
        self.assertFalse(args.decode_unroll_safe)

    def test_attn_backend_flag(self):
        parser = create_arg_parser()
        args = parser.parse_args(["--attn-backend", "dense"])
        self.assertEqual(args.attn_backend, "dense")

    def test_prefill_ramp_budget_flag(self):
        parser = create_arg_parser()
        args = parser.parse_args(["--prefill-ramp-budget-ms", "200"])
        self.assertEqual(args.prefill_ramp_budget_ms, 200.0)

    def test_prefill_hybrid_flag(self):
        parser = create_arg_parser()
        args = parser.parse_args(["--prefill-hybrid-threshold", "48"])
        self.assertEqual(args.prefill_hybrid_threshold, 48)


if __name__ == "__main__":
    unittest.main()
