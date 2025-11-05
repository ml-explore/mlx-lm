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
_SPEC = importlib.util.spec_from_file_location(
    f"{PACKAGE_NAME}.config", _CONFIG_PATH
)
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
        self.assertEqual(args.prefill_chunk, 1024)

    def test_enable_flag_turns_on_continuous_batching(self):
        parser = create_arg_parser()
        args = parser.parse_args(["--enable-continuous-batching"])
        self.assertTrue(args.enable_continuous_batching)


if __name__ == "__main__":
    unittest.main()
