# Copyright © 2026 Apple Inc.

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RUN_SLOW = os.getenv("RUN_SLOW_QWEN35_QUANT_TESTS")
MODEL = os.getenv("QWEN35_QUANT_TEST_MODEL", "Qwen/Qwen3.5-0.8B")


@unittest.skipUnless(RUN_SLOW, "set RUN_SLOW_QWEN35_QUANT_TESTS=1 to run")
class TestQwen35QuantizationSlow(unittest.TestCase):
    def run_command(self, args):
        subprocess.run(args, check=True)

    def assert_generates(self, model_path):
        self.run_command(
            [
                sys.executable,
                "-m",
                "mlx_lm",
                "generate",
                "--model",
                str(model_path),
                "--prompt",
                "Explain what tool calling is.",
                "--max-tokens",
                "32",
            ]
        )

    def test_qwen3_5_awq_dynamic_gptq_dwq_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)

            awq_path = out / "awq"
            self.run_command(
                [
                    sys.executable,
                    "-m",
                    "mlx_lm",
                    "awq",
                    "--model",
                    MODEL,
                    "--bits",
                    "4",
                    "--num-samples",
                    "4",
                    "--n-grid",
                    "2",
                    "--sequence-length",
                    "64",
                    "--mlx-path",
                    str(awq_path),
                ]
            )
            self.assert_generates(awq_path)

            dynamic_path = out / "dynamic"
            self.run_command(
                [
                    sys.executable,
                    "-m",
                    "mlx_lm",
                    "dynamic_quant",
                    "--model",
                    MODEL,
                    "--target-bpw",
                    "4.5",
                    "--num-samples",
                    "4",
                    "--sequence-length",
                    "64",
                    "--mlx-path",
                    str(dynamic_path),
                ]
            )
            self.assert_generates(dynamic_path)

            gptq_path = out / "gptq"
            self.run_command(
                [
                    sys.executable,
                    "-m",
                    "mlx_lm",
                    "gptq",
                    "--model",
                    MODEL,
                    "--bits",
                    "4",
                    "--num-samples",
                    "4",
                    "--sequence-length",
                    "64",
                    "--mlx-path",
                    str(gptq_path),
                ]
            )
            self.assert_generates(gptq_path)

            dwq_path = out / "dwq"
            self.run_command(
                [
                    sys.executable,
                    "-m",
                    "mlx_lm",
                    "dwq",
                    "--model",
                    MODEL,
                    "--bits",
                    "4",
                    "--group-size",
                    "64",
                    "--num-samples",
                    "4",
                    "--max-seq-length",
                    "64",
                    "--batch-size",
                    "1",
                    "--kl-loss-impl",
                    "mlx",
                    "--mlx-path",
                    str(dwq_path),
                ]
            )
            self.assert_generates(dwq_path)


if __name__ == "__main__":
    unittest.main()
