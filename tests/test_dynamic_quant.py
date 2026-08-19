# Copyright © 2026 Apple Inc.

import unittest
from unittest.mock import patch

from mlx_lm.quant.dynamic_quant import main


class TestDynamicQuant(unittest.TestCase):

    def test_target_bpw_at_low_precision_floor_fails_before_model_load(self):
        cases = [
            (["--target-bpw", "4.5"], 4.5),
            (
                [
                    "--target-bpw",
                    "4.0",
                    "--low-bits",
                    "3",
                    "--low-group-size",
                    "32",
                ],
                4.0,
            ),
        ]
        for arguments, minimum in cases:
            with self.subTest(arguments=arguments):
                with patch("sys.argv", ["dynamic_quant.py", *arguments]):
                    with patch("mlx_lm.quant.dynamic_quant.load") as load:
                        with patch(
                            "mlx_lm.quant.dynamic_quant.mx.distributed.init"
                        ) as distributed_init:
                            load.side_effect = AssertionError(
                                "model loading should not be reached"
                            )

                            with self.assertRaisesRegex(
                                ValueError,
                                rf"is at or below the minimum \({minimum:.4g}\)",
                            ):
                                main()

                            distributed_init.assert_not_called()
                            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
