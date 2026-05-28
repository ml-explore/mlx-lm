# Copyright © 2024 Apple Inc.

import json
import unittest
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.models.recurrent_profile import profile_recurrent_call


class TestRecurrentProfile(unittest.TestCase):
    def test_disabled_profile_returns_result_without_output(self):
        with patch.dict("os.environ", {}, clear=True), patch("sys.stderr") as stderr:
            result = profile_recurrent_call(
                op="toy",
                path="path",
                metadata={"T": 1},
                fn=lambda: mx.array([1]),
            )
            mx.eval(result)

        self.assertTrue(bool(mx.array_equal(result, mx.array([1]))))
        stderr.write.assert_not_called()

    def test_enabled_profile_emits_json(self):
        with patch.dict("os.environ", {"MLX_LM_PROFILE_RECURRENT": "1"}):
            with patch("sys.stderr") as stderr:
                result = profile_recurrent_call(
                    op="toy",
                    path="path",
                    metadata={"T": 2},
                    fn=lambda: mx.array([2]),
                )
                mx.eval(result)

        payload = json.loads(stderr.write.call_args_list[0].args[0])
        self.assertEqual(payload["event"], "recurrent_profile")
        self.assertEqual(payload["op"], "toy")
        self.assertEqual(payload["path"], "path")
        self.assertEqual(payload["T"], 2)
        self.assertIn("elapsed_ms", payload)


if __name__ == "__main__":
    unittest.main()
