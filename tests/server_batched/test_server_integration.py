# ABOUTME: Ensures server wiring injects continuous batching runtime config.
# ABOUTME: Verifies handler receives shared runtime state from run().

import argparse
import types
import unittest
from unittest.mock import patch

from .util import ensure_mlx_stub

ensure_mlx_stub()

import mlx_lm.server as server


class SpyHandler:
    captured_state = None

    def __init__(
        self,
        model_provider,
        *args,
        runtime_state=None,
        **kwargs,
    ):
        SpyHandler.captured_state = runtime_state


class SpyServer:
    def __init__(self, address, handler_factory):
        handler_factory(None, None, None)

    def serve_forever(self):
        return


class RunIntegrationTests(unittest.TestCase):
    @patch("mlx_lm.server.socket.getaddrinfo")
    def test_run_populates_runtime_config(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [(None, None, None, None, ("127.0.0.1", 0))]
        server.run(
            host="127.0.0.1",
            port=0,
            model_provider=types.SimpleNamespace(),
            server_class=SpyServer,
            handler_class=SpyHandler,
            args=argparse.Namespace(
                enable_continuous_batching=True,
                max_num_seqs=8,
                max_tokens_per_step=2048,
                prefill_chunk=512,
            ),
        )

        state = SpyHandler.captured_state
        self.assertIsNotNone(state)
        self.assertTrue(state["config"]["enabled"])
        self.assertEqual(state["config"]["max_num_seqs"], 8)
        self.assertEqual(state["config"]["prefill_chunk"], 512)

    @patch("mlx_lm.server.socket.getaddrinfo")
    def test_run_disables_runtime_when_flag_off(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [(None, None, None, None, ("127.0.0.1", 0))]
        server.run(
            host="127.0.0.1",
            port=0,
            model_provider=types.SimpleNamespace(),
            server_class=SpyServer,
            handler_class=SpyHandler,
            args=argparse.Namespace(enable_continuous_batching=False),
        )
        state = SpyHandler.captured_state
        self.assertIsNotNone(state)
        self.assertFalse(state["config"]["enabled"])


if __name__ == "__main__":
    unittest.main()
