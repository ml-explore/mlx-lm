# Copyright © 2026 Apple Inc.

import unittest

from mlx_lm.generate import TextStateMachine
from mlx_lm.server_v2.engine import protocol


class TestProtocol(unittest.TestCase):

    def test_update_from_events(self):
        tokens = protocol.TokensEvent(token_ids=(1,), delta=TextStateMachine.Delta())
        usage = protocol.Usage(prompt_tokens=1, cached_tokens=0, completion_tokens=1)
        end = protocol.FinishedEvent(reason=protocol.FinishReason.EOS, usage=usage)
        self.assertTrue(protocol.Update.from_events([]).is_empty)
        update = protocol.Update.from_events([tokens, tokens])
        self.assertEqual(update, protocol.Update([tokens, tokens]))
        self.assertFalse(update.is_empty)
        self.assertEqual(
            protocol.Update.from_events([tokens, end]),
            protocol.Update([tokens], end),
        )
        self.assertEqual(protocol.Update.from_events([end]), protocol.Update([], end))
        failure = RuntimeError("x")
        with self.assertRaises(RuntimeError) as e:
            protocol.Update.from_events([tokens, protocol.FailedEvent(error=failure)])
        self.assertIs(e.exception, failure)
        # An event after the end means a bug in the engine.
        with self.assertRaises(AssertionError):
            protocol.Update.from_events([end, tokens])


if __name__ == "__main__":
    unittest.main()
