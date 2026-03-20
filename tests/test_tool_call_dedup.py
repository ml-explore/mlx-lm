"""Unit tests for ToolCallDedup."""

import logging
import unittest

from mlx_lm.tool_call_dedup import ToolCallDedup


class TestToolCallDedup(unittest.TestCase):
    """Consecutive duplicate tool call detection."""

    def test_first_call_never_duplicate(self):
        dedup = ToolCallDedup()
        self.assertFalse(dedup.is_duplicate('{"name": "run", "arguments": {}}'))

    def test_different_calls_not_duplicate(self):
        dedup = ToolCallDedup()
        self.assertFalse(dedup.is_duplicate('{"name": "run", "arguments": {"cmd": "ls"}}'))
        self.assertFalse(dedup.is_duplicate('{"name": "run", "arguments": {"cmd": "pwd"}}'))

    def test_consecutive_identical_is_duplicate(self):
        dedup = ToolCallDedup()
        text = '{"name": "run", "arguments": {"cmd": "ls"}}'
        self.assertFalse(dedup.is_duplicate(text))
        self.assertTrue(dedup.is_duplicate(text))

    def test_non_consecutive_identical_not_duplicate(self):
        """A-B-A pattern should NOT trigger (only consecutive)."""
        dedup = ToolCallDedup()
        a = '{"name": "run", "arguments": {"cmd": "ls"}}'
        b = '{"name": "run", "arguments": {"cmd": "pwd"}}'
        self.assertFalse(dedup.is_duplicate(a))
        self.assertFalse(dedup.is_duplicate(b))
        self.assertFalse(dedup.is_duplicate(a))  # not consecutive

    def test_whitespace_difference_not_duplicate(self):
        """Exact text comparison — whitespace matters."""
        dedup = ToolCallDedup()
        self.assertFalse(dedup.is_duplicate('{"name":"run"}'))
        self.assertFalse(dedup.is_duplicate('{"name": "run"}'))

    def test_logs_warning_on_duplicate(self, ):
        dedup = ToolCallDedup()
        text = '{"name": "run", "arguments": {}}'
        dedup.is_duplicate(text)
        with self.assertLogs("mlx_lm.tool_call_dedup", level="WARNING") as cm:
            dedup.is_duplicate(text)
        self.assertTrue(any("duplicate" in msg.lower() for msg in cm.output))

    def test_prev_not_updated_on_duplicate(self):
        """After duplicate detected, prev stays the same for next check."""
        dedup = ToolCallDedup()
        text = '{"name": "run", "arguments": {}}'
        dedup.is_duplicate(text)
        self.assertTrue(dedup.is_duplicate(text))
        # Third consecutive should also be duplicate
        self.assertTrue(dedup.is_duplicate(text))

    def test_prev_updates_on_new_call(self):
        dedup = ToolCallDedup()
        a = '{"name": "a"}'
        b = '{"name": "b"}'
        dedup.is_duplicate(a)
        dedup.is_duplicate(b)
        # Now b-b should be duplicate
        self.assertTrue(dedup.is_duplicate(b))


if __name__ == "__main__":
    unittest.main()
