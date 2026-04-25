# Copyright © 2023-2024 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.sample_utils import make_thinking_budget_processor

# Token IDs used throughout all tests
THINK_START = (1,)  # <think>
THINK_END = (2,)  # </think>
STOP_TOKENS = mx.array([2, 3])  # early-stop injection: </think> then \n
VOCAB = 10


def _logits():
    """Uniform logits over VOCAB tokens, shape (1, VOCAB)."""
    return mx.ones((1, VOCAB))


class TestThinkingBudgetProcessor(unittest.TestCase):

    # ------------------------------------------------------------------
    # 1. Passthrough when not in thinking state
    # ------------------------------------------------------------------
    def test_passthrough_not_in_thinking(self):
        proc = make_thinking_budget_processor(THINK_START, THINK_END, 5, STOP_TOKENS)
        # No tokens generated yet; never entered thinking state.
        tokens = mx.array([])
        logits = _logits()
        result = proc(tokens, logits)
        self.assertTrue(mx.array_equal(result, logits))

    # ------------------------------------------------------------------
    # 2. Passthrough during thinking below budget
    # ------------------------------------------------------------------
    def test_passthrough_below_budget(self):
        proc = make_thinking_budget_processor(THINK_START, THINK_END, 5, STOP_TOKENS)
        # Enter thinking, generate 3 tokens (budget=5, so still under)
        base = [THINK_START[0], 10, 20, 30]
        for i, tok in enumerate(base):
            tokens = mx.array(base[: i + 1])
            logits = _logits()
            result = proc(tokens, logits)
            self.assertTrue(mx.array_equal(result, logits))

    # ------------------------------------------------------------------
    # 3. Forces first injection token when budget reached
    # ------------------------------------------------------------------
    def test_forces_first_token_at_budget(self):
        proc = make_thinking_budget_processor(THINK_START, THINK_END, 3, STOP_TOKENS)
        # Pump: <think> + 3 thinking tokens (budget=3, now exhausted)
        seq = [THINK_START[0], 10, 20, 30]
        for i in range(len(seq)):
            tokens = mx.array(seq[: i + 1])
            logits = _logits()
            result = proc(tokens, logits)

        # Last call should have forced STOP_TOKENS[0]
        forced_id = STOP_TOKENS[0].item()
        self.assertAlmostEqual(result[0, forced_id].item(), 0.0)
        # Every other token should be -inf
        for v in range(VOCAB):
            if v != forced_id:
                self.assertEqual(result[0, v].item(), float("-inf"))

    # ------------------------------------------------------------------
    # 4. Forces full injection sequence across calls
    # ------------------------------------------------------------------
    def test_forces_full_sequence(self):
        proc = make_thinking_budget_processor(THINK_START, THINK_END, 2, STOP_TOKENS)
        # <think> + 2 thinking tokens exhausts the budget.
        # The call that hits the budget itself returns the first forced logit.
        seq = [THINK_START[0], 10, 20]
        forced_results = []
        for i in range(len(seq)):
            tokens = mx.array(seq[: i + 1])
            result = proc(tokens, _logits())
            if i == len(seq) - 1:
                # Last pump call triggered forcing; capture its result.
                forced_results.append(result)

        # Subsequent calls with forced tokens appended should force the rest.
        stop = STOP_TOKENS.tolist()
        for call_idx in range(1, len(stop)):
            tokens = mx.array(seq + stop[:call_idx])
            result = proc(tokens, _logits())
            forced_results.append(result)

        # Each call_idx should have forced STOP_TOKENS[call_idx].
        for call_idx, forced_id in enumerate(stop):
            r = forced_results[call_idx]
            self.assertAlmostEqual(r[0, forced_id].item(), 0.0)
            for v in range(VOCAB):
                if v != forced_id:
                    self.assertEqual(r[0, v].item(), float("-inf"))

    # ------------------------------------------------------------------
    # 5. Returns to passthrough after injection complete
    # ------------------------------------------------------------------
    def test_passthrough_after_injection(self):
        proc = make_thinking_budget_processor(THINK_START, THINK_END, 2, STOP_TOKENS)
        seq = [THINK_START[0], 10, 20]
        # Exhaust budget
        for i in range(len(seq)):
            proc(mx.array(seq[: i + 1]), _logits())
        # Drain the injection sequence
        injected = seq + STOP_TOKENS.tolist()
        for i in range(len(STOP_TOKENS)):
            proc(mx.array(injected[: len(seq) + i]), _logits())
        # After injection, logits should pass through unmodified
        full = mx.array(injected)
        logits = _logits()
        result = proc(full, logits)
        self.assertTrue(mx.array_equal(result, logits))

    # ------------------------------------------------------------------
    # 6. No intervention without any thinking tokens
    # ------------------------------------------------------------------
    def test_no_intervention_without_think_start(self):
        proc = make_thinking_budget_processor(THINK_START, THINK_END, 2, STOP_TOKENS)
        # Generate many tokens, never entering thinking state
        tokens = mx.array([5, 6, 7, 8, 9, 10, 11])
        logits = _logits()
        result = proc(tokens, logits)
        self.assertTrue(mx.array_equal(result, logits))

    # ------------------------------------------------------------------
    # 7. Natural think_end resets count (no intervention)
    # ------------------------------------------------------------------
    def test_natural_end_resets_no_intervention(self):
        proc = make_thinking_budget_processor(THINK_START, THINK_END, 10, STOP_TOKENS)
        # Enter thinking, generate 3 tokens, then close naturally
        seq = [THINK_START[0], 10, 20, 30, THINK_END[0]]
        for i in range(len(seq)):
            tokens = mx.array(seq[: i + 1])
            logits = _logits()
            result = proc(tokens, logits)
            # No token should be masked
            self.assertTrue(mx.array_equal(result, logits))
        # After </think>, we are no longer in thinking state
        tokens = mx.array(seq + [99])
        logits = _logits()
        result = proc(tokens, logits)
        self.assertTrue(mx.array_equal(result, logits))


class TestMakeLogitsProcessors(unittest.TestCase):

    def test_thinking_budget_creates_processor(self):
        from mlx_lm.sample_utils import make_logits_processors
        processors = make_logits_processors(
            thinking_budget=100,
            think_start_tokens=(100,),
            think_end_tokens=(101,),
            early_stop_tokens=mx.array([50, 51, 101]),
        )
        self.assertEqual(len(processors), 1)

    def test_thinking_budget_none_skips(self):
        from mlx_lm.sample_utils import make_logits_processors
        processors = make_logits_processors()
        self.assertEqual(len(processors), 0)


class TestEarlyStopPromptBuilder(unittest.TestCase):

    def test_build_default_message(self):
        from mlx_lm.sample_utils import build_early_stop_tokens

        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [200 + i for i in range(len(text.split()))]

        result = build_early_stop_tokens(MockTokenizer(), (101,))
        self.assertIsInstance(result, mx.array)
        self.assertEqual(result[-1].item(), 101)
        self.assertGreater(len(result), 1)

    def test_build_custom_message(self):
        from mlx_lm.sample_utils import build_early_stop_tokens

        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [300, 301]

        result = build_early_stop_tokens(MockTokenizer(), (101,), message="Stop now.")
        self.assertEqual(len(result), 3)  # 300, 301, 101
        self.assertEqual(result[-1].item(), 101)


if __name__ == "__main__":
    unittest.main()
