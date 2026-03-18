import unittest
from unittest.mock import MagicMock

import mlx.core as mx

from mlx_lm.thinking_budget import (
    FORCED_LOGIT_VALUE,
    MASKED_LOGIT_VALUE,
    ThinkingBudgetProcessor,
    has_open_think_block,
)

VOCAB_SIZE = 1000
THINK_START = 100
THINK_END = 101
EOS_ID = 2
EOS_ID_2 = 3


def _logits():
    return mx.zeros((1, VOCAB_SIZE))


def _make_args(thinking_budget):
    from mlx_lm.server import (
        GenerationArguments,
        LogitsProcessorArguments,
        ModelDescription,
        SamplingArguments,
    )

    return GenerationArguments(
        model=ModelDescription(model="test", draft=None, adapter=None),
        sampling=SamplingArguments(
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            xtc_probability=0.0,
            xtc_threshold=0.0,
        ),
        logits=LogitsProcessorArguments(
            logit_bias=None,
            repetition_penalty=0.0,
            repetition_context_size=20,
            presence_penalty=0.0,
            presence_context_size=20,
            frequency_penalty=0.0,
            frequency_context_size=20,
        ),
        stop_words=[],
        max_tokens=512,
        num_draft_tokens=0,
        logprobs=False,
        top_logprobs=-1,
        seed=None,
        chat_template_kwargs=None,
        thinking_budget=thinking_budget,
    )


def _make_handler(thinking_budget):
    # Bypass __init__ (requires a live socket) to test validate_model_parameters
    # in isolation. Must be kept in sync with do_POST's attribute assignments.
    from mlx_lm.server import APIHandler

    handler = object.__new__(APIHandler)
    handler.stream = False
    handler.max_tokens = 512
    handler.temperature = 1.0
    handler.top_p = 1.0
    handler.top_k = 0
    handler.min_p = 0.0
    handler.num_draft_tokens = 0
    handler.repetition_penalty = 0.0
    handler.repetition_context_size = 20
    handler.presence_penalty = 0.0
    handler.presence_context_size = 20
    handler.frequency_penalty = 0.0
    handler.frequency_context_size = 20
    handler.logprobs = False
    handler.top_logprobs = -1
    handler.logit_bias = None
    handler.xtc_probability = 0.0
    handler.xtc_threshold = 0.0
    handler.requested_model = "test-model"
    handler.adapter = None
    handler.seed = None
    handler.thinking_budget = thinking_budget
    return handler


def _thinking_tokenizer():
    tok = MagicMock()
    tok.has_thinking = True
    tok.think_start_id = THINK_START
    tok.think_end_id = THINK_END
    tok.eos_token_ids = {EOS_ID}
    return tok


def _non_thinking_tokenizer():
    tok = MagicMock()
    tok.has_thinking = False
    return tok


class TestThinkingBudgetProcessor(unittest.TestCase):
    def test_passthrough_when_no_think_token(self):
        proc = ThinkingBudgetProcessor(THINK_START, THINK_END, budget=10)
        tokens = mx.array([1, 2, 3])
        logits = _logits()
        result = proc(tokens, logits)
        self.assertTrue(mx.array_equal(result, logits))
        self.assertFalse(proc.in_thinking)

    def test_budget_forces_think_end_at_limit(self):
        proc = ThinkingBudgetProcessor(THINK_START, THINK_END, budget=3)

        proc(mx.array([1, THINK_START]), _logits())
        self.assertTrue(proc.in_thinking)
        self.assertEqual(proc.count, 0)

        proc(mx.array([1, THINK_START, 50]), _logits())
        self.assertEqual(proc.count, 1)
        proc(mx.array([1, THINK_START, 50, 51]), _logits())
        self.assertEqual(proc.count, 2)
        result = proc(mx.array([1, THINK_START, 50, 51, 52]), _logits())
        self.assertEqual(proc.count, 3)

        self.assertEqual(result[0, THINK_END].item(), FORCED_LOGIT_VALUE)
        self.assertEqual(result[0, 0].item(), MASKED_LOGIT_VALUE)
        self.assertEqual(result[0, 99].item(), MASKED_LOGIT_VALUE)

    def test_budget_zero_forces_immediate_close(self):
        proc = ThinkingBudgetProcessor(THINK_START, THINK_END, budget=0)
        result = proc(mx.array([1, THINK_START]), _logits())
        self.assertEqual(result[0, THINK_END].item(), FORCED_LOGIT_VALUE)
        self.assertEqual(result[0, 0].item(), MASKED_LOGIT_VALUE)

    def test_think_end_clears_thinking_state(self):
        proc = ThinkingBudgetProcessor(THINK_START, THINK_END, budget=10)
        proc(mx.array([1, THINK_START]), _logits())
        self.assertTrue(proc.in_thinking)
        proc(mx.array([1, THINK_START, THINK_END]), _logits())
        self.assertFalse(proc.in_thinking)

    def test_multiple_think_blocks_reset_count(self):
        proc = ThinkingBudgetProcessor(THINK_START, THINK_END, budget=2)

        proc(mx.array([THINK_START]), _logits())
        proc(mx.array([THINK_START, 50]), _logits())
        proc(mx.array([THINK_START, 50, THINK_END]), _logits())
        self.assertFalse(proc.in_thinking)

        proc(mx.array([THINK_START, 50, THINK_END, THINK_START]), _logits())
        self.assertTrue(proc.in_thinking)
        self.assertEqual(proc.count, 0)

    def test_natural_close_before_budget_no_force(self):
        proc = ThinkingBudgetProcessor(THINK_START, THINK_END, budget=100)
        proc(mx.array([THINK_START]), _logits())
        proc(mx.array([THINK_START, 50]), _logits())
        self.assertEqual(proc.count, 1)
        result = proc(mx.array([THINK_START, 50, THINK_END]), _logits())
        self.assertFalse(proc.in_thinking)
        self.assertNotEqual(result[0, 0].item(), MASKED_LOGIT_VALUE)

    def test_in_thinking_init_forces_immediate_close_at_budget_zero(self):
        """When prompt already contains <think>, processor starts in_thinking=True."""
        proc = ThinkingBudgetProcessor(
            THINK_START, THINK_END, budget=0, in_thinking=True
        )
        self.assertTrue(proc.in_thinking)
        # First generated token (not <think>) triggers budget enforcement
        result = proc(mx.array([50]), _logits())
        self.assertEqual(proc.count, 1)
        self.assertEqual(result[0, THINK_END].item(), FORCED_LOGIT_VALUE)
        self.assertEqual(result[0, 0].item(), MASKED_LOGIT_VALUE)

    def test_in_thinking_init_counts_from_zero(self):
        """With in_thinking=True and budget=3, allows 3 tokens before forcing."""
        proc = ThinkingBudgetProcessor(
            THINK_START, THINK_END, budget=3, in_thinking=True
        )
        # Tokens 1, 2 — under budget
        result = proc(mx.array([50]), _logits())
        self.assertEqual(proc.count, 1)
        self.assertTrue(mx.array_equal(result, _logits()))  # not forced
        result = proc(mx.array([50, 51]), _logits())
        self.assertEqual(proc.count, 2)
        self.assertTrue(mx.array_equal(result, _logits()))
        # Token 3 — hits budget
        result = proc(mx.array([50, 51, 52]), _logits())
        self.assertEqual(proc.count, 3)
        self.assertEqual(result[0, THINK_END].item(), FORCED_LOGIT_VALUE)

    def test_repr(self):
        proc = ThinkingBudgetProcessor(THINK_START, THINK_END, budget=42)
        self.assertEqual(repr(proc), "ThinkingBudgetProcessor(budget=42)")

    def test_think_end_in_eos_token_ids_raises(self):
        """think_end_id must not appear in eos_token_ids (forced close conflict)."""
        with self.assertRaises(ValueError):
            ThinkingBudgetProcessor(
                THINK_START,
                THINK_END,
                budget=10,
                eos_token_ids=frozenset({THINK_END, EOS_ID}),
            )


class TestHasOpenThinkBlock(unittest.TestCase):
    def test_open_think_block(self):
        prompt = [1, 2, THINK_START, 50, 51]
        self.assertTrue(has_open_think_block(prompt, THINK_START, THINK_END))

    def test_closed_think_block(self):
        prompt = [1, THINK_START, 50, THINK_END, 3]
        self.assertFalse(has_open_think_block(prompt, THINK_START, THINK_END))

    def test_no_think_tokens(self):
        prompt = [1, 2, 3, 4, 5]
        self.assertFalse(has_open_think_block(prompt, THINK_START, THINK_END))

    def test_empty_prompt(self):
        self.assertFalse(has_open_think_block([], THINK_START, THINK_END))

    def test_think_end_after_think_start(self):
        # Multiple blocks: <think>...</think><think>... — last is open
        prompt = [THINK_START, 50, THINK_END, THINK_START, 60]
        self.assertTrue(has_open_think_block(prompt, THINK_START, THINK_END))

    def test_think_end_is_last_token(self):
        prompt = [THINK_START, 50, THINK_END]
        self.assertFalse(has_open_think_block(prompt, THINK_START, THINK_END))


class TestFromPrompt(unittest.TestCase):
    def test_from_prompt_with_open_think(self):
        prompt = [1, THINK_START, 50]
        proc = ThinkingBudgetProcessor.from_prompt(
            THINK_START, THINK_END, budget=100, prompt=prompt
        )
        self.assertTrue(proc.in_thinking)
        self.assertEqual(proc.budget, 100)

    def test_from_prompt_with_closed_think(self):
        prompt = [1, THINK_START, 50, THINK_END, 3]
        proc = ThinkingBudgetProcessor.from_prompt(
            THINK_START, THINK_END, budget=100, prompt=prompt
        )
        self.assertFalse(proc.in_thinking)

    def test_from_prompt_none(self):
        proc = ThinkingBudgetProcessor.from_prompt(
            THINK_START, THINK_END, budget=100, prompt=None
        )
        self.assertFalse(proc.in_thinking)

    def test_from_prompt_no_think_tokens(self):
        proc = ThinkingBudgetProcessor.from_prompt(
            THINK_START, THINK_END, budget=100, prompt=[1, 2, 3]
        )
        self.assertFalse(proc.in_thinking)


class TestMakeLogitsProcessors(unittest.TestCase):
    def test_processor_appended_when_thinking_supported(self):
        from mlx_lm.server import _make_logits_processors

        args = _make_args(thinking_budget=100)
        processors = _make_logits_processors(args, _thinking_tokenizer())
        thinking_procs = [
            p for p in processors if isinstance(p, ThinkingBudgetProcessor)
        ]
        self.assertEqual(len(thinking_procs), 1)
        self.assertEqual(thinking_procs[0].budget, 100)

    def test_budget_zero_with_thinking_support_appends_processor(self):
        from mlx_lm.server import _make_logits_processors

        args = _make_args(thinking_budget=0)
        processors = _make_logits_processors(args, _thinking_tokenizer())
        thinking_procs = [
            p for p in processors if isinstance(p, ThinkingBudgetProcessor)
        ]
        self.assertEqual(len(thinking_procs), 1)
        self.assertEqual(thinking_procs[0].budget, 0)

    def test_no_processor_when_budget_is_none(self):
        from mlx_lm.server import _make_logits_processors

        args = _make_args(thinking_budget=None)
        processors = _make_logits_processors(args, _thinking_tokenizer())
        thinking_procs = [
            p for p in processors if isinstance(p, ThinkingBudgetProcessor)
        ]
        self.assertEqual(len(thinking_procs), 0)

    def test_warning_when_model_has_no_thinking_support(self):
        from mlx_lm.server import _make_logits_processors

        args = _make_args(thinking_budget=100)
        with self.assertLogs("root", level="WARNING") as cm:
            processors = _make_logits_processors(args, _non_thinking_tokenizer())
        thinking_procs = [
            p for p in processors if isinstance(p, ThinkingBudgetProcessor)
        ]
        self.assertEqual(len(thinking_procs), 0)
        self.assertTrue(any("no thinking support" in msg for msg in cm.output))

    def test_no_warning_when_tokenizer_is_none(self):
        from mlx_lm.server import _make_logits_processors

        args = _make_args(thinking_budget=100)
        with self.assertNoLogs("root", level="WARNING"):
            _make_logits_processors(args, tokenizer=None)

    def test_baseline_no_budget_no_tokenizer(self):
        from mlx_lm.server import _make_logits_processors

        args = _make_args(thinking_budget=None)
        with self.assertNoLogs("root", level="WARNING"):
            processors = _make_logits_processors(args, tokenizer=None)
        thinking_procs = [
            p for p in processors if isinstance(p, ThinkingBudgetProcessor)
        ]
        self.assertEqual(len(thinking_procs), 0)

    def test_prompt_with_open_think_sets_in_thinking(self):
        from mlx_lm.server import _make_logits_processors

        args = _make_args(thinking_budget=512)
        # Prompt ends with <think> followed by newline — no closing </think>
        prompt = [1, 2, 3, THINK_START, 10]  # 10 = newline or other token
        processors = _make_logits_processors(args, _thinking_tokenizer(), prompt)
        thinking_procs = [
            p for p in processors if isinstance(p, ThinkingBudgetProcessor)
        ]
        self.assertEqual(len(thinking_procs), 1)
        self.assertTrue(thinking_procs[0].in_thinking)

    def test_prompt_with_closed_think_block_not_in_thinking(self):
        from mlx_lm.server import _make_logits_processors

        args = _make_args(thinking_budget=512)
        # Prompt has <think>.....</think> — closed block
        prompt = [1, 2, THINK_START, 50, 51, THINK_END, 3]
        processors = _make_logits_processors(args, _thinking_tokenizer(), prompt)
        thinking_procs = [
            p for p in processors if isinstance(p, ThinkingBudgetProcessor)
        ]
        self.assertEqual(len(thinking_procs), 1)
        self.assertFalse(thinking_procs[0].in_thinking)

    def test_prompt_none_defaults_to_not_in_thinking(self):
        from mlx_lm.server import _make_logits_processors

        args = _make_args(thinking_budget=512)
        processors = _make_logits_processors(args, _thinking_tokenizer(), prompt=None)
        thinking_procs = [
            p for p in processors if isinstance(p, ThinkingBudgetProcessor)
        ]
        self.assertEqual(len(thinking_procs), 1)
        self.assertFalse(thinking_procs[0].in_thinking)

    def test_eos_token_ids_passed_to_processor(self):
        """eos_token_ids from the tokenizer are forwarded to the processor."""
        from mlx_lm.server import _make_logits_processors

        args = _make_args(thinking_budget=512)
        processors = _make_logits_processors(args, _thinking_tokenizer())
        thinking_procs = [
            p for p in processors if isinstance(p, ThinkingBudgetProcessor)
        ]
        self.assertEqual(len(thinking_procs), 1)
        self.assertEqual(thinking_procs[0].eos_token_ids, frozenset({EOS_ID}))


class TestThinkingBudgetValidation(unittest.TestCase):
    def test_negative_thinking_budget_raises(self):
        handler = _make_handler(thinking_budget=-1)
        with self.assertRaises(ValueError):
            handler.validate_model_parameters()

    def test_non_integer_thinking_budget_raises(self):
        handler = _make_handler(thinking_budget=1.5)
        with self.assertRaises(ValueError):
            handler.validate_model_parameters()

    def test_zero_thinking_budget_valid(self):
        _make_handler(thinking_budget=0).validate_model_parameters()

    def test_none_thinking_budget_valid(self):
        _make_handler(thinking_budget=None).validate_model_parameters()


class TestEosMaskingAfterThinkClose(unittest.TestCase):
    """Regression tests for EOS masking after any </think> token.

    After any ``</think>`` — whether budget-forced or the model's own natural
    close — the processor must suppress all EOS tokens for exactly one step.
    Without this guard models immediately output EOS after closing their
    thinking block, producing ``finish_reason=stop`` with no visible content.
    """

    def _make_proc(self, budget: int) -> ThinkingBudgetProcessor:
        return ThinkingBudgetProcessor(
            THINK_START,
            THINK_END,
            budget=budget,
            eos_token_ids=frozenset({EOS_ID}),
        )

    def test_forced_close_masks_eos_one_step(self):
        """EOS is suppressed on the first token after a budget-forced </think>."""
        proc = self._make_proc(budget=2)

        proc(mx.array([THINK_START]), _logits())  # enter thinking
        proc(mx.array([THINK_START, 50]), _logits())  # count=1 < 2
        proc(mx.array([THINK_START, 50, 51]), _logits())  # count=2, forces THINK_END

        # Model emits THINK_END (the forced token); processor sees it and
        # must return logits with EOS masked.
        result = proc(mx.array([THINK_START, 50, 51, THINK_END]), _logits())

        self.assertEqual(result[0, EOS_ID].item(), MASKED_LOGIT_VALUE)

    def test_forced_close_no_mask_second_step(self):
        """EOS masking is a single-token guard; the second step is unmasked."""
        proc = self._make_proc(budget=1)

        proc(mx.array([THINK_START]), _logits())
        proc(mx.array([THINK_START, 50]), _logits())  # forces THINK_END
        proc(mx.array([THINK_START, 50, THINK_END]), _logits())  # one-step guard

        # Second token after forced close: guard window is over
        result = proc(mx.array([THINK_START, 50, THINK_END, 200]), _logits())

        self.assertNotEqual(result[0, EOS_ID].item(), MASKED_LOGIT_VALUE)

    def test_natural_close_masks_eos_one_step(self):
        """Natural </think> (before budget) must ALSO trigger EOS masking."""
        proc = self._make_proc(budget=100)

        proc(mx.array([THINK_START]), _logits())
        proc(mx.array([THINK_START, 50]), _logits())
        # Model closes thinking voluntarily (budget not yet reached)
        result = proc(mx.array([THINK_START, 50, THINK_END]), _logits())

        self.assertEqual(result[0, EOS_ID].item(), MASKED_LOGIT_VALUE)

    def test_natural_close_no_mask_second_step(self):
        """Guard window is exactly one token for natural close too."""
        proc = self._make_proc(budget=100)

        proc(mx.array([THINK_START]), _logits())
        proc(mx.array([THINK_START, 50, THINK_END]), _logits())  # one-step guard

        result = proc(mx.array([THINK_START, 50, THINK_END, 200]), _logits())

        self.assertNotEqual(result[0, EOS_ID].item(), MASKED_LOGIT_VALUE)

    def test_multiple_eos_tokens_all_masked(self):
        """All EOS token IDs in the frozenset are masked, not just the first."""
        proc = ThinkingBudgetProcessor(
            THINK_START,
            THINK_END,
            budget=1,
            eos_token_ids=frozenset({EOS_ID, EOS_ID_2}),
        )
        proc(mx.array([THINK_START]), _logits())
        proc(mx.array([THINK_START, 50]), _logits())  # forces THINK_END
        result = proc(mx.array([THINK_START, 50, THINK_END]), _logits())

        self.assertEqual(result[0, EOS_ID].item(), MASKED_LOGIT_VALUE)
        self.assertEqual(result[0, EOS_ID_2].item(), MASKED_LOGIT_VALUE)

    def test_non_eos_tokens_not_affected(self):
        """Non-EOS logits are unchanged by the EOS masking step."""
        proc = self._make_proc(budget=1)
        proc(mx.array([THINK_START]), _logits())
        proc(mx.array([THINK_START, 50]), _logits())  # forces THINK_END
        result = proc(mx.array([THINK_START, 50, THINK_END]), _logits())

        for i in range(VOCAB_SIZE):
            if i == EOS_ID:
                continue
            self.assertEqual(
                result[0, i].item(),
                0.0,
                f"token {i} logit should be unchanged",
            )

    def test_eos_none_backward_compat(self):
        """Without eos_token_ids, </think> returns logits unchanged (backward compat)."""
        proc = ThinkingBudgetProcessor(THINK_START, THINK_END, budget=1)
        proc(mx.array([THINK_START]), _logits())
        proc(mx.array([THINK_START, 50]), _logits())  # forces THINK_END
        result = proc(mx.array([THINK_START, 50, THINK_END]), _logits())

        # All logits unchanged — no masking applied
        for i in range(VOCAB_SIZE):
            self.assertEqual(result[0, i].item(), 0.0)

    def test_forced_close_masks_high_eos_logit(self):
        """EOS masking overrides a high EOS logit (model strongly wants to stop)."""
        proc = self._make_proc(budget=1)
        proc(mx.array([THINK_START]), _logits())
        proc(mx.array([THINK_START, 50]), _logits())  # forces THINK_END

        high_eos_logits = mx.zeros((1, VOCAB_SIZE))
        high_eos_logits[0, EOS_ID] = 1000.0
        result = proc(mx.array([THINK_START, 50, THINK_END]), high_eos_logits)

        self.assertEqual(result[0, EOS_ID].item(), MASKED_LOGIT_VALUE)


class TestEosMaskingDuringThinking(unittest.TestCase):
    """Regression tests for EOS masking DURING thinking (in_thinking=True).

    Bug (observed 2026-03-18, Tasks 2/3/5 in SWE-bench):
    Model generated EOS while still inside the <think> block (before </think>),
    producing finish_reason=stop with completion_tokens=2-3 and empty content.
    Fixed by masking EOS in the ``elif self.in_thinking`` branch.
    """

    def _make_proc(self, budget: int) -> ThinkingBudgetProcessor:
        return ThinkingBudgetProcessor(
            THINK_START,
            THINK_END,
            budget=budget,
            eos_token_ids=frozenset({EOS_ID}),
        )

    def test_eos_must_be_masked_mid_thinking(self):
        """EOS must be suppressed while in_thinking=True.

        Without this guard, models can generate EOS before </think>, producing
        finish_reason=stop with 2-3 tokens and empty visible content.
        """
        proc = self._make_proc(budget=100)

        proc(mx.array([THINK_START]), _logits())
        proc(mx.array([THINK_START, 50]), _logits())  # count=1

        result = proc(mx.array([THINK_START, 50, 51]), _logits())

        self.assertEqual(result[0, EOS_ID].item(), MASKED_LOGIT_VALUE)

    def test_eos_masked_on_first_thinking_token(self):
        """EOS must be blocked from the very first thinking step."""
        proc = self._make_proc(budget=100)

        # Processor sees think_start_id → sets in_thinking=True, returns unmodified logits.
        # Next call: first real thinking token — EOS must be blocked.
        proc(mx.array([THINK_START]), _logits())
        result = proc(mx.array([THINK_START, 50]), _logits())

        self.assertEqual(result[0, EOS_ID].item(), MASKED_LOGIT_VALUE)


if __name__ == "__main__":
    unittest.main()
