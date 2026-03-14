from typing import List, Optional

import mlx.core as mx

FORCED_LOGIT_VALUE = 1e9
MASKED_LOGIT_VALUE = float("-inf")


def has_open_think_block(
    prompt: List[int], think_start_id: int, think_end_id: int
) -> bool:
    """Check if a prompt ends with an unclosed ``<think>`` block.

    Scans backwards from the end of the prompt looking for the most recent
    ``<think>`` or ``</think>`` token.  Returns ``True`` if the last one
    found is ``<think>`` (i.e. the block is still open).
    """
    for i in range(len(prompt) - 1, -1, -1):
        if prompt[i] == think_end_id:
            return False
        elif prompt[i] == think_start_id:
            return True
    return False


class ThinkingBudgetProcessor:
    """Logits processor that enforces a token budget on <think> blocks.

    Signature matches mlx_lm's logits_processors protocol:
    ``(tokens: mx.array, logits: mx.array) -> mx.array``
    """

    def __init__(
        self,
        think_start_id: int,
        think_end_id: int,
        budget: int,
        *,
        in_thinking: bool = False,
    ) -> None:
        self.think_start_id = think_start_id
        self.think_end_id = think_end_id
        self.budget = budget
        self.in_thinking: bool = in_thinking
        self.count: int = 0

    @classmethod
    def from_prompt(
        cls,
        think_start_id: int,
        think_end_id: int,
        budget: int,
        prompt: Optional[List[int]] = None,
    ) -> "ThinkingBudgetProcessor":
        """Create a processor, detecting if the prompt starts inside a
        ``<think>`` block (e.g. Qwen3 chat templates include ``<think>``
        in the assistant preamble)."""
        in_thinking = (
            has_open_think_block(prompt, think_start_id, think_end_id)
            if prompt is not None
            else False
        )
        return cls(
            think_start_id, think_end_id, budget, in_thinking=in_thinking
        )

    def __repr__(self) -> str:
        return f"ThinkingBudgetProcessor(budget={self.budget})"

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        last_token = tokens[-1].item()

        if last_token == self.think_start_id:
            self.in_thinking = True
            self.count = 0
        elif last_token == self.think_end_id:
            self.in_thinking = False
        elif self.in_thinking:
            self.count += 1

        if self.in_thinking and self.count >= self.budget:
            forced = mx.full(logits.shape, MASKED_LOGIT_VALUE)
            forced[:, self.think_end_id] = FORCED_LOGIT_VALUE
            return forced

        return logits
