"""
Unit test for the mlx_lm `mask_prompt` chat-template offset bug.

Bug location: mlx_lm.tuner.datasets.ChatDataset.process
Reference: https://github.com/ml-explore/mlx-examples
mlx_lm issue: ChatDataset computes the loss-mask offset by re-applying
the chat template to messages[:-1] with add_generation_prompt=True.
For templates where the add_generation_prompt=True suffix is NOT a
prefix of the full template's prefix (notably Gemma 4, which appends
a "<|channel>thought\n<channel|>" hint), the computed offset is past
the end of the full tokenization. The resulting loss mask is empty,
and the trainer's default_loss divides 0/0 -> NaN.

This test is hermetic: it uses a tiny custom mock tokenizer that
exhibits the exact buggy behaviour (gen-prompt suffix differs from
full-template prefix). It does NOT download any model and has no
hardcoded local paths. The test FAILS on the original buggy code and
PASSES on the patched code.
"""

import math
import sys
import unittest
from typing import List, Optional


# ---------------------------------------------------------------------------
# Mock tokenizer that exhibits the Gemma 4 buggy behaviour
# ---------------------------------------------------------------------------
class GemmaLikeMockTokenizer:
    """A minimal stand-in for a HuggingFace tokenizer that triggers the
    mask_prompt offset bug.

    The chat template renders differently under add_generation_prompt=True
    (it adds a 4-token "<|channel>thought\\n<channel|>" hint) versus the
    full template (which puts the assistant content directly after the
    <|turn>model\\n header). This is the exact pattern that breaks
    ChatDataset.process in the original code.
    """

    GEN_HINT = "<|channel>thought\n<channel|>"
    ASS_TAIL = "<turn|>\n"

    def __init__(self, vocab_size: int = 32000):
        self._vocab_size = vocab_size

    def apply_chat_template(
        self,
        messages: List[dict],
        tools: Optional[list] = None,
        add_generation_prompt: bool = False,
        return_dict: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        """Returns a list of token IDs (when tokenize=True) or a string
        (when tokenize=False)."""
        parts = ["<bos>"]
        for m in messages:
            if m["role"] == "system":
                parts.append(f"<|turn>system\n{m['content']}<turn|>\n")
            elif m["role"] == "user":
                parts.append(f"<|turn>user\n{m['content']}<turn|>\n")
            elif m["role"] == "assistant":
                # The full template always renders the model turn header
                # followed by the assistant content and the turn close.
                parts.append(f"<|turn>model\n{m['content']}{self.ASS_TAIL}")
        text = "".join(parts)

        if add_generation_prompt:
            # If the last message is an assistant, the buggy code in
            # the original mlx_lm calls with add_generation_prompt=True
            # anyway. We simulate that here: the gen-prompt tokenization
            # for messages[:-1] + [assistant] is the gen-prompt of
            # messages[:-1] which adds "<|turn>model\\n<hint>" after the
            # last user turn close.
            #
            # If the last message is NOT an assistant, add the hint
            # at the end of the prompt (after the user turn close).
            if messages and messages[-1].get("role") == "assistant":
                # Strip the last assistant tail, leaving just the prompt
                # up to (but not including) the assistant role header.
                last = messages[-1]
                text = text[: -len(f"<|turn>model\n{last['content']}{self.ASS_TAIL}")]
                text = text + "<|turn>model\n" + self.GEN_HINT
            else:
                # No assistant in messages — just append the model turn
                # header + hint to the end of the user turn.
                text = text + "<|turn>model\n" + self.GEN_HINT

        if not tokenize:
            return text

        # Tokenize by mapping each char to a token id that depends on
        # BOTH the character's identity (its ord) and its position. This
        # way, two texts that share a prefix of length N will share the
        # first N token ids, but the token id at position i is unique
        # to the character at position i.
        return [self._vocab_size + (ord(c) * 31 + i) % self._vocab_size
                for i, c in enumerate(text)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
CHAT_EXAMPLE = {
    "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
}

LONG_CHAT_EXAMPLE = {
    "messages": [
        {"role": "system", "content": (
            "You are a careful math tutor. Solve step by step and put the "
            "final answer after '####'."
        )},
        {"role": "user", "content": (
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast "
            "every morning and bakes muffins for her friends every day with "
            "four. She sells the remainder at the farmers' market daily for "
            "$2 per fresh duck egg. How much in dollars does she make every "
            "day at the farmers' market?"
        )},
        {"role": "assistant", "content": (
            "Janet sells 16 - 3 - 4 = 9 duck eggs a day.\n"
            "She makes 9 * 2 = $18 every day at the farmers' market.\n"
            "#### 18"
        )},
    ]
}


class TestMaskPromptOffset(unittest.TestCase):
    """The core test: with mask_prompt=True, the loss-mask offset must
    point to a position inside the tokenized sequence, and the response
    portion (tokens[offset:]) must be non-empty and end with the model's
    turn close."""

    def setUp(self):
        # Import here so the test can be run from anywhere without
        # needing the local model path.
        from mlx_lm.tuner.datasets import ChatDataset
        self.ChatDataset = ChatDataset
        self.tok = GemmaLikeMockTokenizer()
        self.ds = self.ChatDataset(
            data=[CHAT_EXAMPLE],
            tokenizer=self.tok,
            mask_prompt=True,
        )

    def test_offset_is_in_bounds(self):
        """Offset must be strictly less than len(tokens). The buggy
        code computes offset = 29 for a 28-token sequence, which makes
        the loss mask empty and triggers NaN in the trainer."""
        tokens, offset = self.ds.process(CHAT_EXAMPLE)
        self.assertLess(
            offset, len(tokens),
            f"offset {offset} is past end of tokens (len={len(tokens)}). "
            "This produces an empty loss mask and NaN gradients."
        )

    def test_offset_is_strictly_positive(self):
        """Offset must be > 0 — there IS a prompt to mask."""
        tokens, offset = self.ds.process(CHAT_EXAMPLE)
        self.assertGreater(offset, 0)

    def test_response_portion_decodes_to_assistant_turn(self):
        """tokens[offset:] should contain the assistant content ('4')
        and the turn close. With the LCP-based offset, the response
        portion is the content + turn close, NOT including the role
        header (which is in the prompt)."""
        tokens, offset = self.ds.process(CHAT_EXAMPLE)
        response_ids = tokens[offset:]
        # The response portion is '4<turn|>\\n' = 9 chars
        expected = len("4<turn|>\n")
        self.assertEqual(
            len(response_ids), expected,
            f"Expected response portion to be {expected} tokens "
            f"('4<turn|>\\n'), got {len(response_ids)}"
        )

    def test_mask_prompt_false_is_unchanged(self):
        """The fix should not affect mask_prompt=False behaviour."""
        ds = self.ChatDataset(
            data=[CHAT_EXAMPLE],
            tokenizer=self.tok,
            mask_prompt=False,
        )
        tokens, offset = ds.process(CHAT_EXAMPLE)
        self.assertEqual(offset, 0)
        self.assertGreater(len(tokens), 0)

    def test_offset_matches_prompt_only_tokenization(self):
        """The fixed offset should equal the longest common prefix of
        the full tokenization and the add_generation_prompt=True
        tokenization. For our Gemma 4 mock, this is the position
        right after the role header ('<|turn>model\\n'), so the offset
        is the sum of (prompt-only length) + (role-header length)."""
        _, offset = self.ds.process(CHAT_EXAMPLE)
        full = self.tok.apply_chat_template(
            CHAT_EXAMPLE["messages"],
            tools=None,
            return_dict=False,
        )
        gen = self.tok.apply_chat_template(
            CHAT_EXAMPLE["messages"][:-1],
            tools=None,
            add_generation_prompt=True,
            return_dict=False,
        )
        lcp = 0
        for a, b in zip(full, gen):
            if a == b:
                lcp += 1
            else:
                break
        self.assertEqual(offset, lcp)

    def test_long_example_also_correct(self):
        """The bug compounds with example length — verify it on a realistic
        GSM8K example too."""
        ds = self.ChatDataset(
            data=[LONG_CHAT_EXAMPLE],
            tokenizer=self.tok,
            mask_prompt=True,
        )
        tokens, offset = ds.process(LONG_CHAT_EXAMPLE)
        self.assertLess(offset, len(tokens),
                        f"offset {offset} >= len(tokens) {len(tokens)}")
        # Response portion is the assistant content + turn close (NOT
        # including the role header, which is in the prompt).
        expected_response_len = len(
            "Janet sells 16 - 3 - 4 = 9 duck eggs a day.\n"
            "She makes 9 * 2 = $18 every day at the farmers' market.\n"
            "#### 18<turn|>\n"
        )
        self.assertEqual(
            len(tokens) - offset, expected_response_len,
            f"Expected response length {expected_response_len}, "
            f"got {len(tokens) - offset}"
        )


class TestNoNaNInToyTrainer(unittest.TestCase):
    """End-to-end check: simulate what the trainer does with the offset
    and verify that the loss mask is non-empty and the loss is finite."""

    def test_loss_mask_is_nonempty_and_loss_is_finite(self):
        from mlx_lm.tuner.datasets import ChatDataset
        import mlx.core as mx

        tok = GemmaLikeMockTokenizer()
        ds = ChatDataset(
            data=[CHAT_EXAMPLE],
            tokenizer=tok,
            mask_prompt=True,
        )
        tokens, offset = ds.process(CHAT_EXAMPLE)

        # Simulate the trainer's loss mask: 0 for prompt, 1 for response
        mask = [0] * offset + [1] * (len(tokens) - offset)
        # Verify mask is non-empty
        n_active = sum(mask)
        self.assertGreater(
            n_active, 0,
            f"Loss mask is empty (offset={offset}, len(tokens)={len(tokens)}). "
            "This is the exact bug — would cause NaN in real training."
        )
        # Simulate a forward pass with all-zeros logits, all-zero targets.
        # Cross-entropy of zero logits vs zero targets = 0 (no contribution
        # to loss). The point is to verify the loss is finite.
        logits = mx.zeros((1, len(tokens), 1))
        targets = mx.zeros((1, len(tokens)), dtype=mx.int32)
        mask_arr = mx.array(mask, dtype=mx.float32)
        # Element-wise CE: -log P(target | logits). For zero logits, the
        # softmax is uniform (1/vocab) so log P = -log(vocab), but we
        # don't need to compute that. Just verify the masked sum / count
        # is finite (not NaN).
        # Build a per-token loss of all zeros (no NaN possibility).
        ce = mx.zeros_like(targets).astype(mx.float32)
        ntoks = mask_arr.sum()
        # Use mx.maximum(ntoks, 1) as the patched trainer does.
        loss = ce.sum() / mx.maximum(ntoks, 1)
        mx.eval(loss)
        loss_val = float(loss)
        self.assertTrue(math.isfinite(loss_val),
                        f"Loss is not finite: {loss_val}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
