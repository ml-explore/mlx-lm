# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.generate import speculative_generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load


class TestSpeculativeRewindGuard(unittest.TestCase):
    """Regression test for the stale-``n`` cache rewind on early abort.

    ``speculative_generate_step`` rewinds the KV caches in a ``finally`` block
    using ``n`` (the number of accepted draft tokens). ``num_draft`` is
    recomputed at the top of every loop iteration and shrinks as generation
    approaches ``max_tokens``. If an exception fires after ``num_draft`` is
    recomputed but before ``n`` is reset, the ``finally`` rewinds with an ``n``
    left over from the previous iteration. When that stale ``n`` exceeds the
    new ``num_draft``, ``trim_prompt_cache`` is called with a negative count,
    which *increases* the cache offset (``offset -= negative``) and corrupts
    the cache.

    A cache rewind must never grow the cache offset. This test drives the
    generator into that window, injects a fault, and asserts the final offset
    does not exceed the offset observed at the moment of the fault.
    """

    @classmethod
    def setUpClass(cls):
        cls.HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        cls.model, cls.tokenizer = load(cls.HF_MODEL_PATH)
        cls.model.set_dtype(mx.float32)

    def test_rewind_never_grows_offset_on_early_abort(self):
        model = self.model
        # Same model as draft so every draft token is accepted (temp=0), which
        # drives ``n`` up to ``num_draft`` early, then shrinks ``num_draft``
        # below it on the final iteration.
        draft_model = model
        sampler = make_sampler(temp=0.0)

        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            add_generation_prompt=True,
        )

        # Independent caches; we hold a reference to the target (model) cache to
        # inspect its offset. ``speculative_generate_step`` slices this list as
        # ``[: len(model.layers)]`` for the target and the rest for the draft.
        model_cache = make_prompt_cache(model)
        draft_cache = make_prompt_cache(draft_model)
        prompt_cache = model_cache + draft_cache

        # Fault injection: the target cache offset only advances on the target
        # verify step (draft steps touch the draft cache only). We count target
        # advances and raise on the *second* one -- iteration 2, where
        # ``num_draft`` has shrunk to 1 while the stale ``n`` from iteration 1
        # is 3. At that point the target forward has already advanced the cache
        # but ``n`` has not yet been reset.
        state = {"prev_off": None, "target_steps": 0, "captured": None}

        def fault(tokens, logits):
            off = model_cache[0].offset
            if state["prev_off"] is not None and off > state["prev_off"]:
                state["target_steps"] += 1
                if state["target_steps"] == 2:
                    state["captured"] = off
                    raise RuntimeError("injected mid-verify fault")
            state["prev_off"] = off
            return logits

        gen = speculative_generate_step(
            mx.array(prompt),
            model,
            draft_model,
            num_draft_tokens=3,
            max_tokens=5,
            sampler=sampler,
            logits_processors=[fault],
            prompt_cache=prompt_cache,
        )

        with self.assertRaises(RuntimeError):
            for _ in gen:
                pass

        # The fault must actually have fired in the target-verify window.
        self.assertIsNotNone(
            state["captured"],
            "fault did not reach the second target verify step; "
            "scenario did not exercise the stale-n path",
        )

        final_off = model_cache[0].offset
        # The rewind in the finally block must never *grow* the cache offset.
        # On the buggy path it trims by a negative amount (num_draft - stale_n
        # = 1 - 3 = -2) and the offset increases past the injected fault point.
        self.assertLessEqual(
            final_off,
            state["captured"],
            f"cache rewind grew the offset from {state['captured']} to "
            f"{final_off}; stale-n negative trim corrupted the cache",
        )


if __name__ == "__main__":
    unittest.main()
