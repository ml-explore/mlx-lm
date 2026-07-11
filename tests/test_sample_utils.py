import math
import unittest

import mlx.core as mx

from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p, apply_xtc


class TestSampleUtils(unittest.TestCase):
    def test_apply_top_p(self):
        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)

        new_logits = apply_top_p(logits, 0.3)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertEqual(actual_probs.tolist(), [1.0, 0.0, 0.0, 0.0])

        new_logits = apply_top_p(logits, 0.95)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertTrue(mx.allclose(probs.squeeze(), actual_probs))

        probs = mx.array([0.0, 0.5, 0.4, 0.1])[None]
        logits = mx.log(probs)
        new_logits = apply_top_p(logits, 0.4)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertEqual(actual_probs.tolist(), [0.0, 1.0, 0.0, 0.0])

        new_logits = apply_top_p(logits, 0.6)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertEqual(
            [round(p, 4) for p in actual_probs.tolist()], [0.0, 0.5556, 0.4444, 0.0]
        )

        new_logits = apply_top_p(logits, 0.95)
        actual_probs = mx.softmax(new_logits.squeeze())
        actual_rounded = [round(p, 4) for p in actual_probs.tolist()]
        expected_rounded = [0.0, 0.5, 0.4, 0.1]
        self.assertEqual(actual_rounded, expected_rounded)
        self.assertAlmostEqual(sum(actual_probs.tolist()), 1.0)

        # Batch mode works
        probs = mx.array([[0.9, 0.0, 0.0, 0.1], [0.0, 0.8, 0.1, 0.1]])
        logits = mx.log(probs)
        new_logits = apply_top_p(logits, 0.5)
        actual_probs = mx.softmax(new_logits, axis=-1)
        self.assertEqual(
            actual_probs.tolist(), [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )

    def test_apply_min_p(self):
        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)
        new_logits = apply_min_p(logits, 0.8)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertEqual(actual_probs.tolist(), [1.0, 0.0, 0.0, 0.0])

        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)
        new_logits = apply_min_p(logits, 0.05)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertTrue(mx.allclose(actual_probs, mx.squeeze(probs)))

        # Batch mode works
        probs = mx.array([[0.9, 0.0, 0.0, 0.1], [0.0, 0.8, 0.0, 0.1]])
        logits = mx.log(probs)
        new_logits = apply_min_p(logits, 0.7)
        actual_probs = mx.softmax(new_logits, axis=-1)
        self.assertEqual(
            actual_probs.tolist(), [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )

    def test_apply_top_k(self):
        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)

        new_logits = apply_top_k(logits, 1)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertEqual(actual_probs.tolist(), [1.0, 0.0, 0.0, 0.0])

        probs = mx.array([0.6, 0.0, 0.1, 0.3])[None]
        logits = mx.log(probs)
        new_logits = apply_top_k(logits, 2)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertEqual(
            [round(p, 4) for p in actual_probs.tolist()], [0.6667, 0.0, 0.0, 0.3333]
        )

        # Batch mode works
        probs = mx.array([[0.9, 0.0, 0.0, 0.1], [0.0, 0.8, 0.0, 0.1]])
        logits = mx.log(probs)

        new_logits = apply_top_k(logits, 1)
        actual_probs = mx.softmax(new_logits, axis=-1)
        self.assertEqual(
            actual_probs.tolist(), [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )

    def test_apply_xtc(self):
        # Test the threshold
        probs = mx.array([[0.4, 0.3, 0.15, 0.15]])
        new_probs = mx.softmax(apply_xtc(mx.log(probs), 1, 0.2, []), -1)
        expected = mx.array([[0, 0.5, 0.25, 0.25]])
        self.assertTrue(mx.allclose(new_probs, expected))
        probs = mx.array([[0.4, 0.3, 0.15, 0.15]])
        new_probs = mx.softmax(apply_xtc(mx.log(probs), 1, 0.1, []), -1)
        expected = mx.array([[0, 0.0, 0.5, 0.5]])
        self.assertTrue(mx.allclose(new_probs, expected))

        # Test the special tokens
        probs = mx.array([[0.4, 0.3, 0.15, 0.15]])
        new_probs = mx.softmax(apply_xtc(mx.log(probs), 1, 0.1, [0]), -1)
        expected = mx.array([[4 / 7, 0.0, 1.5 / 7, 1.5 / 7]])
        self.assertTrue(mx.allclose(new_probs, expected))

        # Test that with probability 0 the probs don't change
        probs = mx.array([[0.4, 0.3, 0.15, 0.15]])
        new_probs = mx.softmax(apply_xtc(mx.log(probs), 0, 0.1, [0]), -1)
        self.assertTrue(mx.allclose(new_probs, probs))

    def test_presence_penalty(self):
        from mlx_lm.sample_utils import make_presence_penalty

        # Token appears multiple times - penalty applied once
        tokens = mx.array([0, 0, 0, 1, 1])
        logits = mx.zeros((1, 4))
        processor = make_presence_penalty(0.5, context_size=5)
        result = processor(tokens, logits)
        # Token 0 appears 3 times, token 1 appears 2 times - both penalized once
        self.assertAlmostEqual(result[0, 0].item(), -0.5)
        self.assertAlmostEqual(result[0, 1].item(), -0.5)
        # Tokens not in context not penalized
        self.assertAlmostEqual(result[0, 2].item(), 0.0)
        self.assertAlmostEqual(result[0, 3].item(), 0.0)

    def test_frequency_penalty(self):
        from mlx_lm.sample_utils import make_frequency_penalty

        # Token appears multiple times - penalty applied proportionally
        tokens = mx.array([0, 0, 0, 1, 1])
        logits = mx.zeros((1, 4))
        processor = make_frequency_penalty(0.5, context_size=5)
        result = processor(tokens, logits)
        # Token 0 appears 3 times -> 3 * 0.5 = 1.5 penalty
        self.assertAlmostEqual(result[0, 0].item(), -1.5)
        # Token 1 appears 2 times -> 2 * 0.5 = 1.0 penalty
        self.assertAlmostEqual(result[0, 1].item(), -1.0)
        # Tokens not in context not penalized
        self.assertAlmostEqual(result[0, 2].item(), 0.0)
        self.assertAlmostEqual(result[0, 3].item(), 0.0)

    def test_reasoning_budget_hard_cap(self):
        from mlx_lm.sample_utils import make_reasoning_budget

        close, vocab = 5, 10
        proc = make_reasoning_budget(think_close=close, max_think_tokens=8)
        logits = mx.zeros((1, vocab))

        toks = []
        # 7 tokens inside the (implicitly open) channel: under budget, untouched
        for i in range(7):
            toks.append(i % 4)  # content ids, never the close id
            out = proc(mx.array(toks), logits)
            self.assertTrue(mx.all(out == logits).item())
        # 8th token hits the budget -> close is forced
        toks.append(3)
        out = proc(mx.array(toks), logits)
        self.assertEqual(int(mx.argmax(out[0]).item()), close)
        self.assertEqual(out[0, close].item(), 0.0)
        self.assertTrue(math.isinf(out[0, 0].item()))

    def test_reasoning_budget_resets_on_natural_close(self):
        from mlx_lm.sample_utils import make_reasoning_budget

        proc = make_reasoning_budget(think_close=5, max_think_tokens=4)
        logits = mx.zeros((1, 8))
        toks = []
        # Channel closes on its own after two tokens (id 5)...
        for t in [0, 1, 5]:
            toks.append(t)
            proc(mx.array(toks), logits)
        # ...so subsequent tokens are outside it and never trigger a force,
        # even well past the budget.
        for t in range(10):
            toks.append(t % 3)
            out = proc(mx.array(toks), logits)
            self.assertTrue(mx.all(out == logits).item())

    def test_reasoning_budget_open_token_gates(self):
        from mlx_lm.sample_utils import make_reasoning_budget

        proc = make_reasoning_budget(think_close=5, max_think_tokens=4, think_open=4)
        logits = mx.zeros((1, 8))
        toks = []
        # Before the open token, content does not count toward the budget.
        for _ in range(6):
            toks.append(0)
            out = proc(mx.array(toks), logits)
            self.assertTrue(mx.all(out == logits).item())
        # Open the channel, then 4 tokens hit the budget.
        toks.append(4)
        proc(mx.array(toks), logits)
        forced = None
        for i in range(4):
            toks.append(i % 2)  # 0/1, not the open/close ids
            forced = proc(mx.array(toks), logits)
        self.assertEqual(int(mx.argmax(forced[0]).item()), 5)

    def test_reasoning_budget_token_cycle(self):
        from mlx_lm.sample_utils import make_reasoning_budget

        proc = make_reasoning_budget(
            think_close=5, max_think_tokens=10_000, check_every=4
        )
        logits = mx.zeros((1, 10))
        toks, out = [], None
        for _ in range(20):  # a period-2 loop that never terminates
            toks += [7, 8]
            out = proc(mx.array(toks), logits)
        self.assertEqual(int(mx.argmax(out[0]).item()), 5)

    def test_reasoning_budget_line_repetition(self):
        from mlx_lm.sample_utils import make_reasoning_budget

        class _Tok:
            def decode(self, ids):
                return "this is the same reasoning line\n" * 4

        proc = make_reasoning_budget(
            think_close=5, max_think_tokens=10_000, tokenizer=_Tok(), check_every=1
        )
        logits = mx.zeros((1, 10))
        toks, out = [], None
        for i in range(3):
            toks.append(i)
            out = proc(mx.array(toks), logits)
        self.assertEqual(int(mx.argmax(out[0]).item()), 5)

    def test_reasoning_budget_callable_matches_static(self):
        # A constant callable budget must close at the exact same step as the
        # static int (regression pin for the static path).
        from mlx_lm.sample_utils import make_reasoning_budget

        close, vocab, budget = 5, 10, 8
        static = make_reasoning_budget(think_close=close, max_think_tokens=budget)
        dynamic = make_reasoning_budget(
            think_close=close, max_think_tokens=lambda: budget
        )
        logits = mx.zeros((1, vocab))
        toks = []
        for i in range(budget):
            toks.append(i % 4)
            out_s = static(mx.array(toks), logits)
            out_d = dynamic(mx.array(toks), logits)
            self.assertTrue(mx.all(out_s == out_d).item())
            if i < budget - 1:
                self.assertTrue(mx.all(out_s == logits).item())
        # Both force the close on the same (budget-th) step.
        self.assertEqual(int(mx.argmax(out_s[0]).item()), close)
        self.assertEqual(int(mx.argmax(out_d[0]).item()), close)

    def test_reasoning_budget_callable_dynamic(self):
        # A live budget is re-evaluated each step: shrinking it mid-generation
        # trips the close earlier than its starting value would.
        from mlx_lm.sample_utils import make_reasoning_budget

        close, vocab = 5, 10
        budget = {"v": 100}
        proc = make_reasoning_budget(
            think_close=close,
            max_think_tokens=lambda: budget["v"],
            check_every=10**9,  # isolate the budget path
        )
        logits = mx.zeros((1, vocab))
        toks = []
        for i in range(4):
            toks.append(i % 3)
            out = proc(mx.array(toks), logits)
            self.assertTrue(mx.all(out == logits).item())
        budget["v"] = 3  # cost spiked: tighten below the 4 tokens already spent
        toks.append(1)
        out = proc(mx.array(toks), logits)
        self.assertEqual(int(mx.argmax(out[0]).item()), close)

    def test_reasoning_budget_dynamic_monotonic_close(self):
        # (a) A trip latches: a budget that grows again before the forced
        # close token is emitted cannot un-trip it. (b) After the channel
        # closes, a shrinking budget cannot re-force outside the channel.
        from mlx_lm.sample_utils import make_reasoning_budget

        close, vocab = 5, 10
        budget = {"v": 100}
        proc = make_reasoning_budget(
            think_close=close,
            max_think_tokens=lambda: budget["v"],
            check_every=10**9,
        )
        logits = mx.zeros((1, vocab))
        toks = [0, 1, 2]
        proc(mx.array(toks), logits)
        budget["v"] = 2  # trip
        toks.append(0)
        out = proc(mx.array(toks), logits)
        self.assertEqual(int(mx.argmax(out[0]).item()), close)
        budget["v"] = 10**6  # brake released before the close was emitted...
        toks.append(1)  # ...and the stream somehow carried content
        out = proc(mx.array(toks), logits)
        # ...the trip is latched: still forcing the close.
        self.assertEqual(int(mx.argmax(out[0]).item()), close)
        # The channel now actually closes; shrink the budget to its minimum.
        toks.append(close)
        proc(mx.array(toks), logits)
        budget["v"] = 1
        for t in range(6):
            toks.append(t % 3)
            out = proc(mx.array(toks), logits)
            self.assertTrue(mx.all(out == logits).item())

    def test_reasoning_budget_latch_survives_in_channel_open(self):
        # F1: an in-channel `think_open` (no close ever emitted) must NOT
        # release a tripped latch — only a real close does.
        from mlx_lm.sample_utils import make_reasoning_budget

        close, opn, vocab = 5, 4, 10
        budget = {"v": 100}
        proc = make_reasoning_budget(
            think_close=close,
            think_open=opn,
            max_think_tokens=lambda: budget["v"],
            check_every=10**9,
        )
        logits = mx.zeros((1, vocab))
        toks = [opn, 0, 1, 2]
        proc(mx.array(toks), logits)
        budget["v"] = 2  # trip
        toks.append(0)
        out = proc(mx.array(toks), logits)
        self.assertEqual(int(mx.argmax(out[0]).item()), close)
        # A stray in-channel re-open plus a grown budget: still latched.
        budget["v"] = 10**6
        toks.extend([opn, 1])
        out = proc(mx.array(toks), logits)
        self.assertEqual(int(mx.argmax(out[0]).item()), close)
        # A real close releases it; a fresh open re-arms a clean channel.
        toks.append(close)
        proc(mx.array(toks), logits)
        toks.extend([opn, 0, 1])
        out = proc(mx.array(toks), logits)
        self.assertTrue(mx.all(out == logits).item())

    def test_reasoning_budget_no_budget_eval_while_latched(self):
        # F3: once tripped, the budget callable is not invoked again — a cost
        # sensor that starts raising after the trip cannot crash the forced
        # close.
        from mlx_lm.sample_utils import make_reasoning_budget

        close, vocab = 5, 10
        sensor = {"down": False}

        def budget():
            if sensor["down"]:
                raise RuntimeError("cost sensor offline")
            return 3

        proc = make_reasoning_budget(
            think_close=close, max_think_tokens=budget, check_every=10**9
        )
        logits = mx.zeros((1, vocab))
        toks = [0, 1, 2]
        out = proc(mx.array(toks), logits)  # trip at the 3rd think token
        self.assertEqual(int(mx.argmax(out[0]).item()), close)
        sensor["down"] = True
        toks.append(1)
        out = proc(mx.array(toks), logits)  # latched: sensor never queried
        self.assertEqual(int(mx.argmax(out[0]).item()), close)

    def test_reasoning_budget_callable_return_validation(self):
        # F4: a dynamic budget must return a finite positive number, matching
        # the static constructor's positivity check; failures are labeled.
        from mlx_lm.sample_utils import make_reasoning_budget

        logits = mx.zeros((1, 10))
        for bad in (0, -5, float("inf"), float("nan")):
            proc = make_reasoning_budget(think_close=5, max_think_tokens=lambda: bad)
            with self.assertRaises(ValueError) as ctx:
                proc(mx.array([0]), logits)
            self.assertIn("max_think_tokens", str(ctx.exception))

    def test_cost_braked_budget_brake(self):
        from mlx_lm.sample_utils import make_cost_braked_budget

        # Default medium/medium: base 1280 * 1.25 = 1600, brake 0.35 at full
        # cost -> 1600 * 0.65 = 1040.
        cost = {"v": 0.0}
        b = make_cost_braked_budget(cost_fn=lambda: cost["v"])
        self.assertEqual(b(), 1600)
        cost["v"] = 1.0
        self.assertEqual(b(), 1040)
        cost["v"] = 0.5
        self.assertEqual(b(), 1320)
        # Out-of-range cost is clamped, not amplified.
        cost["v"] = 7.0
        self.assertEqual(b(), 1040)
        cost["v"] = -3.0
        self.assertEqual(b(), 1600)
        # N4: a NaN cost is an explicit, labeled error — never a silent brake.
        cost["v"] = float("nan")
        with self.assertRaises(ValueError) as ctx:
            b()
        self.assertIn("cost_fn", str(ctx.exception))

    def test_cost_braked_budget_clamps(self):
        from mlx_lm.sample_utils import make_cost_braked_budget

        # Floor: short/easy base 512 fully braked (cost_brake=1) -> 0 -> floor.
        b = make_cost_braked_budget(
            length_class="short",
            difficulty="easy",
            cost_brake=1.0,
            cost_fn=lambda: 1.0,
        )
        self.assertEqual(b(), 256)
        # Ceil: long/hard 3200 * 1.5 = 4800 caps at 4096 even with zero cost.
        b = make_cost_braked_budget(
            length_class="long", difficulty="hard", cost_fn=lambda: 0.0
        )
        self.assertEqual(b(), 4096)

    def test_cost_braked_budget_length_class(self):
        from mlx_lm.sample_utils import make_cost_braked_budget

        zero = lambda: 0.0
        # Derived from prompt_len (difficulty defaults to medium, 1.25x).
        self.assertEqual(make_cost_braked_budget(10, cost_fn=zero)(), 640)
        self.assertEqual(make_cost_braked_budget(1000, cost_fn=zero)(), 1600)
        self.assertEqual(make_cost_braked_budget(5000, cost_fn=zero)(), 4000)
        # Explicit length_class wins over prompt_len.
        self.assertEqual(
            make_cost_braked_budget(5000, length_class="short", cost_fn=zero)(), 640
        )

    def test_cost_braked_budget_wall_clock(self):
        # The default cost signal is wall-clock elapsed since first
        # evaluation, ramping to 1 over brake_horizon (injected clock).
        from mlx_lm.sample_utils import make_cost_braked_budget

        t = {"now": 100.0}
        b = make_cost_braked_budget(clock=lambda: t["now"], brake_horizon=60.0)
        self.assertEqual(b(), 1600)  # first call anchors the clock: cost 0
        t["now"] = 130.0  # half the horizon -> cost 0.5
        self.assertEqual(b(), 1320)
        t["now"] = 1000.0  # far past the horizon -> cost clamps at 1
        self.assertEqual(b(), 1040)

    def test_cost_braked_budget_validation(self):
        from mlx_lm.sample_utils import make_cost_braked_budget

        with self.assertRaises(ValueError):
            make_cost_braked_budget(floor=0)
        with self.assertRaises(ValueError):
            make_cost_braked_budget(floor=512, ceil=256)
        with self.assertRaises(ValueError):
            make_cost_braked_budget(cost_brake=1.5)
        with self.assertRaises(ValueError):
            make_cost_braked_budget(difficulty="impossible")
        with self.assertRaises(ValueError):
            make_cost_braked_budget(length_class="epic")
        with self.assertRaises(ValueError):
            make_cost_braked_budget(brake_horizon=0.0)
        with self.assertRaises(ValueError):
            make_cost_braked_budget(short_prompt=2048, long_prompt=256)
        with self.assertRaises(ValueError):
            make_cost_braked_budget(prompt_len=-1)

    def test_reasoning_budget_with_cost_braked_budget(self):
        # End to end: a cost-braked budget drives the processor, and a cost
        # spike mid-generation pulls the close in.
        from mlx_lm.sample_utils import make_cost_braked_budget, make_reasoning_budget

        close, vocab = 5, 10
        cost = {"v": 0.0}
        budget = make_cost_braked_budget(
            length_class="short",
            difficulty="easy",  # base 512
            floor=1,
            cost_fn=lambda: cost["v"],
        )
        proc = make_reasoning_budget(
            think_close=close, max_think_tokens=budget, check_every=10**9
        )
        logits = mx.zeros((1, vocab))
        toks = []
        for i in range(332):  # under both 512 (cost 0) and 332 (full cost)
            toks.append(i % 4)
            out = proc(mx.array(toks), logits)
            self.assertTrue(mx.all(out == logits).item())
        cost["v"] = 1.0  # host under full load: 512 * 0.65 = 332
        toks.append(0)  # 333rd think token >= braked budget -> forced close
        out = proc(mx.array(toks), logits)
        self.assertEqual(int(mx.argmax(out[0]).item()), close)

    def test_make_logits_processors(self):
        from mlx_lm.sample_utils import make_logits_processors

        # Create processors with all three penalty types
        tokens = mx.array([0, 0, 0, 1, 1])
        # Use non-zero logits so repetition penalty has effect
        logits = mx.array([[1.0, 0.5, 0.0, -0.5]])
        processors = make_logits_processors(
            repetition_penalty=1.5,
            repetition_context_size=5,
            presence_penalty=0.5,
            presence_context_size=5,
            frequency_penalty=0.25,
            frequency_context_size=5,
        )
        # Apply all processors
        for processor in processors:
            logits = processor(tokens, logits)
        # Token 0 (appears 3x): 1.0/1.5 - 0.5 - 0.75 = -0.5833
        # Token 1 (appears 2x): 0.5/1.5 - 0.5 - 0.5 = -0.6667
        # Token 2 (not in context): 0.0 (no penalty)
        # Token 3 (not in context): -0.5 (no penalty)
        self.assertAlmostEqual(logits[0, 0].item(), -0.5833, places=4)
        self.assertAlmostEqual(logits[0, 1].item(), -0.6667, places=4)
        self.assertAlmostEqual(logits[0, 2].item(), 0.0, places=4)
        self.assertAlmostEqual(logits[0, 3].item(), -0.5, places=4)


if __name__ == "__main__":
    unittest.main()
