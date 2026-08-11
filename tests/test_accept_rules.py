# Copyright © 2026 Apple Inc.

"""Tests for the speculative decoding ``accept_rule`` options.

Kernel tests drive the acceptance helpers directly with constructed
distributions, where any bias in the committed-token distribution is
unmistakable: the residual (rejection sampling) rule must commit tokens
distributed exactly as the target while accepting drafts the sampled-token
exact-match rule rejects, and block verification must preserve the target
distribution while accepting at least as many tokens as the per-token rule.

End-to-end tests use tiny random models: the default rule must be
bit-identical to ``accept_rule="exact"``, greedy decoding must be identical
across all rules, and the temperature > 0 committed marginals of every rule
must match plain non-speculative sampling. Every trial is explicitly seeded
so all statistics are deterministic; the bounds guard against math
regressions, not sampling noise. Both the target and draft distributions of
the tiny random models are near uniform, so the end-to-end chi-square is a
consistency check — the kernel tests carry the exactness burden.
"""

import unittest
from collections import Counter

import mlx.core as mx

from mlx_lm.generate import (
    _acceptance_probs,
    _block_verify,
    _residual_sample,
    _sampling_logprobs,
    generate_step,
    speculative_generate_step,
)
from mlx_lm.models import llama
from mlx_lm.sample_utils import make_sampler


def _lp(probs):
    return mx.log(mx.array(probs, dtype=mx.float32))


def _tv(counts, ref):
    """Total variation between an empirical Counter and a reference pmf."""
    total = sum(counts.values())
    return 0.5 * sum(abs(counts.get(i, 0) / total - ref[i]) for i in range(len(ref)))


# Constructed distributions: TV(P1, Q1) = 0.5, per-token acceptance
# sum(min(p, q)) = 0.5 at position 1 and 0.55 at position 2.
P1 = [0.5, 0.25, 0.125, 0.125]
Q1 = [0.125, 0.125, 0.25, 0.5]
P2 = [0.25, 0.25, 0.25, 0.25]
Q2 = [0.7, 0.1, 0.1, 0.1]
P3 = [0.5, 0.25, 0.125, 0.125]  # bonus row; any pmf works


class TestAcceptRuleKernels(unittest.TestCase):
    """Acceptance-rule helpers on constructed distributions — no model."""

    def test_acceptance_probs_exact_values(self):
        probs = _acceptance_probs(_lp([P1]), _lp([Q1]), [0])
        self.assertAlmostEqual(probs[0].item(), 1.0, places=6)  # p > q
        probs = _acceptance_probs(_lp([P1]), _lp([Q1]), [3])
        self.assertAlmostEqual(probs[0].item(), 0.25, places=5)  # p / q
        # p == q: the ratio is exactly 1 for every token, so the residual
        # rule accepts every draft, while exact matching of two independent
        # samples only accepts with probability sum(p^2).
        probs = _acceptance_probs(_lp([P1] * 4), _lp([P1] * 4), [0, 1, 2, 3])
        self.assertEqual(probs.tolist(), [1.0] * 4)

    def test_exact_match_accepts_sum_p_squared_when_p_equals_q(self):
        lp = _lp(P1)
        n = 2000
        hits = 0
        for i in range(n):
            mx.random.seed(50_000 + i)
            d = mx.random.categorical(lp).item()
            t = mx.random.categorical(lp).item()
            hits += d == t
        # sum(p^2) = 0.34375 for P1; residual accepts 1.0 (previous test)
        self.assertLess(abs(hits / n - 0.34375), 0.04)

    def test_subfloor_probabilities_accept_correctly(self):
        # min(1, p/q) must be computed in log space: q=1e-35 and p=1e-34 are
        # representable float32 probabilities and p/q = 10 means CERTAIN
        # acceptance. A linear-space clamp such as max(q, 1e-30) would give
        # p/floor = 1e-4 instead and reject nearly always, biasing the
        # committed marginal.
        tlp = mx.log(mx.array([0.4, 0.6, 1e-34, 1e-35], dtype=mx.float32))
        dlp = mx.log(mx.array([0.5, 0.5, 1e-35, 1e-34], dtype=mx.float32))
        probs = _acceptance_probs(tlp[None], dlp[None], [2])
        self.assertEqual(probs[0].item(), 1.0)
        # The sub-floor ratio must also be honored below 1: token 3 has
        # p/q = 0.1 (clamped math would give ~1e-5, i.e. never accept).
        probs = _acceptance_probs(tlp[None], dlp[None], [3])
        self.assertAlmostEqual(probs[0].item(), 0.1, places=3)

    def test_residual_rule_commits_target_distribution(self):
        # Draft from q, accept with probability min(1, p/q), resample
        # rejections from relu(p - q): the committed token must be
        # distributed as p and the acceptance rate must sit at
        # sum(min(p, q)) = 0.5.
        lp1, lq1 = _lp(P1), _lp(Q1)
        n = 4000
        counts = Counter()
        accepted = 0
        for i in range(n):
            mx.random.seed(10_000 + i)
            d = mx.random.categorical(lq1).item()
            prob = _acceptance_probs(lp1[None], lq1[None], [d])[0].item()
            if mx.random.uniform().item() <= prob:
                counts[d] += 1
                accepted += 1
            else:
                counts[_residual_sample(lp1, lq1)] += 1
        self.assertLess(_tv(counts, P1), 0.05)
        self.assertGreater(_tv(counts, Q1), 0.4)  # power: not the draft dist
        self.assertLess(abs(accepted / n - 0.5), 0.05)

    def test_block_verify_preserves_target_and_beats_per_token(self):
        # k=2 block: the first committed token must still be ~ p1, and the
        # expected accepted length must be at least the per-token rule's
        # (arXiv 2403.10444, Theorem 1; the cumulative-ratio rescue is worth
        # ~+0.12 tokens per block on these distributions).
        lp1, lq1, lq2 = _lp(P1), _lp(Q1), _lp(Q2)
        target = mx.stack([lp1, _lp(P2), _lp(P3)])
        drafts_lp = mx.stack([lq1, lq2])
        n = 2500
        counts = Counter()
        block_total = token_total = 0
        for i in range(n):
            mx.random.seed(200_000 + i)
            d1 = mx.random.categorical(lq1).item()
            d2 = mx.random.categorical(lq2).item()
            n_accept, correction = _block_verify(target, drafts_lp, [d1, d2])
            counts[d1 if n_accept >= 1 else correction] += 1
            block_total += n_accept
            # Per-token residual rule on the SAME drafts, its own coins.
            mx.random.seed(700_000 + i)
            probs = _acceptance_probs(target[:2], drafts_lp, [d1, d2])
            us = mx.random.uniform(shape=(2,))
            if us[0].item() <= probs[0].item():
                token_total += 1
                if us[1].item() <= probs[1].item():
                    token_total += 1
        self.assertLess(_tv(counts, P1), 0.06)
        self.assertGreater(_tv(counts, Q1), 0.35)
        self.assertGreaterEqual(block_total / n, token_total / n + 0.05)

    def test_block_verify_k1_matches_residual(self):
        # Block size 1 degenerates to per-token rejection sampling:
        # acceptance min(1, p/q) and residual relu(p - q), so the committed
        # distribution is p and the acceptance rate is sum(min(p, q)) = 0.5.
        lp1, lq1 = _lp(P1), _lp(Q1)
        target = mx.stack([lp1, _lp(P2)])
        n = 2000
        counts = Counter()
        accepted = 0
        for i in range(n):
            mx.random.seed(400_000 + i)
            d1 = mx.random.categorical(lq1).item()
            n_accept, correction = _block_verify(target, lq1[None], [d1])
            counts[d1 if n_accept >= 1 else correction] += 1
            accepted += n_accept
        self.assertLess(_tv(counts, P1), 0.06)
        self.assertLess(abs(accepted / n - 0.5), 0.06)

    def test_sampling_logprobs_temperature(self):
        lp = _lp(P1)
        self.assertTrue(mx.allclose(_sampling_logprobs(lp, 1.0), lp))
        scaled = _sampling_logprobs(lp, 0.5)
        expected = 2.0 * lp - mx.logsumexp(2.0 * lp)
        self.assertTrue(mx.allclose(scaled, expected))


def _tiny_model(seed):
    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        rms_norm_eps=1e-5,
        vocab_size=64,
        tie_word_embeddings=True,
    )
    mx.random.seed(seed)
    model = llama.Model(args)
    mx.eval(model.parameters())
    return model


class TestAcceptRuleEndToEnd(unittest.TestCase):
    """accept_rule through speculative_generate_step on tiny models."""

    @classmethod
    def setUpClass(cls):
        cls.model = _tiny_model(0)
        cls.draft_model = _tiny_model(1)
        mx.random.seed(2)
        cls.prompt = mx.random.randint(0, 64, (16,))

    def _spec(self, seed, temp, max_tokens=12, **kwargs):
        mx.random.seed(seed)
        sampler = make_sampler(temp)
        return [
            (token, from_draft)
            for token, _, from_draft in speculative_generate_step(
                self.prompt,
                self.model,
                self.draft_model,
                num_draft_tokens=2,
                max_tokens=max_tokens,
                sampler=sampler,
                **kwargs,
            )
        ]

    def test_default_is_exact(self):
        # The default must be bit-identical to accept_rule="exact" for every
        # temperature and seed: same tokens AND same draft/target labels.
        for temp in (0.0, 0.7, 1.0):
            for seed in range(3):
                default = self._spec(seed, temp)
                explicit = self._spec(seed, temp, accept_rule="exact")
                self.assertEqual(default, explicit)

    def test_greedy_identical_across_rules(self):
        # At temperature 0 every rule reduces to exact matching, so the
        # streams must be identical.
        base = self._spec(5, 0.0)
        for rule in ("exact", "residual", "block"):
            self.assertEqual(self._spec(5, 0.0, accept_rule=rule), base)

    def test_invalid_rule_raises(self):
        gen = speculative_generate_step(
            self.prompt,
            self.model,
            self.draft_model,
            max_tokens=4,
            accept_rule="bogus",
        )
        with self.assertRaises(ValueError):
            next(gen)

    def test_unsupported_sampler_raises(self):
        # residual/block need the sampled distribution, which cannot be
        # recovered from a filtering or custom sampler.
        for sampler in (make_sampler(0.8, top_p=0.9), lambda x: mx.argmax(x, -1)):
            for rule in ("residual", "block"):
                gen = speculative_generate_step(
                    self.prompt,
                    self.model,
                    self.draft_model,
                    max_tokens=4,
                    sampler=sampler,
                    accept_rule=rule,
                )
                with self.assertRaises(ValueError):
                    next(gen)

    @staticmethod
    def _binned_chi2(a, b, n_top=8):
        # Two-sample chi-square over the reference arm's top-n_top tokens
        # plus a pooled tail: sum (a_i - b_i)^2 / (a_i + b_i), df = n_top.
        top = [t for t, _ in a.most_common(n_top)]

        def cnt(c, tok):
            if tok == "other":
                return sum(v for k, v in c.items() if k not in top)
            return c.get(tok, 0)

        stat = 0.0
        for tok in top + ["other"]:
            x, y = cnt(a, tok), cnt(b, tok)
            if x + y > 0:
                stat += (x - y) ** 2 / (x + y)
        return stat

    def test_temp_marginals_match_plain_sampling(self):
        # Per-position marginals over N seeded trials for each rule vs a
        # plain non-speculative reference. Threshold 32 ~ chi2(df=8) 0.9999
        # quantile; the observed values with these seeds sit well below it.
        temp = 0.8
        new_tokens = 4
        n = 250
        base_seed = {
            "plain": 1000,
            "exact": 20_000,
            "residual": 40_000,
            "block": 60_000,
        }
        arms = {}
        accept_per_block = {}
        for name in ("plain", "exact", "residual", "block"):
            counts = [Counter() for _ in range(new_tokens)]
            n_drafted = n_blocks = 0
            for i in range(n):
                seed = base_seed[name] + i
                if name == "plain":
                    mx.random.seed(seed)
                    toks = [
                        token
                        for token, _ in generate_step(
                            self.prompt,
                            self.model,
                            max_tokens=new_tokens,
                            sampler=make_sampler(temp),
                        )
                    ]
                else:
                    stream = self._spec(
                        seed, temp, max_tokens=new_tokens, accept_rule=name
                    )
                    toks = [t for t, _ in stream]
                    n_drafted += sum(fd for _, fd in stream)
                    n_blocks += sum(not fd for _, fd in stream)
                for j, t in enumerate(toks):
                    counts[j][t] += 1
            arms[name] = counts
            accept_per_block[name] = n_drafted / max(n_blocks, 1)
        for name in ("exact", "residual", "block"):
            for j in range(new_tokens):
                stat = self._binned_chi2(arms["plain"][j], arms[name][j])
                self.assertLess(stat, 32.0, f"arm={name} position={j} chi2={stat:.1f}")
        # The point of the new rules: at temperature > 0 the residual rule
        # accepts a large fraction of the drafts the sampled-token
        # exact-match rule throws away, and block verification accepts at
        # least as many.
        self.assertGreater(
            accept_per_block["residual"], accept_per_block["exact"] + 0.3
        )
        self.assertGreaterEqual(
            accept_per_block["block"], accept_per_block["residual"] - 0.15
        )


if __name__ == "__main__":
    unittest.main()
