# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.models import dflash_laguna, laguna


class TestDFlashLaguna(unittest.TestCase):
    def _target(self, hidden_size=2048, vocab_size=64, n_layers=2):
        args = laguna.ModelArgs(
            model_type="laguna",
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            num_hidden_layers=n_layers,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=hidden_size // 16,
            max_position_embeddings=512,
        )
        return laguna.Model(args)

    def test_draft_block_forward(self):
        block_size = 4
        n_aux = 2
        # dflash's projections are hard-coded to hidden_size 2048.
        hidden = 2048
        vocab = 64

        target = self._target(hidden_size=hidden, vocab_size=vocab)

        d_args = dflash_laguna.ModelArgs(
            hidden_size=hidden,
            num_hidden_layers=1,
            num_aux_hidden_states=n_aux,
            block_size=block_size,
        )
        drafter = dflash_laguna.Model(d_args)

        # A speculator has no LM of its own: calling it as a causal LM must error.
        with self.assertRaises(RuntimeError):
            drafter(mx.array([[1, 2, 3]]))

        # Build the mask block [anchor, MASK*(block-1)] and embed it via the
        # TARGET's embedding — the coupling this PR stacks on #1223 for.
        anchor = 7
        block_ids = mx.array([[anchor] + [d_args.mask_token_id] * (block_size - 1)])
        target_embed = target.model.embed_tokens(block_ids)
        self.assertEqual(target_embed.shape, (1, block_size, hidden))

        # Aux hidden states injected as extra K/V context. In production these
        # come from selected target layers; here we exercise the fuse + draft
        # path with correctly-shaped context.
        ctx_len = block_size
        aux = [mx.random.normal((1, ctx_len, hidden)) for _ in range(n_aux)]
        target_hidden = drafter.fuse(aux)
        self.assertEqual(target_hidden.shape, (1, ctx_len, hidden))

        # Run the block-draft forward through the draft layers.
        block_hidden = drafter.draft_block(target_embed, target_hidden, anchor_pos=0)
        self.assertEqual(block_hidden.shape, (1, block_size, hidden))
        self.assertFalse(mx.any(mx.isnan(block_hidden)).item())

        # Draft logits come from the TARGET's lm_head (reused, not its own).
        logits = target.lm_head(block_hidden)
        self.assertEqual(logits.shape, (1, block_size, vocab))
        self.assertFalse(mx.any(mx.isnan(logits)).item())


if __name__ == "__main__":
    unittest.main()
