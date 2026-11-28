# Copyright © 2026 Apple Inc.

from typing import Any, Optional

from mlx import core as mx
from mlx import nn

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.rope import initialize_rope
from mlx_lm.models.gptj.config import ModelArgs

__all__ = [
    "GPTJAttention",
    "GPTJBlock",
    "GPTJMLP",
]


class GPTJAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.max_position = args.n_positions
        self.embed_dim = args.n_embd
        self.num_attention_heads = args.n_head

        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )

        self.scale_attn = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.rotary_dim = args.rotary_dim or self.head_dim  # avoid rotary_dim = None
        self.rope = initialize_rope(
            dims=self.rotary_dim,
            base=10000,
            traditional=True,
            max_position_embeddings=self.max_position,
        )

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:

        B, L, D = inputs.shape

        query = self.q_proj(inputs)
        key = self.k_proj(inputs)
        value = self.v_proj(inputs)

        query = query.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
        key = key.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
        value = value.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)

        k_rot = key[:, :, :, : self.rotary_dim]
        k_pass = key[:, :, :, self.rotary_dim :]

        q_rot = query[:, :, :, : self.rotary_dim]
        q_pass = query[:, :, :, self.rotary_dim :]
        if cache is not None:
            k_rot = self.rope(k_rot, offset=cache.offset)
            q_rot = self.rope(q_rot, offset=cache.offset)
        else:
            k_rot = self.rope(k_rot)
            q_rot = self.rope(q_rot)
        query = mx.concatenate([q_rot, q_pass], axis=-1)
        key = mx.concatenate([k_rot, k_pass], axis=-1)
        if cache is not None:
            key, value = cache.update_and_fetch(key, value)

        output = scaled_dot_product_attention(
            query, key, value, cache=cache, scale=self.scale_attn, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class GPTJMLP(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        embed_dim = args.n_embd
        intermediate_size = 4 * embed_dim
        self.fc_in = nn.Linear(embed_dim, intermediate_size, bias=True)
        self.fc_out = nn.Linear(intermediate_size, embed_dim, bias=True)

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:

        hidden_states = self.fc_in(inputs)
        hidden_states = nn.gelu_approx(hidden_states)  # gelu_new
        hidden_states = self.fc_out(hidden_states)
        return hidden_states


class GPTJBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.ln_1 = nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon)
        self.attn = GPTJAttention(args=args)
        self.mlp = GPTJMLP(args=args)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:

        residual = inputs
        hidden_states = self.ln_1(inputs)
        attn_output = self.attn(inputs=hidden_states, mask=mask, cache=cache)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        return hidden_states
