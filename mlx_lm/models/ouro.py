# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "ouro"
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    intermediate_size: int = 5632
    vocab_size: int = 49152
    head_dim: int = 128
    max_position_embeddings: int = 65536
    rope_theta: float = 1000000.0
    rope_scaling: Optional[dict] = None
    rms_norm_eps: float = 1e-6
    total_ut_steps: int = 4
    early_exit_step: Optional[int] = None
    early_exit_threshold: Optional[float] = None
    tie_word_embeddings: bool = False


@partial(mx.compile, shapeless=True)
def swiglu(gate: mx.array, up: mx.array) -> mx.array:
    return nn.silu(gate) * up


class OuroAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()

        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(
            args.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        keys = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim)
        values = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class OuroMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class OuroDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = OuroAttention(args, layer_idx)
        self.mlp = OuroMLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.input_layernorm_2 = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_attention_layernorm_2 = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, mask, cache)
        h = self.input_layernorm_2(h)
        h = residual + h

        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        h = self.post_attention_layernorm_2(h)
        return residual + h


class OuroModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.total_ut_steps = args.total_ut_steps
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            OuroDecoderLayer(args, layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.early_exit_gate = nn.Linear(args.hidden_size, 1, bias=True)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[List[KVCache]] = None,
    ) -> Tuple[mx.array, Optional[List[mx.array]], Optional[List[mx.array]]]:
        h = self.embed_tokens(x)

        if cache is None:
            cache = [None] * (self.total_ut_steps * len(self.layers))

        mask = create_attention_mask(h, cache[0])
        hidden_states_list = []
        gate_list = []

        for current_ut in range(self.total_ut_steps):
            for layer_idx, layer in enumerate(self.layers):
                cache_idx = current_ut * len(self.layers) + layer_idx
                layer_cache = cache[cache_idx]
                h = layer(h, mask, layer_cache)

            h = self.norm(h)
            hidden_states_list.append(h)
            gate = self.early_exit_gate(h)
            gate_list.append(gate)

        return h, hidden_states_list, gate_list


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = OuroModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[KVCache]] = None,
        use_weighted_exit: bool = False,
        exit_at_step: Optional[int] = None,
        exit_threshold: Optional[float] = None,
        logits_to_keep: int = 0,
    ) -> mx.array:
        hidden_states, hidden_states_list, gate_list = self.model(inputs, cache)
        exit_pdf = self._compute_exit_distribution(gate_list)

        exit_at_step = (
            exit_at_step if exit_at_step is not None else self.args.early_exit_step
        )
        exit_threshold = (
            exit_threshold
            if exit_threshold is not None
            else self.args.early_exit_threshold
        )

        return self._compute_logits(
            hidden_states,
            hidden_states_list,
            exit_pdf,
            use_weighted_exit,
            exit_at_step,
            exit_threshold,
            logits_to_keep,
        )

    def _compute_exit_distribution(self, gate_list: List[mx.array]) -> mx.array:
        pdf_list = []
        remaining_prob = mx.ones_like(gate_list[0].squeeze(-1))

        for gate_tensor in gate_list[:-1]:
            lambda_i = mx.sigmoid(gate_tensor.squeeze(-1))
            p_i = lambda_i * remaining_prob
            remaining_prob = remaining_prob * (1.0 - lambda_i)
            pdf_list.append(p_i)

        pdf_list.append(remaining_prob)
        return mx.stack(pdf_list, axis=-1)

    def _select_token_positions(
        self, tensor: mx.array, logits_to_keep: int
    ) -> mx.array:
        return tensor if logits_to_keep == 0 else tensor[:, -logits_to_keep:]

    def _hidden_to_logits(
        self, hidden_states: mx.array, logits_to_keep: int
    ) -> mx.array:
        return self._apply_lm_head(
            self._select_token_positions(hidden_states, logits_to_keep)
        )

    def _compute_logits(
        self,
        final_hidden_states: mx.array,
        hidden_states_list: Optional[List[mx.array]],
        exit_pdf: Optional[mx.array],
        use_weighted_exit: bool,
        exit_at_step: Optional[int],
        exit_threshold: Optional[float],
        logits_to_keep: int,
    ) -> mx.array:
        if not hidden_states_list or exit_pdf is None:
            return self._hidden_to_logits(final_hidden_states, logits_to_keep)

        if exit_at_step is not None and 0 <= exit_at_step < len(hidden_states_list):
            return self._hidden_to_logits(
                hidden_states_list[exit_at_step], logits_to_keep
            )

        if exit_threshold is not None:
            return self._compute_threshold_exit_logits(
                hidden_states_list, exit_pdf, exit_threshold, logits_to_keep
            )

        if use_weighted_exit:
            return self._compute_weighted_exit_logits(
                hidden_states_list, exit_pdf, logits_to_keep
            )

        return self._hidden_to_logits(final_hidden_states, logits_to_keep)

    def _apply_lm_head(self, hidden_states: mx.array) -> mx.array:
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden_states)
        return self.lm_head(hidden_states)

    def _compute_weighted_exit_logits(
        self,
        hidden_states_list: List[mx.array],
        exit_pdf: mx.array,
        logits_to_keep: int,
    ) -> mx.array:
        token_exit_pdf = self._select_token_positions(exit_pdf, logits_to_keep)
        expected_logits = None

        for step_idx, hidden in enumerate(hidden_states_list):
            step_logits = self._hidden_to_logits(hidden, logits_to_keep)
            weight = mx.expand_dims(token_exit_pdf[..., step_idx], axis=-1)
            weighted = step_logits * weight

            if expected_logits is None:
                expected_logits = weighted
            else:
                expected_logits = expected_logits + weighted

        return expected_logits

    def _compute_threshold_exit_logits(
        self,
        hidden_states_list: List[mx.array],
        exit_pdf: mx.array,
        exit_threshold: float,
        logits_to_keep: int,
    ) -> mx.array:
        cumulative_probs = mx.cumsum(exit_pdf, axis=-1)
        threshold_mask = cumulative_probs >= exit_threshold

        exit_steps = mx.argmax(threshold_mask, axis=-1)
        exit_steps = mx.where(
            mx.any(threshold_mask, axis=-1), exit_steps, len(hidden_states_list) - 1
        )

        stacked_hidden = mx.stack(hidden_states_list, axis=2)
        batch_indices = mx.arange(exit_steps.shape[0])[:, None]
        seq_indices = mx.arange(exit_steps.shape[1])[None, :]
        final_hidden_states = stacked_hidden[batch_indices, seq_indices, exit_steps, :]

        return self._hidden_to_logits(final_hidden_states, logits_to_keep)

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if "early_exit_gate" in path:
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    def make_cache(self):
        total_caches = self.args.total_ut_steps * len(self.layers)
        return [KVCache() for _ in range(total_caches)]

    def sanitize(self, weights):
        return weights
