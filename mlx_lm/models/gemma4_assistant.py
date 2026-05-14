# Copyright © 2026 Apple Inc.
"""
Gemma 4 MTP drafter (assistant) model.

This is the speculative-decoding companion released alongside Gemma 4. The
drafter has NO key/value projections of its own — at each layer it cross-
attends to the target model's K/V via `shared_kv_states`. See the spec at
core/life/docs/superpowers/specs/2026-05-13-mlx-lm-gemma4-assistant.md
(broomva workspace) or the HF reference at
transformers/models/gemma4_assistant/modeling_gemma4_assistant.py.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, scaled_dot_product_attention
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gemma4_assistant"
    backbone_hidden_size: int = 2560
    num_centroids: int = 2048
    centroid_intermediate_top_k: int = 32
    use_ordered_embeddings: bool = True
    tie_word_embeddings: bool = True
    text_config: Dict[str, Any] = field(default_factory=dict)
    vocab_size: int = 262144  # echoed at top level for convenience

    def __post_init__(self):
        # Mirror gemma4.ModelArgs: vocab_size flows down into text_config
        if "vocab_size" not in self.text_config:
            self.text_config["vocab_size"] = self.vocab_size
        # Defaults for fields the HF config sometimes elides
        self.text_config.setdefault("num_attention_heads", 4)
        self.text_config.setdefault("num_key_value_heads", 2)
        self.text_config.setdefault("head_dim", 256)


class AssistantAttention(nn.Module):
    """Q-only cross-attention. K/V are supplied by the target model.

    Per-layer wiring follows the assistant safetensors:
      - self_attn.q_proj.weight       : (n_heads * head_dim, hidden_size)
      - self_attn.q_norm.weight       : (head_dim,)
      - self_attn.o_proj.weight       : (hidden_size, n_heads * head_dim)
    The drafter has no k_proj/v_proj weights — those come from the target.
    """

    def __init__(self, config: ModelArgs, layer_type: str):
        super().__init__()
        tc = config.text_config
        self.layer_type = layer_type
        self.hidden_size = tc["hidden_size"]
        self.n_heads = tc["num_attention_heads"]
        self.n_kv_heads = tc["num_key_value_heads"]
        self.head_dim = tc["head_dim"]
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=tc.get("rms_norm_eps", 1e-6))
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, self.hidden_size, bias=False
        )

        # Matching RoPE to the target so Q and target's pre-RoPE'd K stay in
        # the same rotational frame.
        rope_params = tc.get("rope_parameters", {}).get(layer_type, {})
        self.rope = initialize_rope(
            dims=self.head_dim,
            traditional=False,
            base=rope_params.get("rope_theta", 10000.0),
            scaling_config=rope_params,
            max_position_embeddings=tc.get("max_position_embeddings", 131072),
        )

    def __call__(
        self,
        x: mx.array,                      # (B, L, hidden_size)
        keys: mx.array,                   # (B, n_kv_heads, L_target, head_dim)
        values: mx.array,                 # (B, n_kv_heads, L_target, head_dim)
        position_ids: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        # Project Q and apply head-dim RMSNorm.
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        q = self.q_norm(q)
        q = q.transpose(0, 2, 1, 3)  # (B, n_heads, L, head_dim)

        # Apply RoPE to Q. Target has already RoPE'd K with the same params.
        # position_ids is typically a constant (single-position MTP); broadcast
        # the offset across L.
        offset = 0
        if position_ids is not None:
            # Use the scalar offset of the last seen token.
            offset = int(position_ids.reshape(-1)[-1].item())
        q = self.rope(q, offset=offset)

        # GQA: repeat K/V from n_kv_heads to n_heads along head axis.
        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            keys = mx.repeat(keys, n_rep, axis=1)
            values = mx.repeat(values, n_rep, axis=1)

        attn_out = scaled_dot_product_attention(
            q, keys, values, cache=None, scale=self.scale, mask=mask
        )
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(attn_out)


def _swiglu(gate: mx.array, x: mx.array) -> mx.array:
    return nn.silu(gate) * x


class AssistantMLP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        tc = config.text_config
        hidden = tc["hidden_size"]
        inter = tc["intermediate_size"]
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # Gemma uses gelu-tanh in gemma4_text, but the assistant config
        # specifies hidden_activation=gelu_pytorch_tanh too. Use gelu_approx
        # (mlx's gelu-tanh approximation) for parity.
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class AssistantDecoderLayer(nn.Module):
    """Gemma-family double-norm decoder block, Q-only attention."""

    def __init__(self, config: ModelArgs, layer_type: str):
        super().__init__()
        tc = config.text_config
        hidden = tc["hidden_size"]
        eps = tc.get("rms_norm_eps", 1e-6)
        self.layer_type = layer_type
        self.self_attn = AssistantAttention(config, layer_type)
        self.mlp = AssistantMLP(config)
        self.input_layernorm = nn.RMSNorm(hidden, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden, eps=eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(hidden, eps=eps)
        self.post_feedforward_layernorm = nn.RMSNorm(hidden, eps=eps)
        # Per-layer scalar (11th weight per layer in the checkpoint).
        self.layer_scalar = mx.ones((1,))

    def __call__(
        self,
        x: mx.array,
        keys: mx.array,
        values: mx.array,
        position_ids: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        # Attention sublayer: pre-norm input, post-norm sublayer output, residual.
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, keys, values, position_ids=position_ids, mask=mask)
        h = self.post_attention_layernorm(h)
        h = residual + h

        # MLP sublayer: same pattern.
        residual = h
        h = self.pre_feedforward_layernorm(h)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        h = residual + h

        return h * self.layer_scalar


class AssistantTextModel(nn.Module):
    """4-layer Q-only decoder stack with shared embedding table."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        tc = args.text_config
        self.args = args
        self.vocab_size = tc["vocab_size"]
        self.num_hidden_layers = tc["num_hidden_layers"]
        layer_types = tc.get("layer_types") or [
            "full_attention"
        ] * self.num_hidden_layers
        assert len(layer_types) == self.num_hidden_layers, (
            f"layer_types length {len(layer_types)} != "
            f"num_hidden_layers {self.num_hidden_layers}"
        )

        self.embed_tokens = nn.Embedding(self.vocab_size, tc["hidden_size"])
        self.layers = [
            AssistantDecoderLayer(args, layer_type=lt)
            for lt in layer_types
        ]
        self.norm = nn.RMSNorm(tc["hidden_size"], eps=tc.get("rms_norm_eps", 1e-6))

    def __call__(
        self,
        inputs_embeds: mx.array,
        shared_kv_states: Dict[str, Tuple[mx.array, mx.array]],
        position_ids: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        h = inputs_embeds
        for layer in self.layers:
            k, v = shared_kv_states[layer.layer_type]
            h = layer(h, k, v, position_ids=position_ids, mask=mask)
        return self.norm(h)


class MaskedEmbedder(nn.Module):
    """Centroid-clustered logit head.

    Vocab is split into `num_centroids` clusters of `vocab_size_per_centroid`
    tokens each. For each (batch, position) we:
      1. Score all centroids via `self.centroids` (a tiny Linear).
      2. Pick the top-K clusters.
      3. Compute logits only for the tokens in those clusters via a gather
         + matmul against `lm_head_weight` (which is tied with embed_tokens).
      4. Scatter those logits back into a (V,) tensor, with non-selected
         positions filled with `min(selected_logits) - 1.0` so they never win.

    The `token_ordering` buffer maps cluster-ordered positions back to the
    canonical token id space.
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        text_config = config.text_config
        self.hidden_size = text_config["hidden_size"]
        self.vocab_size = text_config["vocab_size"]
        self.num_centroids = config.num_centroids
        self.top_k = config.centroid_intermediate_top_k
        assert self.vocab_size % self.num_centroids == 0, (
            f"vocab_size {self.vocab_size} not divisible by "
            f"num_centroids {self.num_centroids}"
        )
        self.vocab_size_per_centroid = self.vocab_size // self.num_centroids

        self.centroids = nn.Linear(self.hidden_size, self.num_centroids, bias=False)
        # token_ordering is a buffer (not trained). Stored as int32 for MLX gather.
        self.token_ordering = mx.zeros((self.vocab_size,), dtype=mx.int32)

    def __call__(self, hidden_states: mx.array, lm_head_weight: mx.array) -> mx.array:
        B, L = hidden_states.shape[:2]
        V = self.vocab_size
        V_pc = self.vocab_size_per_centroid

        # 1. Score centroids: (B, L, num_centroids)
        centroid_logits = self.centroids(hidden_states)

        # 2. Top-K clusters by score: (B, L, top_k)
        # argpartition gives unsorted top-k; that's fine here.
        top_k_indices = mx.argpartition(
            -centroid_logits, kth=self.top_k - 1, axis=-1
        )[..., : self.top_k]

        # 3. canonical_positions_per_cluster: (num_centroids, V_pc)
        canonical = self.token_ordering.reshape(self.num_centroids, V_pc)

        # 4. Gather the V_pc canonical token ids for each of the top_k clusters:
        # selected_canonical: (B, L, top_k, V_pc)
        selected_canonical = canonical[top_k_indices]

        # 5. Gather rows of lm_head_weight at those canonical positions.
        # lm_head_weight: (V, H). Flat index → reshape.
        flat = selected_canonical.reshape(-1)  # (B*L*top_k*V_pc,)
        selected_emb = lm_head_weight[flat].reshape(
            B, L, self.top_k * V_pc, self.hidden_size
        )

        # 6. Dot product: (B, L, 1, H) @ (B, L, H, top_k*V_pc) → (B, L, top_k*V_pc)
        h_exp = mx.expand_dims(hidden_states, -2)               # (B, L, 1, H)
        selected_logits = (h_exp @ selected_emb.swapaxes(-1, -2)).squeeze(-2)

        # 7. Scatter into full-vocab output, with floor-1 mask for non-selected.
        mask_value = mx.min(selected_logits).item() - 1.0
        output = mx.full(
            (B, L, V), mask_value, dtype=hidden_states.dtype
        )
        scatter_idx = selected_canonical.reshape(B, L, -1)      # (B, L, top_k*V_pc)
        return mx.put_along_axis(output, scatter_idx, selected_logits, axis=-1)


class Model(nn.Module):
    """Top-level Gemma 4 MTP drafter.

    Forward signature:
      inputs_embeds:        (B, L, 2 * backbone_hidden_size)
                            = concat(target_embed(last_token), target_last_hidden)
      shared_kv_states:     {"full_attention": (k, v), "sliding_attention": (k, v)}
                            k, v shape: (B, n_kv_heads, L_target, head_dim)
      position_ids:         (B, L) scalar broadcast (single-position MTP)
      mask:                 optional (B, 1, L, L_target) bidirectional mask
    Returns:
      last_hidden_state:    (B, L, backbone_hidden_size) — fed back to caller
      logits:               (B, L, vocab_size)
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        tc = args.text_config

        self.pre_projection = nn.Linear(
            2 * args.backbone_hidden_size, tc["hidden_size"], bias=False
        )
        self.model = AssistantTextModel(args)
        self.post_projection = nn.Linear(
            tc["hidden_size"], args.backbone_hidden_size, bias=False
        )
        self.masked_embedding = (
            MaskedEmbedder(args) if args.use_ordered_embeddings else None
        )

    def __call__(
        self,
        inputs_embeds: mx.array,
        shared_kv_states: Dict[str, Tuple[mx.array, mx.array]],
        position_ids: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        # 5120 → 256
        h = self.pre_projection(inputs_embeds)

        # 4 decoder layers, each cross-attending to shared_kv_states
        h = self.model(h, shared_kv_states, position_ids=position_ids, mask=mask)

        # 256 → 2560 (fed back to caller as next-step input)
        last_hidden = self.post_projection(h)

        # Logits: clustered (masked_embedding) or tied-embedding matmul
        if self.masked_embedding is not None:
            logits = self.masked_embedding(h, self.model.embed_tokens.weight)
        else:
            logits = self.model.embed_tokens.as_linear(h)

        return last_hidden, logits

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Coerce token_ordering to int32 (checkpoint stores int64)."""
        out = {}
        for k, v in weights.items():
            if k == "masked_embedding.token_ordering":
                out[k] = v.astype(mx.int32)
                continue
            out[k] = v
        return out

    @property
    def quant_predicate(self):
        def predicate(path, _module):
            # Centroid Linear is only 2048*256 = 0.5M params. 4-bit hurts
            # cluster discrimination more than the ~0.25MB save is worth.
            if path.endswith("masked_embedding.centroids"):
                return False
            return True
        return predicate

    def make_cache(self):
        # The assistant owns no KV cache — it cross-attends to target K/V
        # each forward pass. Return [] so any caller that mistakenly invokes
        # make_prompt_cache(assistant) immediately fails on cache[0] rather
        # than silently miscomputing.
        return []
