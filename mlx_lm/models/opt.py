# mlx_lm/models/opt.py
from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache


def _act(name: str):
    n = (name or "relu").lower()
    if n in ("relu",):
        return nn.relu
    if n in ("gelu", "gelu_new"):
        return nn.gelu
    if n in ("silu", "swish"):
        return nn.silu
    # OPT is typically relu; fallback safely
    return nn.relu


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    ffn_dim: int
    num_hidden_layers: int
    num_attention_heads: int
    max_position_embeddings: int
    layer_norm_eps: float = 1e-5

    # OPT-specific (present in HF configs)
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_function: str = "relu"

    # Embedding projection (OPT has this field; sometimes equals hidden_size)
    word_embed_proj_dim: int = 0

    pad_token_id: int = 1

    # OPT-specific: If True, layer norm is applied before attention (pre-LN)
    # If False, there's no final_layer_norm in the decoder
    do_layer_norm_before: bool = True

    # Some OPT checkpoints have this set to True (removes final layer norm)
    _remove_final_layer_norm: bool = False


class OPTAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.head_dim = dim // self.n_heads
        assert dim % self.n_heads == 0, "hidden_size must be divisible by num_attention_heads"
        self.scale = self.head_dim**-0.5

        # OPT uses biases in attention projections
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, D = x.shape

        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            # mlx-lm cache API pattern :contentReference[oaicite:1]{index=1}
            k, v = cache.update_and_fetch(k, v)

        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.out_proj(out)


class OPTMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.fc1 = nn.Linear(dim, args.ffn_dim, bias=True)
        self.fc2 = nn.Linear(args.ffn_dim, dim, bias=True)
        self.act = _act(args.activation_function)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class OPTDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.do_layer_norm_before = args.do_layer_norm_before
        self.self_attn = OPTAttention(args)
        self.self_attn_layer_norm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.mlp = OPTMLP(args)
        self.final_layer_norm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        if self.do_layer_norm_before:
            # Pre-LayerNorm: LN -> Attention -> Residual
            r = self.self_attn(self.self_attn_layer_norm(x), mask=mask, cache=cache)
            x = x + r
            r = self.mlp(self.final_layer_norm(x))
            x = x + r
        else:
            # Post-LayerNorm: Attention -> Residual -> LN
            r = self.self_attn(x, mask=mask, cache=cache)
            x = self.self_attn_layer_norm(x + r)
            r = self.mlp(x)
            x = self.final_layer_norm(x + r)
        return x


class OPTDecoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size
        self.num_hidden_layers = args.num_hidden_layers

        embed_dim = args.word_embed_proj_dim or args.hidden_size

        # HF OPT uses embed_tokens in embed_dim, then (optionally) project_in -> hidden_size
        self.embed_tokens = nn.Embedding(args.vocab_size, embed_dim)

        # HF OPT learned positional embeddings have +2 offset; weights are (max_pos + 2, hidden_size)
        self.embed_positions = nn.Embedding(args.max_position_embeddings + 2, args.hidden_size)

        # Always define these so checkpoints that include them load cleanly
        self.project_in = nn.Linear(embed_dim, args.hidden_size, bias=False)
        self.project_out = nn.Linear(args.hidden_size, embed_dim, bias=False)

        self.layers = [OPTDecoderLayer(args) for _ in range(args.num_hidden_layers)]

        # HF OPT has decoder.final_layer_norm only when do_layer_norm_before=True
        # When do_layer_norm_before=False, there's no final layer norm
        if args.do_layer_norm_before and not args._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        else:
            self.final_layer_norm = None

    def _position_ids(self, inputs: mx.array, cache) -> mx.array:
        # inputs: [B, L]
        L = inputs.shape[1]
        if cache is not None and len(cache) > 0 and cache[0] is not None:
            offset = getattr(cache[0], "offset", 0)
        else:
            offset = 0
        # HF offset is +2
        return mx.arange(offset, offset + L, dtype=mx.int32) + 2

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        # inputs: [B, L]
        h = self.embed_tokens(inputs)
        h = self.project_in(h)

        pos = self._position_ids(inputs, cache)
        pos_emb = self.embed_positions(pos)[None, :, :]
        h = h + pos_emb.astype(h.dtype)

        # build causal mask if not provided
        if mask is None:
            mask = create_attention_mask(h, cache)

        # optional padding mask support (right-pad safe; still good to apply)
        if attention_mask is not None:
            # attention_mask: 1 for real, 0 for pad
            am = attention_mask.astype(mx.bool_)
            # create_attention_mask typically masks keys dimension; broadcast AND
            # If cache extends keys, left-extend with True for past tokens.
            k_len = mask.shape[-1]
            if am.shape[1] != k_len:
                past = k_len - am.shape[1]
                if past > 0:
                    am = mx.concatenate([mx.ones((am.shape[0], past), mx.bool_), am], axis=1)
            mask = mask & am[:, None, None, :]

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask=mask, cache=c)

        # Apply final layer norm if it exists
        if self.final_layer_norm is not None:
            h = self.final_layer_norm(h)
        return h


class Model(nn.Module):
    """
    Top-level OPT CausalLM for mlx-lm.

    IMPORTANT: We intentionally expose parameters under:
      - decoder.*
      - lm_head.*
    so HF safetensors with keys like `decoder.layers.0.self_attn.q_proj.weight`
    load without fights.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type

        self.decoder = OPTDecoder(args)

        # OPT ties lm_head with token embeddings
        # lm_head operates on embed_dim, not hidden_size (project_out converts hidden->embed)
        embed_dim = args.word_embed_proj_dim or args.hidden_size
        self.lm_head = nn.Linear(embed_dim, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.decoder(inputs, mask=mask, cache=cache, attention_mask=attention_mask)
        # Apply project_out if it exists to convert hidden_size -> embed_dim
        if self.decoder.project_out is not None and self.args.word_embed_proj_dim != self.args.hidden_size:
            h = self.decoder.project_out(h)
        return self.lm_head(h)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        import re

        out: Dict[str, mx.array] = {}

        for k, v in weights.items():
            nk = k

            # Strip top-level HF prefix
            if nk.startswith("model."):
                nk = nk[len("model.") :]

            # Normalize common OPT decoder prefix variants
            if nk.startswith("model.decoder."):
                nk = "decoder." + nk[len("model.decoder.") :]
            if nk.startswith("decoder."):
                pass  # ok
            elif nk.startswith("decoder_"):
                nk = "decoder." + nk[len("decoder_") :]

            # ---- MLP name fix (HF: layers.N.fc1/fc2) -> (ours: layers.N.mlp.fc1/fc2)
            nk = re.sub(r"(decoder\.layers\.\d+)\.fc1\.", r"\1.mlp.fc1.", nk)
            nk = re.sub(r"(decoder\.layers\.\d+)\.fc2\.", r"\1.mlp.fc2.", nk)

            # ---- HARD MAP: decoder final layer norm
            # HF variants sometimes appear as:
            #   decoder.final_layer_norm.weight
            #   model.decoder.final_layer_norm.weight  (already stripped)
            #   decoder.layer_norm.*  (rare, but handle)
            if nk in ("final_layer_norm.weight", "final_layer_norm.bias"):
                nk = "decoder." + nk
            if nk in ("decoder.layer_norm.weight", "decoder.layer_norm.bias"):
                nk = nk.replace("decoder.layer_norm.", "decoder.final_layer_norm.")

            # ---- HARD MAP: lm_head
            # HF variants:
            #   lm_head.weight
            #   model.lm_head.weight (already stripped)
            if nk == "lm_head.weight":
                # keep as-is (mlx-lm convert expects lm_head.weight)
                pass
            elif nk == "model.lm_head.weight":
                nk = "lm_head.weight"

            out[nk] = v

        # OPT ties lm_head weights to embed_tokens - if lm_head.weight is missing,
        # copy it from decoder.embed_tokens.weight
        if "lm_head.weight" not in out:
            if "decoder.embed_tokens.weight" in out:
                # HF OPT: lm_head is tied to embed_tokens
                # If word_embed_proj_dim != hidden_size, we need to account for project_out
                # But typically for OPT, the tied weight is directly from embed_tokens
                embed_weight = out["decoder.embed_tokens.weight"]
                
                # Check if we need to apply project_out
                # project_out maps from hidden_size back to embed_dim
                if "decoder.project_out.weight" in out:
                    # lm_head should use project_out @ hidden instead of direct embed
                    # But actually, in HF OPT the lm_head is tied to embed_tokens directly
                    # So we just copy embed_tokens.weight
                    out["lm_head.weight"] = embed_weight
                else:
                    out["lm_head.weight"] = embed_weight

        # Check if decoder.final_layer_norm exists in the checkpoint
        # If not, remove it from the model structure
        has_final_layer_norm = any("final_layer_norm" in k for k in out.keys())
        if not has_final_layer_norm and self.decoder.final_layer_norm is not None:
            # Remove final_layer_norm from decoder if checkpoint doesn't have it
            self.decoder.final_layer_norm = None

        return out

    @property
    def layers(self):
        return self.decoder.layers

    def make_cache(self):
        # Standard per-layer KVCache list :contentReference[oaicite:2]{index=2}
        return [KVCache() for _ in range(len(self.layers))]
