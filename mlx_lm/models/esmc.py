"""MLX implementation of ESM-C (EvolutionaryScale Cambrian) protein language model.

ESM-C is an encoder-only (bidirectional) transformer with RoPE, QK-LayerNorm,
SwiGLU feed-forward networks, and ESM3-style residual scaling. This port mirrors
the PyTorch reference in the `transformers` ESMC implementation; weight keys are
remapped in `sanitize()` (the reference fuses LayerNorm+QKV via Transformer
Engine module names).

The model is an encoder, so it is used for representations / masked-LM logits
rather than autoregressive generation. `Model.__call__` returns MLM logits;
`Model.encode` returns the final hidden state plus every intermediate hidden
state (matching the reference `output_hidden_states` stack).
"""

import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs

# Every published ESMC uses this SwiGLU expansion.
EXPANSION_RATIO = 8 / 3


# Indices are fixed by the weights; mirrors esm's SEQUENCE_VOCAB.
SEQUENCE_VOCAB = (
    "<cls> <pad> <eos> <unk> L A G V S E R T I D P K Q N F Y M H W C X B U Z O "
    ". - | <mask>"
).split()
_TOKEN_TO_ID = {t: i for i, t in enumerate(SEQUENCE_VOCAB)}
BOS_ID, PAD_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


def encode(sequence: str) -> list[int]:
    """Amino-acid string -> ``<cls>`` + residues + ``<eos>``."""
    return [BOS_ID] + [_TOKEN_TO_ID.get(c, UNK_ID) for c in sequence] + [EOS_ID]


def batch_encode(sequences) -> tuple[mx.array, mx.array]:
    """Right-pad a list of sequences. Returns (input_ids, attention_mask)."""
    ids = [encode(s) for s in sequences]
    width = max(len(i) for i in ids)
    padded = [i + [PAD_ID] * (width - len(i)) for i in ids]
    return (
        mx.array(padded),
        mx.array([[1] * len(i) + [0] * (width - len(i)) for i in ids]),
    )


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "esmc"
    vocab_size: int = 64
    hidden_size: int = 2560
    num_attention_heads: int = 40
    num_hidden_layers: int = 80
    # SwiGLU width. Authoritative when the checkpoint carries it; otherwise
    # derived from hidden_size the way esm's esmc_intermediate_size does.
    intermediate_size: Optional[int] = None
    pad_token_id: int = 1
    mask_token_id: int = 32
    # ESM3 residue-scaling reference depth.
    residue_scaling_base: float = 36.0
    layer_norm_eps: float = 1e-5
    rope_base: float = 10000.0

    @classmethod
    def from_dict(cls, params: dict) -> "ModelArgs":
        # esm's pre-alignment field names, mapped in esm/models/esmc/config.py
        # and scheduled for removal once every biohub/ESMC-* repo is republished.
        legacy = {
            "d_model": "hidden_size",
            "n_heads": "num_attention_heads",
            "n_layers": "num_hidden_layers",
        }
        params = {legacy.get(k, k): v for k, v in params.items()}
        return super().from_dict(params)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def ffn_hidden(self) -> int:
        if self.intermediate_size is not None:
            return self.intermediate_size
        return int(((EXPANSION_RATIO * self.hidden_size) + 255) // 256 * 256)

    @property
    def residue_scaling_factor(self) -> float:
        return math.sqrt(self.num_hidden_layers / self.residue_scaling_base)


class Attention(nn.Module):
    """Multi-head self-attention with QK-LayerNorm (over full hidden_size) and RoPE."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        d = args.hidden_size
        # Fused pre-norm + QKV projection in the reference (no bias on the proj).
        self.ln_qkv = nn.LayerNorm(d, eps=args.layer_norm_eps)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        # QK-LayerNorm is applied over the full model dim before the head split.
        self.q_ln = nn.LayerNorm(d, eps=args.layer_norm_eps, bias=False)
        self.k_ln = nn.LayerNorm(d, eps=args.layer_norm_eps, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=args.rope_base)

    def __call__(self, x: mx.array, mask: Optional[mx.array]) -> mx.array:
        B, L, _ = x.shape
        qkv = self.qkv(self.ln_qkv(x))
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = self.q_ln(q)
        k = self.k_ln(k)

        # (B, L, D) -> (B, heads, L, head_dim)
        def heads(t):
            return t.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = heads(q), heads(k), heads(v)
        q = self.rope(q)
        k = self.rope(k)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(out)


class FFN(nn.Module):
    """Pre-norm SwiGLU feed-forward network (bias-free)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        d, h = args.hidden_size, args.ffn_hidden
        self.ln = nn.LayerNorm(d, eps=args.layer_norm_eps)
        self.fc1 = nn.Linear(d, 2 * h, bias=False)
        self.fc2 = nn.Linear(h, d, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(self.ln(x))
        x1, x2 = mx.split(x, 2, axis=-1)
        return self.fc2((x1 * mx.sigmoid(x1)) * x2)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn = Attention(args)
        self.ffn = FFN(args)
        self.scaling_factor = args.residue_scaling_factor

    def __call__(self, x: mx.array, mask: Optional[mx.array]) -> mx.array:
        x = x + self.attn(x, mask) / self.scaling_factor
        x = x + self.ffn(x) / self.scaling_factor
        return x


class TransformerStack(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.blocks = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps, bias=False)

    def __call__(self, x: mx.array, mask: Optional[mx.array], collect):
        """``collect`` is a set of layer indices to keep, or None for none.

        Index ``i < n_layers`` is the residual stream entering block ``i``;
        index ``n_layers`` is the final post-norm output.
        """
        hidden = {}
        for i, block in enumerate(self.blocks):
            if collect and i in collect:
                hidden[i] = x
            x = block(x, mask)
        normed = self.norm(x)
        last = len(self.blocks)
        if collect and last in collect:
            hidden[last] = normed
        return normed, hidden


class ESMCModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed = nn.Embedding(args.vocab_size, args.hidden_size)
        self.transformer = TransformerStack(args)

    def __call__(self, input_ids, mask, collect):
        return self.transformer(self.embed(input_ids), mask, collect)


def _attention_mask(
    attention_mask: Optional[mx.array],
    dtype,
    sequence_id: Optional[mx.array] = None,
) -> Optional[mx.array]:
    """Build the additive chain-aware mask matching the reference.

    Attention is allowed where ``sequence_id_i == sequence_id_j``. ``sequence_id``
    may be passed directly -- which is what multi-chain inputs need, since each
    chain carries its own id -- or derived from ``attention_mask`` as the
    reference does (``sequence_id = attention_mask - 1``: padding -> -1, real -> 0).
    """
    if sequence_id is None:
        if attention_mask is None:
            return None
        sid = attention_mask.astype(mx.int32) - 1  # (B, L)
    else:
        sid = sequence_id.astype(mx.int32)
    # Short-circuit only when every token shares one id, i.e. genuinely full
    # attention. Testing "no padding" instead would wrongly drop the mask for a
    # multi-chain input, letting chains attend to each other.
    if bool(mx.all(sid == sid[:, :1]).item()):
        return None
    allowed = sid[:, :, None] == sid[:, None, :]  # (B, L, L)
    neg = mx.array(-1e9, dtype=dtype)  # large additive penalty (fp32-safe)
    add = mx.where(allowed, mx.array(0.0, dtype=dtype), neg)
    return add[:, None, :, :]  # (B, 1, L, L)


@dataclass
class EsmcOutput:
    """Mirrors esm's ``EsmcOutput``."""

    last_hidden_state: mx.array
    hidden_states: Optional[mx.array] = None
    sae_outputs: Optional[dict] = None


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.esmc = ESMCModel(args)
        # MLM head: Linear -> GELU -> LayerNorm -> Linear (indices match HF keys).
        self.lm_head = [
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.GELU(),
            nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps),
            nn.Linear(args.hidden_size, args.vocab_size),
        ]
        self.sae_layers = {}

    def add_sae_layers(self, layers):
        """Register SAEs keyed by the backbone layer each was trained against."""
        for layer in layers:
            if layer.layer in self.sae_layers:
                raise ValueError(f"an SAE is already registered at layer {layer.layer}")
            self.sae_layers[layer.layer] = layer

    def encode(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
        sequence_id=None,
        compute_sae=True,
        normalize_sae=False,
    ) -> EsmcOutput:
        """Encode ``input_ids``.

        ``hidden_states`` is (n_layers+1, B, L, hidden_size) when
        ``output_hidden_states``; index ``i`` is the input to block ``i`` and the
        last entry is the post-norm output. ``sequence_id`` gives chain-aware
        attention and takes precedence over ``attention_mask``.
        """
        n = self.args.num_hidden_layers
        sae = self.sae_layers if compute_sae else {}
        if output_hidden_states:
            collect = set(range(n + 1))
        else:
            collect = set(sae)
        mask = _attention_mask(
            attention_mask, self.esmc.embed.weight.dtype, sequence_id=sequence_id
        )
        last, hidden = self.esmc(input_ids, mask, collect)
        return EsmcOutput(
            last_hidden_state=last,
            hidden_states=(
                mx.stack([hidden[i] for i in range(n + 1)], axis=0)
                if output_hidden_states
                else None
            ),
            sae_outputs={
                i: layer(hidden[i], normalize=normalize_sae) for i, layer in sae.items()
            }
            or None,
        )

    def _lm_head(self, x):
        for layer in self.lm_head:
            x = layer(x)
        return x

    def __call__(self, input_ids, attention_mask=None):
        return self._lm_head(self.encode(input_ids, attention_mask).last_hidden_state)

    def sanitize(self, weights):
        out = {}
        for k, v in weights.items():
            if k.endswith("._extra_state"):
                continue
            k = k.replace(
                ".attn.layernorm_qkv.layer_norm_weight", ".attn.ln_qkv.weight"
            )
            k = k.replace(".attn.layernorm_qkv.layer_norm_bias", ".attn.ln_qkv.bias")
            k = k.replace(".attn.layernorm_qkv.weight", ".attn.qkv.weight")
            k = k.replace(".ffn.layer_norm_weight", ".ffn.ln.weight")
            k = k.replace(".ffn.layer_norm_bias", ".ffn.ln.bias")
            k = k.replace(".ffn.fc1_weight", ".ffn.fc1.weight")
            k = k.replace(".ffn.fc2_weight", ".ffn.fc2.weight")
            out[k] = v
        return out

    @property
    def layers(self):
        return self.esmc.transformer.blocks


def from_pretrained(repo: str = "biohub/ESMC-6B", dtype=mx.bfloat16) -> Model:
    """Load a published ESMC checkpoint. Mirrors esm's ``from_pretrained``."""
    import glob
    import json

    from huggingface_hub import snapshot_download

    local = snapshot_download(repo, allow_patterns=["config.json", "*.safetensors"])
    model = Model(ModelArgs.from_dict(json.load(open(f"{local}/config.json"))))
    weights = {}
    # Shard by shard: loading the fp32 whole and casting after peaks ~55 GB.
    for shard in sorted(glob.glob(f"{local}/*.safetensors")):
        part = {k: _cast(v, dtype) for k, v in mx.load(shard).items()}
        mx.eval(list(part.values()))
        weights.update(part)
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    model.set_dtype(dtype)
    model.eval()
    mx.eval(model.parameters())
    return model


def _cast(v, dtype):
    floating = (mx.float32, mx.float16, mx.bfloat16, mx.float64)
    return v.astype(dtype) if v.dtype in floating else v
