import mlx.core as mx
import mlx.nn as nn
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelArgs:
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    vocab_size: int
    intermediate_size: int
    intermediate_size_mlp: int = None
    num_key_value_heads: int = 0
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    head_dim: int = None
    use_dual_mlp: bool = False
    tie_word_embeddings: bool = True
    use_qk_norm: bool = False
    attn_scale: float = 1.0
    no_rope_layers: list | None = None
    attention_chunk_size: int | None = None
    attn_temperature_tuning: bool = False

    @classmethod
    def from_dict(cls, params):
        return cls(
            hidden_size=params["hidden_size"],
            num_attention_heads=params["num_attention_heads"],
            num_hidden_layers=params["num_hidden_layers"],
            vocab_size=params["vocab_size"],
            intermediate_size=params["intermediate_size"],
            intermediate_size_mlp=params.get("intermediate_size_mlp"),
            num_key_value_heads=params.get("num_key_value_heads", 0),
            rms_norm_eps=params.get("rms_norm_eps", 1e-5),
            rope_theta=params.get("rope_theta", 10000.0),
            head_dim=params.get("head_dim"),
            # Default: off. We'll detect from weights in load_model.
            use_dual_mlp=False,
            tie_word_embeddings=params.get("tie_word_embeddings", True),
            use_qk_norm=params.get("use_qk_norm", False),
            attn_scale=params.get("attn_scale", 1.0),
            no_rope_layers=params.get("no_rope_layers"),
            attention_chunk_size=params.get("attention_chunk_size"),
            attn_temperature_tuning=params.get("attn_temperature_tuning", False),
        )


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def _norm(self, x):
        return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = (
            args.num_key_value_heads
            if args.num_key_value_heads > 0
            else args.num_attention_heads
        )
        self.head_dim = (
            args.head_dim
            if getattr(args, "head_dim", None) is not None
            else (args.hidden_size // self.n_heads)
        )
        # Use standard LLaMA scaling. The attn_scale field in some configs
        # does not correspond to SDPA scaling and degrades outputs if applied here.
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, args.hidden_size, bias=False
        )
        self.q_norm = (
            RMSNorm(self.head_dim, eps=args.rms_norm_eps)
            if getattr(args, "use_qk_norm", False)
            else None
        )
        self.k_norm = (
            RMSNorm(self.head_dim, eps=args.rms_norm_eps)
            if getattr(args, "use_qk_norm", False)
            else None
        )
        # Llama 4 text models commonly use traditional RoPE application
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=args.rope_theta)

    def __call__(
        self,
        x,
        mask=None,
        cache=None,
        apply_rope: bool = True,
        attn_temp: float | None = None,
    ):
        B, L, D = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if self.q_norm is not None:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        # Optionally apply RoPE depending on per-layer setting
        if apply_rope:
            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
                keys, values = cache.update_and_fetch(keys, values)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)
        else:
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            keys = mx.repeat(keys, repeat, axis=1)
            values = mx.repeat(values, repeat, axis=1)

        # Optional attention temperature tuning (scale the softmax input)
        scale = self.scale if attn_temp is None else (self.scale * attn_temp)
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class SwiGLUMLP(nn.Module):
    """Standard LLaMA-style gated MLP (SwiGLU)."""

    def __init__(self, dim, intermediate_size, activation=nn.silu):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

        # self.activation = activation

    def __call__(self, x):
        # return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DualMLP(nn.Module):
    """Dense dual-branch MLP: gated + plain."""

    def __init__(self, dim, intermediate_gated, intermediate_plain, activation=nn.silu):
        super().__init__()
        self.g_up = nn.Linear(dim, intermediate_gated, bias=False)
        self.g_gate = nn.Linear(dim, intermediate_gated, bias=False)
        self.g_down = nn.Linear(intermediate_gated, dim, bias=False)

        self.p_up = nn.Linear(dim, intermediate_plain, bias=False)
        self.p_down = nn.Linear(intermediate_plain, dim, bias=False)

        # self.activation = activation

    def __call__(self, x):
        # gated_out = self.g_down(self.activation(self.g_gate(x)) * self.g_up(x))
        # plain_out = self.p_down(self.activation(self.p_up(x)))
        gated_out = self.g_down(nn.silu(self.g_gate(x)) * self.g_up(x))
        plain_out = self.p_down(nn.silu(self.p_up(x)))

        return gated_out + plain_out


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.attention = Attention(args)
        self.layer_idx = layer_idx
        # RoPE gating per layer.
        # If the config provides a per-layer no_rope mask:
        # - If it disables ALL layers, ignore it (apply RoPE everywhere)
        # - Otherwise, honor the per-layer flag.
        if (
            isinstance(args.no_rope_layers, list)
            and len(args.no_rope_layers) > layer_idx
        ):
            all_marked = all(bool(v) for v in args.no_rope_layers)
            if all_marked:
                disable_rope = False
            else:
                disable_rope = bool(args.no_rope_layers[layer_idx])
        else:
            disable_rope = False
        self.apply_rope = not disable_rope
        self.layer_idx = layer_idx

        if args.use_dual_mlp and args.intermediate_size_mlp:
            self.feed_forward = DualMLP(
                args.hidden_size,
                args.intermediate_size,
                args.intermediate_size_mlp,
            )
        else:
            self.feed_forward = SwiGLUMLP(
                args.hidden_size,
                args.intermediate_size_mlp,
            )

        self.attention_norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x, mask=None, cache=None):
        L = x.shape[1]
        # Use standard causal mask; iRoPE chunking is not applied for now
        attn_mask = (
            None
            if L <= 1
            else nn.MultiHeadAttention.create_additive_causal_mask(L).astype(x.dtype)
        )
        args = self.attention.args
        apply_rope = self.apply_rope
        attn_temp = 1.0 if getattr(args, "attn_temperature_tuning", False) else None

        r = self.attention(
            self.attention_norm(x),
            attn_mask,
            cache,
            apply_rope=apply_rope,
            attn_temp=attn_temp,
        )
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        return h + r


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        # Plain Python list is fine in MLX
        self.layers = [
            TransformerBlock(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        if not self.args.tie_word_embeddings:
            self.output = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        h = self.tok_embeddings(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, None, c)

        h = self.norm(h)

        if self.args.tie_word_embeddings:
            return h @ self.tok_embeddings.weight.T
        else:
            return self.output(h)


def load_model(model_path: str):
    model_path = Path(model_path)
    with open(model_path / "config.json", "r") as f:
        config = json.load(f)

    from safetensors import safe_open
    from mlx.utils import tree_unflatten

    # Peek at weights to decide MLP variant
    with safe_open(model_path / "model.safetensors", framework="mlx") as f:
        keys = list(f.keys())
    has_dual = any(
        (".feed_forward.g_up.weight" in k)
        or (".mlp.g_up.weight" in k)
        or (".feed_forward.p_up.weight" in k)
        or (".mlp.p_up.weight" in k)
        for k in keys
    )

    args = ModelArgs.from_dict(config)
    args.use_dual_mlp = bool(has_dual)
    model = Model(args)

    weights = {}
    with safe_open(model_path / "model.safetensors", framework="mlx") as f:
        for k in f.keys():
            v = f.get_tensor(k)
            # The keys in the safetensors file are from the Hugging Face model.
            # We need to map them to the names in our MLX model.
            k = k.replace("model.embed_tokens", "tok_embeddings")
            k = k.replace("model.layers", "layers")
            k = k.replace("self_attn", "attention")
            k = k.replace("input_layernorm", "attention_norm")
            k = k.replace("post_attention_layernorm", "ffn_norm")
            k = k.replace("mlp.", "feed_forward.")
            k = k.replace("model.norm", "norm")

            # For the MLP, the names are conveniently the same if using SwiGLUMLP
            # k = k.replace("feed_forward.gate_proj", "feed_forward.gate_proj")
            # k = k.replace("feed_forward.up_proj", "feed_forward.up_proj")
            # k = k.replace("feed_forward.down_proj", "feed_forward.down_proj")

            weights[k] = v

    # The output layer is tied to the token embeddings, so we don't load weights for it separately.
    if config.get("tie_word_embeddings", True):
        weights.pop("output.weight", None)

    model.update(tree_unflatten(list(weights.items())))
    return model
