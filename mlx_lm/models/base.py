# Copyright © 2023 Apple Inc.

import inspect
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
from mlx.utils import tree_map


def _flat_rope_config(rope_parameters) -> dict:
    """Recover the flat RoPE config keys from transformers >= 5 ``rope_parameters``.

    transformers 5 saves ``rope_theta``, ``rope_scaling`` and
    ``partial_rotary_factor`` inside ``rope_parameters``, nested by layer type
    when full and sliding attention layers use different RoPE settings.
    """
    if not isinstance(rope_parameters, dict):
        return {}
    config = {}
    if "rope_theta" not in rope_parameters and "full_attention" in rope_parameters:
        sliding = rope_parameters.get("sliding_attention") or {}
        if "rope_theta" in sliding:
            config["rope_local_base_freq"] = sliding["rope_theta"]
        rope_parameters = rope_parameters["full_attention"] or {}
    for key in (
        "rope_theta",
        "partial_rotary_factor",
        "original_max_position_embeddings",
    ):
        if key in rope_parameters:
            config[key] = rope_parameters[key]
    rope_type = rope_parameters.get("rope_type") or rope_parameters.get(
        "type", "default"
    )
    if rope_type == "default":
        config["rope_scaling"] = None
    else:
        scaling = {
            k: v
            for k, v in rope_parameters.items()
            if k not in ("rope_theta", "partial_rotary_factor")
        }
        scaling["type"] = scaling["rope_type"] = rope_type
        config["rope_scaling"] = scaling
    return config


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        fields = inspect.signature(cls).parameters
        # Models that read rope_parameters themselves are left to do so.
        if "rope_parameters" in params and "rope_parameters" not in fields:
            params = {**_flat_rope_config(params["rope_parameters"]), **params}
        return cls(**{k: v for k, v in params.items() if k in fields})


def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: Optional[int] = None,
    right_padding: Optional[mx.array] = None,
    left_padding: Optional[mx.array] = None,
):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    if right_padding is not None:
        mask = mask & (rinds < mx.expand_dims((offset + N) - right_padding, (1, 2, 3)))
    if left_padding is not None:
        mask = mask & (mx.expand_dims(left_padding, (1, 2, 3)) <= rinds)
    return mask


def create_attention_mask(
    h, cache=None, window_size: Optional[int] = None, return_array: bool = False
):
    N = h.shape[1]
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(N, return_array=return_array, window_size=window_size)
    if N == 1:
        return None
    if return_array or (window_size and N > window_size):
        return create_causal_mask(N, window_size=window_size)
    return "causal"


def create_ssm_mask(h, cache=None):
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(h.shape[1])
    return None


def quantized_scaled_dot_product_attention(
    queries: mx.array,
    q_keys: tuple[mx.array, mx.array, mx.array],
    q_values: tuple[mx.array, mx.array, mx.array],
    scale: float,
    mask: Optional[mx.array],
    group_size: int = 64,
    bits: int = 8,
) -> mx.array:
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    queries *= scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True, group_size=group_size, bits=bits
    )
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        if n_repeats > 1 and mask.ndim > 3:
            mask = mx.expand_dims(mask, -3)
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores += mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(
        scores, *q_values, transpose=False, group_size=group_size, bits=bits
    )

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))

    return out


def scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[mx.array],
    sinks: Optional[mx.array] = None,
) -> mx.array:
    if hasattr(cache, "bits"):
        if sinks is not None:
            raise ValueError("Quantized SDPA does not support attention sinks.")
        return quantized_scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            group_size=cache.group_size,
            bits=cache.bits,
        )
    else:
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )


def gather_last_axis(x: mx.array, group: mx.distributed.Group) -> mx.array:
    """Gather the last axis of ``x``, which is split across ``group``."""
    n, size = group.size(), x.shape[-1]
    parts = mx.distributed.all_gather(x.reshape(-1, size), group=group)
    # (ranks * rows, size) to (rows, ranks * size), in rank order.
    parts = parts.reshape(n, -1, size).transpose(1, 0, 2)
    return parts.reshape(*x.shape[:-1], n * size)
