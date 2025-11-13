# ABOUTME: Installs server-scoped paged attention hooks at runtime.
# ABOUTME: Wraps model attention layers and patches SDPA to call paged kernels.

from __future__ import annotations

import functools
import logging
import os
import sys
from typing import Optional

import mlx.core as mx

from ..models.cache import PagedKVCacheAdapter  # noqa: F401 to ensure module load
from ..server_batched import slot_generator  # delayed import to avoid cyclical load
from .paged_context import current_batch_view, current_layer

try:  # pragma: no cover - depends on mlx install layout
    from mlx.nn.paged_kv import (
        _paged_prefill_reference as _mx_paged_prefill_reference,  # type: ignore[attr-defined]
    )
except Exception:  # pragma: no cover - fallback when import path changes
    _mx_paged_prefill_reference = None  # type: ignore[assignment]

_PATCHED = False
_DISABLED_QUANT_MODES: set[str] = set()


def _paged_attention_call(
    queries,
    k_cache,
    v_cache,
    block_tables,
    context_lens,
    kv_head_mapping,
    scale,
    quant_kwargs,
    prefill_base_lens=None,
):
    call_kwargs = dict(quant_kwargs)
    call_kwargs["kv_head_mapping"] = kv_head_mapping
    call_kwargs["scale"] = scale
    call_kwargs.setdefault("layer_idx", quant_kwargs.get("layer_idx"))

    paged_attention_fn = getattr(mx.fast, "paged_attention", None)
    if paged_attention_fn is None:
        return None

    def _invoke(q_slice):
        q_slice = mx.array(q_slice)
        q_slice = mx.contiguous(q_slice)
        kv_mode = call_kwargs.get("kv_quant_mode")
        if kv_mode in _DISABLED_QUANT_MODES:
            return None
        try:
            return paged_attention_fn(
                q_slice,
                k_cache,
                v_cache,
                block_tables,
                context_lens,
                **call_kwargs,
            )
        except Exception as exc:
            if kv_mode:
                if kv_mode not in _DISABLED_QUANT_MODES:
                    _DISABLED_QUANT_MODES.add(kv_mode)
                    logging.warning(
                        "mx.fast.paged_attention disabled for kv_quant=%s; falling back to SDPA",
                        kv_mode,
                    )
            else:
                logging.warning(
                    "mx.fast.paged_attention failed; falling back to SDPA: %s", exc
                )
            return None

    _, _, lq, _ = queries.shape
    if lq == 1:
        return _invoke(queries)

    if prefill_base_lens is not None:
        paged_prefill = getattr(mx.fast, "paged_prefill", None)
        if paged_prefill is not None:
            queries = mx.contiguous(queries)
            try:
                return paged_prefill(
                    queries,
                    k_cache,
                    v_cache,
                    block_tables,
                    prefill_base_lens,
                    context_lens,
                    kv_head_mapping=kv_head_mapping,
                    scale=scale,
                    **quant_kwargs,
                )
            except Exception as exc:
                kv_mode = quant_kwargs.get("kv_quant_mode")
                if kv_mode:
                    _DISABLED_QUANT_MODES.add(kv_mode)
                    logging.warning(
                        "mx.fast.paged_prefill disabled for kv_quant=%s; falling back to decode slices",
                        kv_mode,
                    )
                else:
                    logging.warning(
                        "mx.fast.paged_prefill failed; falling back to decode slices: %s",
                        exc,
                    )
        if _mx_paged_prefill_reference is not None:
            try:
                return _mx_paged_prefill_reference(
                    queries,
                    k_cache,
                    v_cache,
                    block_tables,
                    context_lens,
                    scale=scale,
                    base_lens=prefill_base_lens,
                    kv_head_mapping=kv_head_mapping,
                    **quant_kwargs,
                )
            except Exception as exc:
                logging.warning(
                    "reference paged_prefill failed; falling back to decode slices: %s",
                    exc,
                )
        logging.warning("paged_prefill unavailable; falling back to decode tiles")

    tile = max(1, int(os.getenv("MLXLM_PREFILL_AS_DECODE_TILE", "1")))
    outputs = []
    for start in range(0, lq, tile):
        end = min(start + tile, lq)
        q_tile = queries[:, :, start:end, :]
        result = _invoke(q_tile)
        if result is None:
            return None
        outputs.append(result)
    return mx.concatenate(outputs, axis=2)


def install_paged_attention_patch() -> None:
    """Monkey-patch scaled_dot_product_attention once per process."""
    global _PATCHED
    if _PATCHED:
        return
    import mlx_lm.models.base as base

    original = base.scaled_dot_product_attention

    @functools.wraps(original)
    def _patched(
        queries,
        keys,
        values,
        cache,
        scale: float,
        mask: Optional[mx.array],
        sinks: Optional[mx.array] = None,
    ):
        layer_idx = current_layer()
        view = current_batch_view()
        if layer_idx is not None and view is not None:
            (
                k_cache,
                v_cache,
                block_tables,
                context_lens,
                _layer_token,
                kv_head_mapping,
                quant_kwargs,
            ) = view.args_for_layer(layer_idx)
            quant_kwargs = dict(quant_kwargs)
            quant_kwargs["layer_idx"] = layer_idx
            base_lens = getattr(view, "prefill_base_lens", None)
            result = _paged_attention_call(
                queries,
                k_cache,
                v_cache,
                block_tables,
                context_lens,
                kv_head_mapping,
                scale,
                quant_kwargs,
                prefill_base_lens=base_lens,
            )
            if result is not None:
                return result
        logging.info(
            "paged_attention_fallback layer=%s view=%s",
            layer_idx,
            "set" if view is not None else "none",
        )
        return original(
            queries,
            keys,
            values,
            cache=cache,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )

    base.scaled_dot_product_attention = _patched
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if not name.startswith("mlx_lm.models."):
            continue
        if hasattr(module, "scaled_dot_product_attention"):
            setattr(module, "scaled_dot_product_attention", _patched)
    _PATCHED = True


__all__ = ["install_paged_attention_patch"]
