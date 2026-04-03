"""SSD-backed residual streaming for Apple Silicon.

Streams quantized residuals from SSD per-layer during inference,
leveraging Apple Silicon's fast NVMe (~7 GB/s) and unified memory.

During forward pass:
1. Before computing layer N, prefetch layer N+1's residuals from SSD
2. Attach residuals to layer N's RPSLinear modules
3. After layer N completes, detach and free residuals
4. Repeat

This keeps only ~250MB of residuals in memory at a time (one layer),
while the base model (3-bit) stays fully resident.
"""

from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import RPSConfig
from .linear import RPSLinear


class ResidualStreamer:
    """Manages per-layer residual streaming from SSD."""

    def __init__(self, residual_dir: str, config: Optional[RPSConfig] = None):
        self.residual_dir = Path(residual_dir)
        self.config = config or RPSConfig()

        # Discover layer files
        self._layer_files = sorted(self.residual_dir.glob("residual-layer-*.safetensors"))
        self._cache: Dict[int, dict] = {}
        self._attached_layer: Optional[int] = None

    @property
    def num_layers(self) -> int:
        return len(self._layer_files)

    def prefetch(self, layer_idx: int):
        """Load a layer's residuals into memory (mmap-backed, fast)."""
        if layer_idx in self._cache or layer_idx >= len(self._layer_files):
            return
        self._cache[layer_idx] = mx.load(str(self._layer_files[layer_idx]))

    def get(self, layer_idx: int, weight_name: str):
        """Get residual arrays for a specific weight."""
        if layer_idx not in self._cache:
            self.prefetch(layer_idx)
        data = self._cache.get(layer_idx, {})
        r_w = data.get(f"{weight_name}.weight")
        r_s = data.get(f"{weight_name}.scales")
        r_b = data.get(f"{weight_name}.biases")
        if r_w is not None:
            mx.eval(r_w, r_s, r_b)  # force page-in from SSD
        return r_w, r_s, r_b

    def evict(self, layer_idx: int):
        """Free a layer's residuals from memory."""
        self._cache.pop(layer_idx, None)

    def attach_to_layer(self, model_layer, layer_idx: int, r_bits: int = 2, r_group_size: int = 64):
        """Attach residuals to all RPSLinear modules in a model layer."""
        attached = 0
        for attr in ("self_attn", "attention", "attn"):
            attn = getattr(model_layer, attr, None)
            if attn is None:
                continue
            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                proj = getattr(attn, proj_name, None)
                if isinstance(proj, RPSLinear) and not proj.has_residual:
                    r_w, r_s, r_b = self.get(layer_idx, f"{attr}.{proj_name}")
                    if r_w is not None:
                        proj.attach_residual(r_w, r_s, r_b, r_bits, r_group_size)
                        attached += 1
            break

        mlp = getattr(model_layer, "mlp", None)
        if mlp is not None:
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                proj = getattr(mlp, proj_name, None)
                if isinstance(proj, RPSLinear) and not proj.has_residual:
                    r_w, r_s, r_b = self.get(layer_idx, f"mlp.{proj_name}")
                    if r_w is not None:
                        proj.attach_residual(r_w, r_s, r_b, r_bits, r_group_size)
                        attached += 1

        self._attached_layer = layer_idx
        return attached

    def detach_from_layer(self, model_layer):
        """Detach all residuals from a model layer's RPSLinear modules."""
        for attr in ("self_attn", "attention", "attn"):
            attn = getattr(model_layer, attr, None)
            if attn is None:
                continue
            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                proj = getattr(attn, proj_name, None)
                if isinstance(proj, RPSLinear):
                    proj.detach_residual()
            break

        mlp = getattr(model_layer, "mlp", None)
        if mlp is not None:
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                proj = getattr(mlp, proj_name, None)
                if isinstance(proj, RPSLinear):
                    proj.detach_residual()

        if self._attached_layer is not None:
            self.evict(self._attached_layer)
            self._attached_layer = None


class _StreamingLayerWrapper(nn.Module):
    """Wraps a transformer layer to attach/detach residuals per forward pass."""

    def __init__(self, layer, layer_idx, streamer):
        super().__init__()
        self._inner = layer
        self._idx = layer_idx
        self._streamer = streamer

    def __call__(self, *args, **kwargs):
        # Attach residuals for this layer
        self._streamer.attach_to_layer(
            self._inner, self._idx,
            r_bits=self._streamer.config.residual_bits,
            r_group_size=self._streamer.config.residual_group_size,
        )
        # Prefetch next layer
        self._streamer.prefetch(self._idx + 1)

        result = self._inner(*args, **kwargs)

        # Detach and free
        self._streamer.detach_from_layer(self._inner)
        return result


def streaming_generate(
    model,
    tokenizer,
    streamer: ResidualStreamer,
    prompt: str,
    max_tokens: int = 100,
    verbose: bool = True,
    **kwargs,
):
    """Generate with per-layer SSD residual streaming.

    Wraps each transformer layer to attach/detach residuals around its
    computation. Only one layer's residuals are in memory at a time.
    """
    from mlx_lm import generate as mlx_generate

    # Wrap layers with streaming hooks
    original_layers = model.layers
    model.layers = [
        _StreamingLayerWrapper(layer, i, streamer)
        for i, layer in enumerate(original_layers)
    ]

    try:
        output = mlx_generate(model, tokenizer, prompt=prompt,
                              max_tokens=max_tokens, verbose=verbose, **kwargs)
    finally:
        # Restore original layers
        model.layers = original_layers

    return output
