"""MLX implementation of the ESM-C sparse autoencoders.

Port of `esm.models.esmc.sae`. A published repo holds one SAE per backbone
layer for a fixed `(d_model, codebook_dim, k)`, sharded as `layer_{i}.safetensors`.
`SaeModel.from_pretrained` downloads the snapshot but materializes nothing;
callers load the layers they need with `initialize_layers`.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

CONFIG_NAME = "config.json"


@dataclass
class SaeConfig:
    d_model: int = 2560
    codebook_dim: int = 65536
    k: int = 64
    available_layers: tuple = (0,)

    @classmethod
    def from_pretrained(cls, directory) -> "SaeConfig":
        raw = json.loads((Path(directory) / CONFIG_NAME).read_text())
        return cls(
            d_model=raw["d_model"],
            codebook_dim=raw["codebook_dim"],
            k=raw["k"],
            available_layers=tuple(raw.get("available_layers") or [0]),
        )


class SaeLayer(nn.Module):
    """One backbone layer's SAE: z-score, top-k encode, linear decode."""

    def __init__(self, d_model: int, codebook_dim: int, k: int, layer: int):
        super().__init__()
        self.k = k
        self.layer = layer
        self.W_enc = mx.zeros((d_model, codebook_dim))
        self.W_dec = mx.zeros((codebook_dim, d_model))
        self.b_dec = mx.zeros((d_model,))
        # Per-feature normalization stats; ones make `features / max * idf` a no-op
        # for variants that do not ship them.
        self.idf = mx.ones((codebook_dim,))
        self.max = mx.ones((codebook_dim,))

    def __call__(self, x: mx.array, normalize: bool = False) -> mx.array:
        mean = mx.mean(x, axis=-1, keepdims=True)
        centered = x - mean
        std = mx.sqrt(mx.var(centered, axis=-1, keepdims=True, ddof=1))
        x = centered / (std + 1e-5)

        pre = nn.relu((x - self.b_dec) @ self.W_enc)
        # Zero everything below the k-th largest; equivalent to a top-k scatter.
        kth = mx.min(mx.topk(pre, self.k, axis=-1), axis=-1, keepdims=True)
        features = mx.where(pre >= kth, pre, 0)
        return (features / self.max) * self.idf if normalize else features

    def reconstruct(self, features: mx.array) -> mx.array:
        return features @ self.W_dec + self.b_dec


class SaeModel:
    """Container holding one `SaeLayer` per backbone layer of one repo."""

    def __init__(self, config: SaeConfig, snapshot_dir: str):
        self.config = config
        self.snapshot_dir = Path(snapshot_dir)
        self.layers: dict[int, SaeLayer] = {}

    @classmethod
    def from_pretrained(cls, repo, dtype=mx.float32) -> "SaeModel":
        path = Path(repo)
        if not (path / CONFIG_NAME).exists():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(repo))
        model = cls(SaeConfig.from_pretrained(path), str(path))
        if len(model.config.available_layers) == 1:
            model.initialize_layers(model.config.available_layers, dtype=dtype)
        return model

    def initialize_layers(self, layers, dtype=mx.float32) -> None:
        """Load the named layers from the snapshot. Idempotent."""
        cfg = self.config
        for idx in layers:
            if idx in self.layers:
                continue
            if idx not in cfg.available_layers:
                raise KeyError(
                    f"layer {idx} is not in this repo; "
                    f"available_layers={list(cfg.available_layers)}"
                )
            layer = SaeLayer(cfg.d_model, cfg.codebook_dim, cfg.k, idx)
            weights = mx.load(str(self.snapshot_dir / f"layer_{idx}.safetensors"))
            layer.load_weights([(k, v.astype(dtype)) for k, v in weights.items()])
            layer.eval()
            mx.eval(layer.parameters())
            self.layers[idx] = layer

    def release_layer(self, layer: int) -> None:
        self.layers.pop(layer, None)

    def __call__(self, x: mx.array, layer: Optional[int] = None, **kwargs) -> mx.array:
        if layer is None:
            if len(self.layers) != 1:
                raise RuntimeError(
                    f"select a layer; loaded={sorted(self.layers)}, "
                    f"available={list(self.config.available_layers)}"
                )
            (layer,) = self.layers
        if layer not in self.layers:
            raise KeyError(f"layer {layer} is not loaded; call initialize_layers")
        return self.layers[layer](x, **kwargs)
