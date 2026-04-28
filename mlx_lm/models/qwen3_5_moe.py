# Copyright © 2026 Apple Inc.

from dataclasses import dataclass

import mlx.core as mx

from .base import BaseModelArgs
from .qwen3_5 import Model as Qwen3_5Model


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class Model(Qwen3_5Model):

    def sanitize(self, weights):
        # Dequantize FP8 weights paired with weight_scale_inv. Mirrors the
        # inline FP8 dequant in models/{deepseek_v3,deepseek_v32,minimax,
        # mimo_v2_flash,ministral3}.py. Triggered by checkpoints with
        # quant_method=fp8 (e.g. Qwen/Qwen3.6-35B-A3B-FP8). No-op when no
        # weight_scale_inv keys are present.
        def dequant(weight, scale_inv):
            weight = mx.from_fp8(weight, dtype=mx.bfloat16)
            bs = 128  # block size
            m, n = weight.shape
            pad_bottom = (-m) % bs
            pad_side = (-n) % bs
            weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
            weight = weight.reshape(
                ((m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs)
            )
            weight = (weight * scale_inv[:, None, :, None]).reshape(
                m + pad_bottom, n + pad_side
            )
            return weight[:m, :n].astype(mx.bfloat16)

        if any("weight_scale_inv" in k for k in weights):
            dequanted = {}
            for k, v in weights.items():
                if "weight_scale_inv" in k:
                    wk = k.replace("_scale_inv", "")
                    dequanted[wk] = dequant(weights[wk], v)
                elif "activation_scale" in k:
                    continue
                elif k not in dequanted:
                    dequanted[k] = v
            weights = dequanted

        new_weights = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif key.startswith("language_model."):
                pass
            else:
                key = "language_model." + key
            new_weights[key] = value

        for l in range(self.language_model.args.num_hidden_layers):
            prefix = f"language_model.model.layers.{l}.mlp"
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            if gate_up_key in new_weights:
                gate_up = new_weights.pop(gate_up_key)
                mid = gate_up.shape[-2] // 2
                new_weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[
                    ..., :mid, :
                ]
                new_weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[
                    ..., mid:, :
                ]
                new_weights[f"{prefix}.switch_mlp.down_proj.weight"] = new_weights.pop(
                    f"{prefix}.experts.down_proj"
                )
            elif f"{prefix}.experts.0.gate_proj.weight" in new_weights:
                # Per-expert layout (Qwen/Qwen3.6-35B-A3B-FP8): one tensor per
                # expert per projection. The bf16 master Qwen/Qwen3.6-35B-A3B
                # is already pre-stacked and falls through the combined-format
                # branch above unchanged. Collect all matching keys for this
                # layer, validate the index range is contiguous from 0, then
                # stack along axis 0 into the same shape the combined-format
                # branch produces.
                experts_prefix = f"{prefix}.experts."
                gate_suffix = ".gate_proj.weight"
                indices = set()
                for key in new_weights:
                    if key.startswith(experts_prefix) and key.endswith(gate_suffix):
                        tail = key[len(experts_prefix) : -len(gate_suffix)]
                        if tail.isdigit():
                            indices.add(int(tail))
                expected = set(range(len(indices)))
                if indices != expected:
                    missing = sorted(expected - indices)
                    extra = sorted(indices - expected)
                    raise ValueError(
                        f"Per-expert MoE weights at {prefix}.experts have "
                        f"non-contiguous indices: missing={missing}, "
                        f"unexpected={extra}."
                    )
                gates, ups, downs = [], [], []
                for e in range(len(indices)):
                    gates.append(
                        new_weights.pop(f"{prefix}.experts.{e}.gate_proj.weight")
                    )
                    ups.append(
                        new_weights.pop(f"{prefix}.experts.{e}.up_proj.weight")
                    )
                    downs.append(
                        new_weights.pop(f"{prefix}.experts.{e}.down_proj.weight")
                    )
                new_weights[f"{prefix}.switch_mlp.gate_proj.weight"] = mx.stack(gates)
                new_weights[f"{prefix}.switch_mlp.up_proj.weight"] = mx.stack(ups)
                new_weights[f"{prefix}.switch_mlp.down_proj.weight"] = mx.stack(downs)

        return self.language_model.sanitize(new_weights)
