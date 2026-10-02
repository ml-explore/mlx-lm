# Copyright © 2024-2026 Apple Inc.

from . import ministral3


class ModelArgs(ministral3.ModelArgs):
    pass


class Model(ministral3.Model):
    """
    Nemotron-Labs-Diffusion model in MLX (Autoregressive mode).
    Inherits from Ministral3, remapping diffusion_head to lm_head,
    mapping encoder.* to model.*, and stripping legacy 'language_model.' prefixes.
    """

    def sanitize(self, weights):
        cleaned = {}
        for k, v in weights.items():
            if k.startswith("language_model."):
                k = k[len("language_model.") :]
            if k.startswith("encoder."):
                k = "model." + k[len("encoder.") :]
            if k.startswith("diffusion_head."):
                k = "lm_head." + k[len("diffusion_head.") :]
            cleaned[k] = v
        return super().sanitize(cleaned)
