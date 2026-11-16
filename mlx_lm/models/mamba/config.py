# Copyright © 2024 Apple Inc.

import math
from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    state_size: int
    num_hidden_layers: int
    conv_kernel: int
    use_bias: bool
    use_conv_bias: bool
    time_step_rank: int
    tie_word_embeddings: bool = True
    use_bcdt_rms: bool = False
    mixer_rms_eps: float = 1e-6

    def __post_init__(self):
        if not hasattr(self, "hidden_size") and hasattr(self, "d_model"):
            self.hidden_size = self.d_model
        if not hasattr(self, "intermediate_size") and hasattr(self, "d_inner"):
            self.intermediate_size = self.d_inner
        if not hasattr(self, "state_size") and hasattr(self, "d_state"):
            self.state_size = self.d_state
        if not hasattr(self, "num_hidden_layers") and hasattr(self, "n_layer"):
            self.num_hidden_layers = self.n_layer
        if not hasattr(self, "num_hidden_layers") and hasattr(self, "n_layers"):
            self.num_hidden_layers = self.n_layers
        if not hasattr(self, "conv_kernel") and hasattr(self, "d_conv"):
            self.conv_kernel = self.d_conv
        if not hasattr(self, "use_bias") and hasattr(self, "bias"):
            self.use_bias = self.bias
        if not hasattr(self, "use_conv_bias") and hasattr(self, "conv_bias"):
            self.use_conv_bias = self.conv_bias

        if self.time_step_rank == "auto":
            self.time_step_rank = math.ceil(self.hidden_size / 16)
        if self.model_type == "falcon_mamba":
            self.use_bcdt_rms = True
