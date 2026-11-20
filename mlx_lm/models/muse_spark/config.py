# Copyright © 2026 Apple Inc.

from dataclasses import dataclass, field

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
    "TOKENIZER_DEFAULTS",
    "TextArgs",
    "VisionArgs",
]

# For checkpoints without them. Stop on <|eot|>, not on <|eom|>.
TOKENIZER_DEFAULTS = {
    "chat_template_type": "muse_spark",
    "tool_parser_type": "muse_spark",
    "eos_token_ids": [200001, 200008],
}


@dataclass
class TextArgs(BaseModelArgs):
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    vocab_size: int
    rms_norm_eps: float
    post_norm_eps: float
    rope_parameters: dict
    # A 0 marks a layer without RoPE.
    layer_rope_theta: list[float]
    layer_types: list[str]
    sliding_window: int
    final_logit_softcapping: float
    output_multiplier: float
    num_local_experts: int
    num_experts_per_tok: int
    moe_hidden_size: int
    moe_intermediate_size: int
    qk_scale_factor: float = 5.4730064863875
    routed_scaling_factor: float = 1.0
    num_shared_experts: int = 0
    mlp_layer_types: list[str] | None = None

    def __post_init__(self) -> None:
        if self.num_shared_experts or any(
            t != "sparse" for t in self.mlp_layer_types or []
        ):
            raise NotImplementedError("Dense feed-forward layers are not supported.")
        self.rope_theta = self.rope_parameters["rope_theta"]


@dataclass
class VisionArgs(BaseModelArgs):
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 50
    num_attention_heads: int = 16
    num_channels: int = 3
    patch_size: int = 14
    patch_temporal: int = 2
    merge_size: int = 2
    # The position table; images can be larger or smaller.
    pos_emb_height: int = 32
    pos_emb_width: int = 32
    layer_norm_eps: float = 1e-5
    rope_parameters: dict = field(default_factory=dict)
    # window_attention attends inside pos_emb blocks of patches.
    layer_types: list[str] | None = None
    use_ij_pos_encoding: bool = False

    def __post_init__(self) -> None:
        if self.pos_emb_height != self.pos_emb_width:
            raise NotImplementedError("Only square position tables are supported.")
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.grid_size = self.pos_emb_height
        if self.layer_types is None:
            # As transformers: full attention in every 4th and the last layer.
            self.layer_types = [
                (
                    "full_attention"
                    if (i + 1) % 4 == 0 or i == self.num_hidden_layers - 1
                    else "window_attention"
                )
                for i in range(self.num_hidden_layers)
            ]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict
    vision_config: dict | None = None
    # Image rows go at <|patch|>: the processor expands each <|patch|> of the
    # prompt to the rows of one image.
    image_token_id: int = 200092
    out_hidden_size: int | None = None
    projector_hidden_size: int | None = None

    def __post_init__(self) -> None:
        self.text = TextArgs.from_dict(self.text_config)
        self.vision = None
        if self.vision_config:
            self.vision = VisionArgs.from_dict(self.vision_config)
            if self.out_hidden_size is None:
                self.out_hidden_size = (
                    self.vision.hidden_size * self.vision.merge_size**2
                )
            if self.projector_hidden_size is None:
                self.projector_hidden_size = self.out_hidden_size
