# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from . import lfm2
from .base import BaseModelArgs
from .lfm2_vl_vision import Projector, VisionConfig, VisionModel


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict
    vision_config: Optional[dict] = None
    image_token_id: Optional[int] = None
    image_token_index: Optional[int] = None
    vision_feature_layer: int = -1
    downsample_factor: int = 2
    projector_hidden_size: int = 2560
    projector_hidden_act: str = "gelu"
    projector_bias: bool = True
    projector_use_layernorm: Optional[bool] = True

    def __post_init__(self):
        self.text_config["tie_word_embeddings"] = False
        # Older configs (model_type "lfm2-vl") name it image_token_index.
        if self.image_token_id is None:
            self.image_token_id = self.image_token_index
        if self.projector_use_layernorm is None:
            self.projector_use_layernorm = True


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = lfm2.Model(lfm2.ModelArgs.from_dict(args.text_config))
        self.vision_tower = None
        if args.vision_config is not None and args.image_token_id is not None:
            vision = VisionConfig.from_dict(args.vision_config)
            # Older checkpoints take the features of an earlier layer
            # (vision_feature_layer -2) and store only the layers up to it.
            if args.vision_feature_layer not in (None, -1):
                vision.num_hidden_layers = args.vision_feature_layer % vision.num_hidden_layers + 1
            self.vision_tower = VisionModel(vision)
            self.multi_modal_projector = Projector(
                vision.hidden_size,
                self.language_model.args.hidden_size,
                args.projector_hidden_size,
                args.downsample_factor,
                args.projector_bias,
                args.projector_hidden_act,
                args.projector_use_layernorm,
            )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )

    def image_features(self, crops: List[Tuple[mx.array, int, int]]) -> mx.array:
        """Image tokens [N, D] for crops given as (patches [h * w, C*p*p], h, w),
        in order. Crops of the same size run as one batch."""
        out = [None] * len(crops)
        sizes = {}
        for i, (_, h, w) in enumerate(crops):
            sizes.setdefault((h, w), []).append(i)
        for (h, w), idx in sizes.items():
            patches = mx.stack([crops[i][0] for i in idx])
            feats = self.multi_modal_projector(self.vision_tower(patches, h, w), h, w)
            for k, i in enumerate(idx):
                out[i] = feats[k]
        return mx.concatenate(out, axis=0)

    def embed_with_images(self, input_ids: mx.array, crops) -> mx.array:
        """Token embeddings [1, L, H] with the image tokens' rows replaced by
        the image features, in order."""
        embeds = self.language_model.model.embed_tokens(input_ids)
        features = self.image_features(crops)
        mask = (input_ids == self.args.image_token_id)[0]
        positions = mx.array([i for i, m in enumerate(mask.tolist()) if m], dtype=mx.int32)
        if positions.shape[0] != features.shape[0]:
            raise ValueError(f"{positions.shape[0]} image tokens for {features.shape[0]} image features")
        return embeds.at[0, positions].add(features.astype(embeds.dtype) - embeds[0, positions])

    def sanitize(self, weights):
        text, other = {}, {}
        for key, value in weights.items():
            # transformers 5 names (model.language_model.*, model.vision_tower.
            # vision_model.*, model.multi_modal_projector.*) or the older ones.
            key = key.removeprefix("model.")
            if key.startswith("lm_head"):
                continue
            if key.startswith("language_model."):
                key = key.removeprefix("language_model.").removeprefix("model.")
                text["model." + key] = value
                continue
            if self.vision_tower is None:
                continue
            key = key.replace("vision_tower.vision_model.", "vision_tower.", 1)
            if "position_ids" not in key:
                other[key] = value
        out = {"language_model." + k: v for k, v in self.language_model.sanitize(text).items()}
        out.update(other)
        return out

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        return self.language_model.make_cache()
