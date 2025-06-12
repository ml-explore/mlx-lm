# Copyright © 2025 Apple Inc.

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from ..models.bitlinear_layers import BitLinear

QUANT_LINEAR_MAPPING = {
    'bitnet': BitLinear,
}

def load_data(tokenizer, num_samples: int, sequence_length: int) -> mx.array:
    save_dir = Path.home() / ".cache/mlx-lm/calibration_v5.txt"
    if not save_dir.exists():
        from urllib import request

        save_dir.parent.mkdir(parents=True, exist_ok=True)
        url = "https://gist.githubusercontent.com/tristandruyen/9e207a95c7d75ddf37525d353e00659c/raw/571fda718462de863e5a0171078c175420c7649a/calibration_data_v5_rc.txt"
        request.urlretrieve(url, save_dir)
    with open(save_dir) as fid:
        texts = fid.read()
    tokens = tokenizer.encode(texts, return_tensors="mlx")[0]

    # select random non-overlapping chunks
    tokens = tokens[: (tokens.size // sequence_length) * sequence_length]
    tokens = tokens.reshape(-1, sequence_length)
    segments = mx.random.permutation(tokens.shape[0])
    if num_samples > 0:
        segments = segments[:num_samples]
    return tokens[segments]

def replace_linear_with_quant_linear(model, quant_method = "bitnet", modules_to_not_convert=None):
    quantize_layers = []
    for name, module in model.named_modules():
        if modules_to_not_convert is None:
            modules_to_not_convert = []

        # Replace nn.Linear layers, but skip 'lm_head'
        if name not in modules_to_not_convert and isinstance(module, nn.Linear):
            old_weight = module.weight
            out_features, in_features = old_weight.shape
            bias = "bias" in module
            # Create a new instance of the custom linear layer
            new_layer = QUANT_LINEAR_MAPPING[quant_method](in_features, out_features, bias=bias, invert_weight_scales=True)

            # Replace the layer in the model
            quantize_layers.append((name, new_layer))
    if len(quantize_layers) > 0:
        model.update_modules(tree_unflatten(quantize_layers))
    return model


def apply_hf_quantization(model, config):
    """
    Apply HF quantization to a model if it has a quantization config.
    """
    if config.get("quantization_config", None) is not None:
        quantization_config = config["quantization_config"]
        quant_method = quantization_config.get("quant_method", None)
        modules_to_not_convert = quantization_config.get("modules_to_not_convert", None)

        if quant_method is not None and quant_method in QUANT_LINEAR_MAPPING.keys():
            # Replace linear layers with quantized versions
            model = replace_linear_with_quant_linear(
                model,
                quant_method=quant_method,
                modules_to_not_convert=modules_to_not_convert
            )
    return model