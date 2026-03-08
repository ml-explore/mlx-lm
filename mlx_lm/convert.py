# Copyright © 2023-2024 Apple Inc.

import argparse
from pathlib import Path
from typing import Callable, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

from .utils import (
    dequantize_model,
    load,
    quantize_model,
    save,
    upload_to_hub,
)


def mixed_quant_predicate_builder(
    recipe: str, model: nn.Module, group_size: int = 64
) -> Callable[[str, nn.Module, dict], Union[bool, dict]]:
    mode = "affine"
    high_bits = 6

    if recipe == "mixed_2_6":
        low_bits = 2
    elif recipe == "mixed_3_4":
        low_bits = 3
        high_bits = 4
    elif recipe == "mixed_3_6":
        low_bits = 3
    elif recipe == "mixed_4_6":
        low_bits = 4
    else:
        raise ValueError(f"Invalid quant recipe {recipe}")

    down_keys = [k for k, _ in model.named_modules() if "down_proj" in k]
    if len(down_keys) == 0:
        raise ValueError("Model does not have expected keys for mixed quant.")

    # Look for the layer index location in the path:
    for layer_location, k in enumerate(down_keys[0].split(".")):
        if k.isdigit():
            break
    num_layers = len(model.layers)

    def mixed_quant_predicate(
        path: str,
        module: nn.Module,
    ) -> Union[bool, dict]:
        """Implements mixed quantization predicates with similar choices to, for example, llama.cpp's Q4_K_M.
        Ref: https://github.com/ggerganov/llama.cpp/blob/917786f43d0f29b7c77a0c56767c0fa4df68b1c5/src/llama.cpp#L5265
        By Alex Barron: https://gist.github.com/barronalex/84addb8078be21969f1690c1454855f3
        """
        index = (
            int(path.split(".")[layer_location])
            if len(path.split(".")) > layer_location
            else 0
        )
        use_more_bits = (
            index < num_layers // 8
            or index >= 7 * num_layers // 8
            or (index - num_layers // 8) % 3 == 2
        )
        if (
            "v_proj" in path or "v_a_proj" in path or "v_b_proj" in path
        ) and use_more_bits:
            return {"group_size": group_size, "bits": high_bits, "mode": mode}
        if "down_proj" in path and use_more_bits:
            return {"group_size": group_size, "bits": high_bits, "mode": mode}
        if "lm_head" in path:
            return {"group_size": group_size, "bits": high_bits, "mode": mode}

        return {"group_size": group_size, "bits": low_bits, "mode": mode}

    return mixed_quant_predicate


def jamba_int8_safe_predicate_builder(
    group_size: Optional[int] = None,
) -> Callable[[str, nn.Module], Union[bool, dict]]:
    mode = "affine"
    bits = 8
    resolved_group_size = group_size if group_size is not None else 64
    direct_feed_forward_modules = {"gate_proj", "up_proj", "down_proj"}
    switch_mlp_modules = {"gate_proj", "up_proj", "down_proj"}

    def jamba_int8_safe_predicate(path: str, module: nn.Module) -> Union[bool, dict]:
        parts = path.split(".")

        is_direct_feed_forward = (
            len(parts) == 5
            and parts[0] == "model"
            and parts[1] == "layers"
            and parts[2].isdigit()
            and parts[3] == "feed_forward"
            and parts[4] in direct_feed_forward_modules
        )

        is_switch_mlp = (
            len(parts) == 6
            and parts[0] == "model"
            and parts[1] == "layers"
            and parts[2].isdigit()
            and parts[3] == "feed_forward"
            and parts[4] == "switch_mlp"
            and parts[5] in switch_mlp_modules
        )

        if is_direct_feed_forward or is_switch_mlp:
            return {
                "group_size": resolved_group_size,
                "bits": bits,
                "mode": mode,
            }

        return False

    return jamba_int8_safe_predicate


QUANT_RECIPES = ["mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6", "jamba_int8_safe"]

MODEL_CONVERSION_DTYPES = ["float16", "bfloat16", "float32"]


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_group_size: Optional[int] = None,
    q_bits: Optional[int] = None,
    q_mode: str = "affine",
    dtype: Optional[str] = None,
    upload_repo: str = None,
    revision: Optional[str] = None,
    dequantize: bool = False,
    quant_predicate: Optional[
        Union[Callable[[str, nn.Module, dict], Union[bool, dict]], str]
    ] = None,
    trust_remote_code: bool = False,
):
    # Check the save path is empty
    if isinstance(mlx_path, str):
        mlx_path = Path(mlx_path)

    if mlx_path.exists():
        raise ValueError(
            f"Cannot save to the path {mlx_path} as it already exists."
            " Please delete the file/directory or specify a new path to save to."
        )

    print("[INFO] Loading")
    model, tokenizer, config = load(
        hf_path,
        revision=revision,
        return_config=True,
        tokenizer_config={"trust_remote_code": trust_remote_code},
        lazy=True,
    )

    if isinstance(quant_predicate, str):
        if quant_predicate == "jamba_int8_safe":
            model_type = (
                config.get("model_type")
                or getattr(model, "model_type", None)
                or getattr(getattr(model, "config", None), "model_type", None)
            )
            if model_type != "jamba":
                raise ValueError(
                    "The 'jamba_int8_safe' quantization recipe is only compatible "
                    f"with Jamba models (model_type='jamba'). Got model_type={model_type!r}."
                )
            if q_mode != "affine":
                raise ValueError(
                    "The 'jamba_int8_safe' quantization recipe requires "
                    "--q-mode affine."
                )
            if q_bits is not None and q_bits != 8:
                raise ValueError(
                    "The 'jamba_int8_safe' quantization recipe uses 8-bit "
                    "quantization for recipe-selected layers. Please use --q-bits 8 "
                    "or leave --q-bits unset."
                )
            quant_predicate = jamba_int8_safe_predicate_builder(q_group_size)
        else:
            if q_mode != "affine":
                raise ValueError(f"Quant predicates only support 'affine' quantization.")
            quant_predicate = mixed_quant_predicate_builder(
                quant_predicate,
                model,
                q_group_size,
            )

    if dtype is None:
        dtype = config.get("torch_dtype", None)
    if dtype is None and (text_config := config.get("text_config", None)):
        dtype = text_config.get("dtype", None)
    if dtype in MODEL_CONVERSION_DTYPES:
        print("[INFO] Using dtype:", dtype)
        dtype = getattr(mx, dtype)
        cast_predicate = getattr(model, "cast_predicate", lambda _: True)

        def set_dtype(k, v):
            if cast_predicate(k) and mx.issubdtype(v.dtype, mx.floating):
                return v.astype(dtype)
            else:
                return v

        model.update(tree_map_with_path(set_dtype, model.parameters()))

    if quantize and dequantize:
        raise ValueError("Choose either quantize or dequantize, not both.")

    if quantize:
        print("[INFO] Quantizing")
        model, config = quantize_model(
            model,
            config,
            q_group_size,
            q_bits,
            mode=q_mode,
            quant_predicate=quant_predicate,
        )

    if dequantize:
        print("[INFO] Dequantizing")
        config.pop("quantization", None)
        config.pop("quantization_config", None)
        model = dequantize_model(model)

    save(
        mlx_path,
        hf_path,
        model,
        tokenizer,
        config,
    )

    if upload_repo is not None:
        upload_to_hub(mlx_path, upload_repo)


def configure_parser() -> argparse.ArgumentParser:
    """
    Configures and returns the argument parser for the script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face model to MLX format"
    )

    parser.add_argument(
        "--hf-path",
        "--model",
        type=str,
        help="Path to the model. This can be a local path or a Hugging Face Hub model identifier.",
    )
    parser.add_argument(
        "--mlx-path", type=str, default="mlx_model", help="Path to save the MLX model."
    )
    parser.add_argument(
        "-q", "--quantize", help="Generate a quantized model.", action="store_true"
    )
    parser.add_argument(
        "--q-group-size",
        help="Group size for quantization.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--q-bits",
        help="Bits per weight for quantization.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--q-mode",
        help="The quantization mode.",
        type=str,
        default="affine",
        choices=["affine", "mxfp4", "nvfp4", "mxfp8"],
    )
    parser.add_argument(
        "--quant-predicate",
        help=f"Mixed-bit quantization recipe.",
        choices=QUANT_RECIPES,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--dtype",
        help="Type to save the non-quantized parameters. Defaults to config.json's `torch_dtype` or the current model weights dtype.",
        type=str,
        choices=MODEL_CONVERSION_DTYPES,
        default=None,
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-d",
        "--dequantize",
        help="Dequantize a quantized model.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--trust-remote-code",
        help="Trust remote code when loading tokenizer.",
        action="store_true",
        default=False,
    )
    return parser


def main():
    parser = configure_parser()
    args = parser.parse_args()
    convert(**vars(args))


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.convert ...` directly is deprecated."
        " Use `mlx_lm.convert ...` or `python -m mlx_lm convert ...` instead."
    )
    main()
