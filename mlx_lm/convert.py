# Copyright © 2023-2024 Apple Inc.

import argparse
import glob
import json
from pathlib import Path
from typing import Callable, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

from .utils import (
    dequantize_model,
    hf_repo_to_path,
    load,
    make_shards,
    quantize_model,
    save,
    upload_to_hub,
)

# Weight-name substrings that mark multi-token-prediction ("next-n"/MTP)
# modules, curated from the sanitize methods that drop them: qwen3_5,
# qwen3_next, nemotron_h, exaone_moe (``mtp.``); mimo (``mtp_layers.``);
# mimo_v2_flash, kimi_linear, longcat_flash (``model.mtp``); step3p5
# (``.mtp``); and ernie4_5_moe (``mtp_block.`` / ``mtp_linear_proj.`` /
# ``mtp_hidden_norm.`` / ``mtp_emb_norm.``).
MTP_WEIGHT_PATTERNS = ("mtp.", ".mtp", "mtp_")

# Architectures that carry the MTP module as decoder layers past
# ``num_hidden_layers``, which sanitize drops by layer index. deepseek_v3's
# sanitize hardcodes layer 61 (== num_hidden_layers for DeepSeek-V3
# checkpoints), so the generic rule covers it.
MTP_TRAILING_LAYER_MODEL_TYPES = (
    "deepseek_v3",
    "deepseek_v32",
    "glm4_moe",
    "glm4_moe_lite",
    "step3p5",
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


QUANT_RECIPES = ["mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6"]

MODEL_CONVERSION_DTYPES = ["float16", "bfloat16", "float32"]


def _is_mtp_weight(
    name: str, model_type: Optional[str], num_dense_layers: Optional[int]
) -> bool:
    """Whether ``name`` is a multi-token-prediction weight dropped by sanitize."""
    if any(pattern in name for pattern in MTP_WEIGHT_PATTERNS):
        return True
    if model_type in MTP_TRAILING_LAYER_MODEL_TYPES and num_dense_layers is not None:
        parts = name.split(".")
        if "layers" in parts:
            i = parts.index("layers")
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                return int(parts[i + 1]) >= num_dense_layers
    return False


def _preserve_mtp_weights(mlx_path: Path, source_path: Path, config: dict) -> None:
    """Copy the source MTP weights that sanitize would drop next to the model.

    The multi-token-prediction weights are read from the raw source safetensors
    and written unmodified (source dtype, never quantized) to
    ``mtp*.safetensors`` with their own ``mtp.safetensors.index.json``;
    ``model.safetensors.index.json`` is untouched. ``mlx_lm.load`` reads only
    ``model*.safetensors``, so the converted model loads unchanged.
    """
    model_type = config.get("model_type")
    num_dense_layers = config.get("num_hidden_layers")
    if num_dense_layers is None:
        num_dense_layers = (config.get("text_config") or {}).get("num_hidden_layers")

    with open(mlx_path / "model.safetensors.index.json", "r") as f:
        saved = set(json.load(f)["weight_map"])

    preserved = {}
    for wf in sorted(glob.glob(str(source_path / "model*.safetensors"))):
        weights = mx.load(wf)
        for name, weight in weights.items():
            # Never shadow a weight the conversion already saved.
            if name not in saved and _is_mtp_weight(name, model_type, num_dense_layers):
                preserved[name] = weight

    if not preserved:
        print("[INFO] No MTP weights found to preserve.")
        return

    shards = make_shards(preserved)
    shard_file_format = (
        "mtp-{:05d}-of-{:05d}.safetensors" if len(shards) > 1 else "mtp.safetensors"
    )
    index = {
        "metadata": {
            "total_size": sum(v.nbytes for v in preserved.values()),
            "total_parameters": sum(v.size for v in preserved.values()),
        },
        "weight_map": {},
    }
    for i, shard in enumerate(shards):
        shard_name = shard_file_format.format(i + 1, len(shards))
        mx.save_safetensors(
            str(mlx_path / shard_name), shard, metadata={"format": "mlx"}
        )
        for name in shard:
            index["weight_map"][name] = shard_name

    index["weight_map"] = {
        k: index["weight_map"][k] for k in sorted(index["weight_map"])
    }
    with open(mlx_path / "mtp.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=4)

    print(f"[INFO] Preserved {len(preserved)} MTP weight(s) in {len(shards)} shard(s).")


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
    preserve_mtp: bool = False,
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
        trust_remote_code=trust_remote_code,
    )

    if isinstance(quant_predicate, str):
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

    if preserve_mtp:
        source_path = Path(hf_path)
        if not source_path.exists():
            source_path = hf_repo_to_path(hf_path, revision=revision)
        _preserve_mtp_weights(mlx_path, source_path, config)

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
    parser.add_argument(
        "--preserve-mtp",
        help="Also save the source multi-token-prediction (MTP) weights, which "
        "are otherwise dropped on load, so external MTP tooling can use them.",
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
