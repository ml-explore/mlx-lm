# Copyright © 2023-2024 Apple Inc.

import copy
import gc
import glob
import json
import re
import shutil
import struct
from collections import defaultdict
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .convert import MODEL_CONVERSION_DTYPES, mixed_quant_predicate_builder
from .tokenizer_utils import load as _load_tokenizer
from .utils import (
    MAX_FILE_SIZE_GB,
    _download,
    _get_classes,
    create_model_card,
    load_config,
    save_config,
    upload_to_hub,
)


@dataclass
class WeightChunk:
    layer_idx: Optional[int]
    weight_keys: List[str]
    source_files: Set[str]


def read_safetensors_header(filepath: str) -> dict:
    """Read tensor metadata from safetensors file header.

    Returns {tensor_name: {"shape": [...], "dtype": "...", "size_bytes": N}}
    """
    with open(filepath, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    header.pop("__metadata__", None)
    return {
        name: {
            "shape": meta["shape"],
            "dtype": meta["dtype"],
            "size_bytes": meta["data_offsets"][1] - meta["data_offsets"][0],
        }
        for name, meta in header.items()
    }


def scan_source_tensors(model_path: Path, weight_map: dict) -> dict:
    """Scan all source safetensors headers. Returns complete tensor catalog."""
    catalog = {}
    for source_file in sorted(set(weight_map.values())):
        header = read_safetensors_header(str(model_path / source_file))
        for name, meta in header.items():
            meta["source_file"] = source_file
            catalog[name] = meta
    return catalog


def plan_layer_chunks(weight_map: dict) -> List[WeightChunk]:
    """Group weight keys by transformer layer index."""
    layer_pattern = re.compile(r"(?:model\.layers|transformer\.blocks)\.(\d+)\.")
    layer_groups = defaultdict(list)
    non_layer = []

    for key in sorted(weight_map.keys()):
        m = layer_pattern.search(key)
        if m:
            layer_groups[int(m.group(1))].append(key)
        else:
            non_layer.append(key)

    chunks = []
    # Non-layer weights as a single chunk (embed, lm_head, norms)
    if non_layer:
        chunks.append(
            WeightChunk(
                layer_idx=None,
                weight_keys=non_layer,
                source_files={weight_map[k] for k in non_layer},
            )
        )
    # One chunk per layer, in order
    for idx in sorted(layer_groups):
        keys = layer_groups[idx]
        chunks.append(
            WeightChunk(
                layer_idx=idx,
                weight_keys=keys,
                source_files={weight_map[k] for k in keys},
            )
        )
    return chunks


def estimate_quantized_size(
    shape: list,
    bits: int = 4,
    group_size: int = 64,
    mode: str = "affine",
) -> int:
    """Compute expected byte size of a quantized tensor from shape and params.

    Handles affine (weight + scales + biases) and non-affine (weight + scales).
    """
    out_dim, in_dim = shape[0], shape[-1]
    # Quantized weight: packed into uint32
    # Each uint32 holds 32/bits values
    wq_bytes = out_dim * (in_dim * bits // 32) * 4
    # Scales: one per group, float32 (4 bytes) — mx.quantize returns float32
    n_groups = in_dim // group_size
    scales_bytes = out_dim * n_groups * 4
    # Biases: same shape as scales for affine mode (float32)
    if mode == "affine":
        biases_bytes = scales_bytes
    else:
        biases_bytes = 0
    return wq_bytes + scales_bytes + biases_bytes


def build_quant_plan(
    config: dict,
    q_bits: int = 4,
    q_group_size: int = 64,
    q_mode: str = "affine",
    quant_predicate=None,
    mixed_recipe: str = None,
    tensor_catalog: dict = None,
) -> Tuple[dict, dict]:
    """Pre-compute quantization parameters for every weight.

    Returns (plan, quant_config):
        plan: {weight_key: {"bits": N, "group_size": G, "mode": M} | False}
        quant_config: dict for config.json
    """
    model_class, model_args_class = _get_classes(config)
    model = model_class(model_args_class.from_dict(config))

    # Use model's own quant_predicate if available
    if quant_predicate is None:
        quant_predicate = getattr(model, "quant_predicate", None)

    # Build mixed recipe predicate if specified
    if isinstance(quant_predicate, str):
        quant_predicate = mixed_quant_predicate_builder(
            quant_predicate, model, q_group_size
        )

    default_params = {"group_size": q_group_size, "bits": q_bits, "mode": q_mode}
    plan = {}
    quant_config = dict(default_params)

    # Walk the module graph to find quantizable modules
    for path, module in model.named_modules():
        if not hasattr(module, "to_quantized"):
            continue

        weight_key = f"{path}.weight"
        if tensor_catalog and weight_key not in tensor_catalog:
            continue

        # Check shape divisibility
        if tensor_catalog:
            shape = tensor_catalog[weight_key]["shape"]
            if shape[-1] % q_group_size != 0:
                plan[weight_key] = False
                continue

        # Apply predicate
        params = default_params
        if quant_predicate is not None:
            result = quant_predicate(path, module)
            if result is False:
                plan[weight_key] = False
                continue
            elif isinstance(result, dict):
                params = result
                quant_config[path] = params

        plan[weight_key] = params

    del model
    return plan, quant_config


def validate_chunked_compatible(config: dict) -> None:
    """Check if model is compatible with chunked conversion."""
    model_type = config.get("model_type", "")

    # Vision-language models use tree_unflatten in sanitize — not supported
    vl_indicators = ["vision_config", "image_token_index", "vision_tower"]
    if any(k in config for k in vl_indicators):
        raise ValueError(
            f"Chunked conversion does not support vision-language models "
            f"(model_type={model_type}). Use standard conversion."
        )

    # Custom model_file not supported
    if config.get("model_file"):
        raise ValueError(
            "Chunked conversion does not support custom model files. "
            "Use standard conversion."
        )


def chunked_sanitize(model, chunk_weights, layer_idx):
    """Apply sanitize to a layer chunk."""
    if not hasattr(model, "sanitize"):
        return chunk_weights

    # Detect if chunk has expert keys needing stacking
    has_experts = any("experts." in k for k in chunk_weights)

    if has_experts and layer_idx is not None and layer_idx != 0:
        # Inject dummy layer-0 expert keys to bypass the top-level guard.
        sentinels = _make_layer0_sentinels(chunk_weights)
        for sentinel_key in sentinels:
            if sentinel_key not in chunk_weights:
                chunk_weights[sentinel_key] = mx.zeros((1,))
        result = model.sanitize(chunk_weights)
        # Remove any sentinel keys and any stacked layer-0 output
        for key in list(result.keys()):
            if key in sentinels or _is_layer0_key(key):
                result.pop(key)
        return result

    return model.sanitize(chunk_weights)


def _remap_key_to_layer0(key):
    """Replace the layer index in a key with 0."""
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            parts[i + 1] = "0"
            return ".".join(parts)
    return None


def _is_layer0_key(key):
    """Check if a key belongs to layer 0."""
    return ".layers.0." in key


def _make_layer0_sentinels(chunk_weights):
    """Build layer-0 versions of ALL expert keys in the chunk.

    MoE sanitize methods (Mixtral, Qwen2-MoE, PhiMoE) have a top-level guard
    like ``if "model.layers.0...experts.0.w1.weight" not in weights: return
    weights``. When processing layer N (N > 0) in isolation, this guard causes
    an early return, skipping expert stacking.

    After passing the guard, the sanitize iterates all layers and pops all
    expert keys for each layer it finds. We must inject dummy keys for ALL
    layer-0 experts (not just expert 0) so the pop operations succeed without
    KeyError. The stacked layer-0 output is removed after sanitize returns.
    """
    sentinels = set()
    for key in chunk_weights:
        if "experts." in key:
            layer0_key = _remap_key_to_layer0(key)
            if layer0_key:
                sentinels.add(layer0_key)
    return sentinels


def convert_chunked(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
    q_mode: str = "affine",
    dtype: Optional[str] = None,
    quant_predicate=None,
    upload_repo: str = None,
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
    dequantize: bool = False,
):
    """Memory-efficient chunked model conversion.

    Processes weights one layer at a time, never loading the full model
    into memory. Produces output identical to the standard conversion path.
    """
    if dequantize:
        raise ValueError("--chunked is not compatible with --dequantize")

    mlx_path = Path(mlx_path)
    if mlx_path.exists():
        raise ValueError(
            f"Cannot save to the path {mlx_path} as it already exists."
            " Please delete the file/directory or specify a new path to save to."
        )

    # Phase A: Metadata & Planning
    print("[INFO] Chunked conversion mode")
    model_path = _download(hf_path, revision=revision)
    config = load_config(model_path)

    # Validate compatibility (B4, M1, M5)
    validate_chunked_compatible(config)

    # Read source weight index
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        # Single-file model (M3) — check for single file
        single_file = model_path / "model.safetensors"
        if single_file.exists():
            print(
                "[INFO] Single-file model detected. --chunked is unnecessary "
                "for single-file models, falling back to standard conversion."
            )
            from .convert import convert

            return convert(
                hf_path,
                str(mlx_path),
                quantize,
                q_group_size,
                q_bits,
                q_mode,
                dtype,
                upload_repo,
                revision,
                False,
                quant_predicate,
                trust_remote_code,
            )
        raise FileNotFoundError(
            f"No model.safetensors.index.json or model.safetensors found in {model_path}"
        )

    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    # Scan headers for size estimation
    tensor_catalog = scan_source_tensors(model_path, weight_map)
    print(
        f"[INFO] Scanned {len(set(weight_map.values()))} source files, "
        f"found {len(tensor_catalog)} tensors"
    )

    # Instantiate lightweight model (no weights) for predicates and sanitize (D3)
    model_class, model_args_class = _get_classes(config)
    model = model_class(model_args_class.from_dict(config))
    cast_pred = getattr(model, "cast_predicate", lambda _: True)

    # Resolve dtype
    if dtype is None:
        dtype_str = config.get("torch_dtype", None)
    else:
        dtype_str = dtype
    target_dtype = None
    if dtype_str in MODEL_CONVERSION_DTYPES:
        print(f"[INFO] Using dtype: {dtype_str}")
        target_dtype = getattr(mx, dtype_str)

    # Build quant plan (B1)
    quant_plan, quant_config = {}, {}
    if quantize:
        quant_plan, quant_config = build_quant_plan(
            config,
            q_bits,
            q_group_size,
            q_mode,
            quant_predicate,
            None,
            tensor_catalog,
        )

    # Plan chunks
    chunks = plan_layer_chunks(weight_map)
    print(f"[INFO] Planned {len(chunks)} chunks")

    # Phase B: Stream process
    # Disable Metal buffer cache for aggressive memory reclamation.
    # set_cache_limit returns the previous limit; restored in finally.
    original_cache_limit = mx.set_cache_limit(0)
    mlx_path.mkdir(parents=True, exist_ok=True)

    output_weight_map = {}
    shard_weights = {}
    shard_bytes = 0
    shard_idx = 0
    total_size = 0
    total_params = 0

    try:
        for chunk_i, chunk in enumerate(chunks):
            layer_desc = (
                f"layer {chunk.layer_idx}"
                if chunk.layer_idx is not None
                else "non-layer"
            )
            print(f"[INFO] Processing chunk {chunk_i + 1}/{len(chunks)} ({layer_desc})")

            # Load source files for this chunk (lazy)
            loaded = {}
            for src_file in sorted(chunk.source_files):
                file_data = mx.load(str(model_path / src_file))
                for key in chunk.weight_keys:
                    if weight_map.get(key) == src_file and key in file_data:
                        loaded[key] = file_data.pop(key)
                del file_data

            # Sanitize (D4, B3, B4)
            loaded = chunked_sanitize(model, loaded, chunk.layer_idx)

            # Process each weight in the chunk
            for key in list(loaded.keys()):
                w = loaded.pop(key)

                # Dtype casting
                if (
                    target_dtype
                    and cast_pred(key)
                    and mx.issubdtype(w.dtype, mx.floating)
                ):
                    w = w.astype(target_dtype)

                # Quantize if applicable
                if key in quant_plan and quant_plan[key] is not False:
                    params = quant_plan[key]
                    bits = params["bits"]
                    gs = params["group_size"]
                    mode = params.get("mode", "affine")

                    if w.ndim >= 2 and w.shape[-1] % gs == 0:
                        result = mx.quantize(w, group_size=gs, bits=bits)
                        if mode == "affine":
                            qw, scales, biases = result
                            out = {
                                key: qw,
                                key.replace(".weight", ".scales"): scales,
                                key.replace(".weight", ".biases"): biases,
                            }
                        else:
                            qw, scales = result
                            out = {
                                key: qw,
                                key.replace(".weight", ".scales"): scales,
                            }

                        # Count params (M2): original unquantized param count
                        cat_entry = tensor_catalog.get(key)
                        if cat_entry:
                            total_params += prod(cat_entry["shape"])
                        else:
                            total_params += w.size
                    else:
                        out = {key: w}
                        total_params += w.size
                else:
                    out = {key: w}
                    total_params += w.size

                # Accumulate into shard
                for out_key, out_tensor in out.items():
                    mx.eval(out_tensor)
                    shard_weights[out_key] = out_tensor
                    shard_bytes += out_tensor.nbytes
                    total_size += out_tensor.nbytes

            # Flush shard if needed
            if shard_bytes >= MAX_FILE_SIZE_GB * (1 << 30):
                shard_name = f"model-{shard_idx + 1:05d}-of-XXXXX.safetensors"
                mx.save_safetensors(
                    str(mlx_path / shard_name),
                    shard_weights,
                    metadata={"format": "mlx"},
                )
                for k in shard_weights:
                    output_weight_map[k] = shard_name
                del shard_weights
                gc.collect()
                mx.clear_cache()
                shard_weights = {}
                shard_bytes = 0
                shard_idx += 1

        # Flush final shard
        if shard_weights:
            shard_name = f"model-{shard_idx + 1:05d}-of-XXXXX.safetensors"
            mx.save_safetensors(
                str(mlx_path / shard_name),
                shard_weights,
                metadata={"format": "mlx"},
            )
            for k in shard_weights:
                output_weight_map[k] = shard_name
            del shard_weights
            gc.collect()
            mx.clear_cache()
            shard_idx += 1
    finally:
        mx.set_cache_limit(original_cache_limit)

    # Phase C: Finalize

    # Rename shards with correct total count (m3)
    total_shards = shard_idx
    if total_shards == 1:
        old = mlx_path / "model-00001-of-XXXXX.safetensors"
        new = mlx_path / "model.safetensors"
        old.rename(new)
        output_weight_map = {k: "model.safetensors" for k in output_weight_map}
    else:
        rename_map = {}
        for i in range(total_shards):
            old_name = f"model-{i + 1:05d}-of-XXXXX.safetensors"
            new_name = f"model-{i + 1:05d}-of-{total_shards:05d}.safetensors"
            (mlx_path / old_name).rename(mlx_path / new_name)
            rename_map[old_name] = new_name
        output_weight_map = {
            k: rename_map.get(v, v) for k, v in output_weight_map.items()
        }

    # Write index
    index_data = {
        "metadata": {
            "total_size": total_size,
            "total_parameters": total_params,
        },
        "weight_map": dict(sorted(output_weight_map.items())),
    }
    with open(mlx_path / "model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=4)

    # Write config
    quantized_config = copy.deepcopy(config)
    if quantize:
        quantized_config["quantization"] = quant_config
        quantized_config["quantization_config"] = quant_config
    save_config(quantized_config, mlx_path / "config.json")

    # Copy tokenizer and auxiliary files
    tokenizer = _load_tokenizer(
        model_path,
        {"trust_remote_code": trust_remote_code},
    )
    tokenizer.save_pretrained(mlx_path)
    for p in ["*.py", "generation_config.json"]:
        for file_path in glob.glob(str(model_path / p)):
            shutil.copy(file_path, mlx_path)

    create_model_card(mlx_path, hf_path if not Path(hf_path).exists() else None)

    peak = mx.get_peak_memory() / (1 << 30)
    print(f"[INFO] Conversion complete. Peak memory: {peak:.1f} GB")

    if upload_repo is not None:
        upload_to_hub(str(mlx_path), upload_repo)
