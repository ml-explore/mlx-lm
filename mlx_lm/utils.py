# Copyright © 2023-2024 Apple Inc.

import copy
import glob
import importlib
import inspect
import json
import logging
import os
import resource
import shutil
from pathlib import Path
from textwrap import dedent
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import mlx.core as mx
import mlx.nn as nn

if os.getenv("MLXLM_USE_MODELSCOPE", "False").lower() == "true":
    try:
        from modelscope import snapshot_download
    except ImportError:
        raise ImportError("Run `pip install modelscope` to use ModelScope.")
else:
    from huggingface_hub import snapshot_download

# For large models with lots of files
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, 4096))

from mlx.utils import tree_flatten, tree_map, tree_reduce, tree_unflatten

# Local imports
from .tokenizer_utils import TokenizerWrapper
from .tokenizer_utils import load as _load_tokenizer

# Constants
MODEL_REMAPPING = {
    "mistral": "llama",
    "llava": "mistral3",
    "phi-msft": "phixtral",
    "falcon_mamba": "mamba",
    "kimi_k2": "deepseek_v3",
    "qwen2_5_vl": "qwen2_vl",
}

MAX_FILE_SIZE_GB = 5


def _get_classes(config: dict):
    """
    Retrieve the model and model args classes based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    model_type = config["model_type"]
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    try:
        arch = importlib.import_module(f"mlx_lm.models.{model_type}")
    except ImportError:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)

    return arch.Model, arch.ModelArgs


def get_total_parameters(model):
    leaf_modules = tree_flatten(
        model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)
    )

    def nparams(m):
        if hasattr(m, "bits"):
            n = 0 if not hasattr(m, "bias") else m.bias.size
            return n + m.weight.size * 32 // m.bits
        return sum(v.size for _, v in tree_flatten(m.parameters()))

    return sum(nparams(m) for _, m in leaf_modules)


def compute_bits_per_weight(model):
    model_bytes = tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
    )
    model_params = get_total_parameters(model)
    return model_bytes * 8 / model_params


def _download(
    path_or_hf_repo: str,
    revision: Optional[str] = None,
    allow_patterns: List[str] = None,
) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.
        revision (str, optional): A revision id which can be a branch name, a tag, or a commit hash.

    Returns:
        Path: The local file path.
    """
    model_path = Path(path_or_hf_repo)

    if not model_path.exists():
        allow_patterns = allow_patterns or [
            "*.json",
            "model*.safetensors",
            "*.py",
            "tokenizer.model",
            "*.tiktoken",
            "tiktoken.model",
            "*.txt",
            "*.jsonl",
            "*.jinja",
        ]
        model_path = Path(
            snapshot_download(
                path_or_hf_repo,
                revision=revision,
                allow_patterns=allow_patterns,
            )
        )

    return model_path


def hf_repo_to_path(hf_repo):
    return Path(snapshot_download(hf_repo, local_files_only=True))


def load_config(model_path: Path) -> dict:
    try:
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found in {model_path}")
        raise
    return config


def load_model(
    model_path: Path,
    lazy: bool = False,
    strict: bool = True,
    model_config: Optional[Dict[str, Any]] = None,
    get_model_classes: Callable[[dict], Tuple[Type[nn.Module], Type]] = _get_classes,
) -> Tuple[nn.Module, dict]:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        strict (bool): Whether or not to raise an exception if weights don't
            match. Default: ``True``
        model_config (dict, optional): Optional configuration parameters for the
            model. Defaults to an empty dictionary.
        get_model_classes (Callable[[dict], Tuple[Type[nn.Module], Type]], optional):
            A function that returns the model class and model args class given a config.
            Defaults to the ``_get_classes`` function.

    Returns:
        Tuple[nn.Module, dict[str, Any]]: The loaded and initialized model and config.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """
    config = load_config(model_path)
    if model_config is not None:
        config.update(model_config)

    weight_files = glob.glob(str(model_path / "model*.safetensors"))

    if not weight_files and strict:
        logging.error(f"No safetensors found in {model_path}")
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    model_class, model_args_class = get_model_classes(config=config)

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    def _quantize(quantization):
        def class_predicate(p, m):
            # Handle custom per layer quantizations
            if p in config["quantization"]:
                return config["quantization"][p]
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=class_predicate,
        )

    if (quantization := config.get("quantization", None)) is not None:
        _quantize(quantization)

    elif quantization_config := config.get("quantization_config", False):
        # Handle legacy quantization config
        quant_method = quantization_config["quant_method"]
        if quant_method == "bitnet":
            from .models.bitlinear_layers import bitnet_quantize

            model = bitnet_quantize(model, quantization_config)
        elif quant_method == "mxfp4":
            quantization = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
            config["quantization"] = quantization
            config["quantization_config"] = quantization
            _quantize(quantization)
        elif quant_method == "compressed-tensors":
            quantization = {"group_size": 32, "bits": 4, "mode": "affine"}
            config["quantization"] = quantization
            config["quantization_config"] = quantization
            _quantize(quantization)

    model.load_weights(list(weights.items()), strict=strict)

    if not lazy:
        mx.eval(model.parameters())

    model.eval()
    return model, config


def load_adapters(model: nn.Module, adapter_path: str) -> nn.Module:
    from .tuner.utils import load_adapters as _load_adapters

    return _load_adapters(model, adapter_path)


def load_tokenizer(model_path, tokenizer_config_extra=None, eos_token_ids=None):
    """Load a huggingface tokenizer and try to infer the type of streaming
    detokenizer to use.
    """
    model_path = _download(
        model_path,
        allow_patterns=[
            "*.json",
            "*.py",
            "tokenizer.model",
            "*.tiktoken",
            "tiktoken.model",
            "*.txt",
            "*.jsonl",
            "*.jinja",
        ],
    )
    return _load_tokenizer(
        model_path, tokenizer_config_extra, eos_token_ids=eos_token_ids
    )


def load(
    path_or_hf_repo: str,
    tokenizer_config: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    adapter_path: Optional[str] = None,
    lazy: bool = False,
    return_config: bool = False,
    revision: Optional[str] = None,
) -> Union[
    Tuple[nn.Module, TokenizerWrapper],
    Tuple[nn.Module, TokenizerWrapper, Dict[str, Any]],
]:
    """
    Load the model and tokenizer from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (Path): The path or the huggingface repository to load the model from.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer.
            Defaults to an empty dictionary.
        model_config(dict, optional): Configuration parameters specifically for the model.
            Defaults to an empty dictionary.
        adapter_path (str, optional): Path to the LoRA adapters. If provided, applies LoRA layers
            to the model. Default: ``None``.
        lazy (bool): If ``False`` eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        return_config (bool: If ``True`` return the model config as the last item..
        revision (str, optional): A revision id which can be a branch name, a tag, or a commit hash.
    Returns:
        Union[Tuple[nn.Module, TokenizerWrapper], Tuple[nn.Module, TokenizerWrapper, Dict[str, Any]]]:
            A tuple containing the loaded model, tokenizer and, if requested, the model config.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    model_path = _download(path_or_hf_repo, revision=revision)

    model, config = load_model(model_path, lazy, model_config=model_config)
    if adapter_path is not None:
        model = load_adapters(model, adapter_path)
        model.eval()
    tokenizer = load_tokenizer(
        model_path, tokenizer_config, eos_token_ids=config.get("eos_token_id", None)
    )

    if return_config:
        return model, tokenizer, config
    else:
        return model, tokenizer


def pipeline_load(repo, return_config=False):
    # Get model path with everything but weight safetensors
    model_path = _download(
        repo,
        allow_patterns=[
            "*.json",
            "*.py",
            "tokenizer.model",
            "*.tiktoken",
            "tiktoken.model",
            "*.txt",
            "*.jsonl",
            "*.jinja",
        ],
    )

    # Lazy load and shard model to figure out which weights we need
    model, config = load_model(model_path, lazy=True, strict=False)

    group = mx.distributed.init()
    rank = group.rank()
    model.model.pipeline(group)

    # Figure out which files we need for the local shard
    with open(model_path / "model.safetensors.index.json", "r") as fid:
        weight_index = json.load(fid)["weight_map"]

    local_files = set()
    for k, _ in tree_flatten(model.parameters()):
        if file_name := weight_index.get(k, None) is None:
            raise ValueError(
                "Pipeline loading is only supported for MLX converted models."
            )
        local_files.add(weight_index[k])

    # Download weights for local shard
    _download(repo, allow_patterns=local_files)

    # Load and shard the model, and load the weights
    tokenizer = load_tokenizer(
        model_path,
        {"trust_remote_code": True},
        eos_token_ids=config.get("eos_token_id", None),
    )
    model, _ = load_model(model_path, lazy=True, strict=False)
    model.model.pipeline(group)
    mx.eval(model.parameters())

    # Synchronize processes to avoid timeout
    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
    if return_config:
        return model, tokenizer, config
    else:
        return model, tokenizer


def make_shards(weights: dict, max_file_size_gb: int = MAX_FILE_SIZE_GB) -> list:
    """
    Splits the weights into smaller shards.

    Args:
        weights (dict): Model weights.
        max_file_size_gb (int): Maximum size of each shard in gigabytes.

    Returns:
        list: List of weight shards.
    """
    max_file_size_bytes = max_file_size_gb << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


def create_model_card(path: Union[str, Path], hf_path: Union[str, Path, None]):
    """
    Uploads the model to Hugging Face hub.

    Args:
        path (Union[str, Path]): Local path to the model.
        hf_path (Union[str, Path, None]): Path to the original Hugging Face model.
    """
    from huggingface_hub import ModelCard, ModelCardData

    if hf_path is None:
        card = ModelCard.from_template(ModelCardData(language="en"))
    else:
        card = ModelCard.load(hf_path)
    card.data.library_name = "mlx"
    card.data.pipeline_tag = "text-generation"
    if card.data.tags is None:
        card.data.tags = ["mlx"]
    elif "mlx" not in card.data.tags:
        card.data.tags += ["mlx"]
    if hf_path is not None:
        card.data.base_model = str(hf_path)
    card.text = ""
    card.save(os.path.join(path, "README.md"))


def upload_to_hub(path: str, upload_repo: str):
    """
    Uploads the model to Hugging Face hub.

    Args:
        path (str): Local path to the model.
        upload_repo (str): Name of the HF repo to upload to.
    """
    from huggingface_hub import HfApi, ModelCard, logging

    from . import __version__

    logging.set_verbosity_info()
    card_path = Path(path) / "README.md"
    card = ModelCard.load(card_path)

    hf_path = card.data.base_model

    if hf_path is not None:
        provenance = f"""
        This model [{upload_repo}](https://huggingface.co/{upload_repo}) was
        converted to MLX format from [{hf_path}](https://huggingface.co/{hf_path})
        using mlx-lm version **{__version__}**.
        """
    else:
        provenance = ""

    card.text = dedent(
        f"""
        # {upload_repo}
        {provenance}
        ## Use with mlx

        ```bash
        pip install mlx-lm
        ```

        ```python
        from mlx_lm import load, generate

        model, tokenizer = load("{upload_repo}")

        prompt = "hello"

        if tokenizer.chat_template is not None:
            messages = [{{"role": "user", "content": prompt}}]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_dict=False,
            )

        response = generate(model, tokenizer, prompt=prompt, verbose=True)
        ```
        """
    )
    card.save(card_path)

    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_large_folder(
        folder_path=path,
        repo_id=upload_repo,
        repo_type="model",
    )
    print(f"Upload successful, go to https://huggingface.co/{upload_repo} for details.")


def save_model(
    save_path: Union[str, Path],
    model: nn.Module,
    *,
    donate_model: bool = False,
) -> None:
    """Save model weights and metadata index into specified directory."""
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    weights = dict(tree_flatten(model.parameters()))
    shards = make_shards(weights)
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )

    total_size = sum(v.nbytes for v in weights.values())
    index_data = {
        "metadata": {
            "total_size": total_size,
            "total_parameters": get_total_parameters(model),
        },
        "weight_map": {},
    }
    if donate_model:
        model.update(tree_map(lambda _: mx.array([]), model.parameters()))

    # Write the weights and make sure no references are kept other than the
    # necessary ones
    weights.clear()
    del weights

    for i in range(len(shards)):
        shard = shards[i]
        shards[i] = None
        shard_name = shard_file_format.format(i + 1, shards_count)
        shard_path = save_path / shard_name

        mx.save_safetensors(str(shard_path), shard, metadata={"format": "mlx"})

        for weight_name in shard.keys():
            index_data["weight_map"][weight_name] = shard_name
        del shard

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(
            index_data,
            f,
            indent=4,
        )


def quantize_model(
    model: nn.Module,
    config: dict,
    group_size: int,
    bits: int,
    mode: str = "affine",
    quant_predicate: Optional[Callable[[str, nn.Module], Union[bool, dict]]] = None,
) -> Tuple[nn.Module, dict]:
    """
    Applies quantization to the model weights.

    Args:
        model (nn.Module): The model to be quantized.
        config (dict): Model configuration.
        group_size (int): Group size for quantization.
        bits (int): Bits per weight for quantization.
        mode (str): The quantization mode.
        quant_predicate (Callable): A callable that decides how to quantize
          each layer based on the path. Accepts the layer `path` and the
          `module`. Returns either a bool to signify quantize/no quantize or
          a dict of quantization parameters to pass to `to_quantized`.

    Returns:
        Tuple: Tuple containing quantized model and config.
    """
    quantized_config = copy.deepcopy(config)

    quant_predicate = quant_predicate or getattr(model, "quant_predicate", None)
    quant_params = {"group_size": group_size, "bits": bits, "mode": mode}
    if "quantization" in quantized_config:
        # If the model is already partially quantized, return params so that
        # the config is set on a per-layer basis
        fine_grained_config = True
    else:
        fine_grained_config = False
        quantized_config["quantization"] = quant_params

    def wrapped_predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if module.weight.shape[-1] % group_size != 0:
            return False
        bool_or_params = True
        if quant_predicate is not None:
            bool_or_params = quant_predicate(path, module)
        if isinstance(bool_or_params, dict):
            quantized_config["quantization"][path] = bool_or_params
        elif fine_grained_config and bool_or_params:
            quantized_config["quantization"][path] = quant_params
        return bool_or_params

    nn.quantize(
        model,
        group_size,
        bits,
        mode=mode,
        class_predicate=wrapped_predicate,
    )
    # support hf model tree #957
    quantized_config["quantization_config"] = quantized_config["quantization"]

    bpw = compute_bits_per_weight(model)
    print(f"[INFO] Quantized model with {bpw:.3f} bits per weight.")

    return model, quantized_config


def dequantize_model(model: nn.Module) -> nn.Module:
    """
    Dequantize the quantized layers in the model.

    Args:
        model (nn.Module): The model with quantized layers.

    Returns:
        nn.Module: The model with dequantized layers.
    """
    from .models.switch_layers import QuantizedSwitchLinear, SwitchLinear

    dequantize_layers = []
    for name, module in model.named_modules():
        bias = "bias" in module
        if isinstance(module, nn.QuantizedLinear):
            cls = nn.Linear
            kwargs = {"bias": bias}
        elif isinstance(module, nn.QuantizedEmbedding):
            kwargs = {}
            cls = nn.Embedding
        elif isinstance(module, QuantizedSwitchLinear):
            kwargs = {"bias": bias}
            cls = SwitchLinear
        else:
            continue
        weight = mx.dequantize(
            module.weight,
            module.scales,
            module.biases,
            module.group_size,
            module.bits,
            module.mode,
        )
        args = weight.shape[::-1]
        m = cls(*args, **kwargs)
        if bias:
            m.bias = module.bias
        m.weight = weight
        dequantize_layers.append((name, m))

    if len(dequantize_layers) > 0:
        model.update_modules(tree_unflatten(dequantize_layers))
    return model


def save_config(
    config: dict,
    config_path: Union[str, Path],
) -> None:
    """Save the model configuration to the ``config_path``.

    The final configuration will be sorted before saving for better readability.

    Args:
        config (dict): The model configuration.
        config_path (Union[str, Path]): Model configuration file path.
    """
    # Clean unused keys
    config.pop("_name_or_path", None)
    config.pop("vision_config", None)
    if "quantization" in config:
        config["quantization_config"] = config["quantization"]

    # sort the config for better readability
    config = dict(sorted(config.items()))

    # write the updated config to the config_path (if provided)
    with open(config_path, "w") as fid:
        json.dump(config, fid, indent=4)


def save(
    dst_path: Union[str, Path],
    src_path_or_repo: Union[str, Path],
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    config: Dict[str, Any],
    donate_model: bool = True,
):

    src_path = Path(src_path_or_repo)
    if not src_path.exists():
        hf_repo = src_path_or_repo
        src_path = hf_repo_to_path(hf_repo)
    else:
        hf_repo = None

    dst_path = Path(dst_path)
    save_model(dst_path, model, donate_model=True)
    save_config(config, config_path=dst_path / "config.json")
    tokenizer.save_pretrained(dst_path)

    for p in ["*.py", "generation_config.json"]:
        for file in glob.glob(str(src_path / p)):
            shutil.copy(file, dst_path)

    create_model_card(dst_path, hf_repo)


def common_prefix_len(list1, list2):
    """
    Calculates the length of the common prefix of two lists.

    Args:
        list1: The first list of strings.
        list2: The second list of strings.

    Returns:
        The length of the common prefix. Returns 0 if lists are empty
        or do not match at the first element.
    """
    # Determine the maximum possible length of the common prefix
    min_len = min(len(list1), len(list2))

    # Iterate up to the length of the shorter list
    for i in range(min_len):
        if list1[i] != list2[i]:
            # Mismatch found, the common prefix length is the current index
            return i

    # No mismatch found within the bounds of the shorter list,
    # so the common prefix length is the length of the shorter list.
    return min_len


def does_model_support_input_embeddings(model: nn.Module) -> bool:
    """
    Check if the model supports input_embeddings in its call signature.
    Args:
        model (nn.Module): The model to check.
    Returns:
        bool: True if the model supports input_embeddings, False otherwise.
    """
    try:
        signature = inspect.signature(model.__call__)
        return "input_embeddings" in signature.parameters
    except (ValueError, TypeError):
        return False
