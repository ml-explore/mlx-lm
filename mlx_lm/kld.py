"""
Evaluate KL divergence of MLX models against a cached baseline.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

from mlx_lm.utils import _download, load, load_eval_tokens

CACHE_DIR = Path("kld_cache")
CACHE_KEY_FIELDS = (
    "baseline_model",
    "top_k",
    "data_path",
    "sequence_length",
    "num_samples",
    "seed",
    "batch_size",
)


class BaselineCache:
    def __init__(self, path: Path, manifest: Dict):
        self.path = path
        self.manifest = manifest

    @classmethod
    def open(cls, cache_dir: Path):
        cache_dir = cache_dir.expanduser()
        if not cache_dir.exists():
            raise ValueError(f"Baseline cache does not exist: {cache_dir}")
        cache_dir = cache_dir.resolve()
        cache = cls(cache_dir, load_manifest(cache_dir))
        _log(f"Using baseline cache at {cache.path}")
        return cache

    def token_path(self) -> Path:
        return self.path / "tokens.safetensors"

    def batch_path(self, batch_idx: int) -> Path:
        return self.path / f"baseline_{batch_idx:06d}.safetensors"

    def save_tokens(self, tokens) -> None:
        mx.save_safetensors(str(self.token_path()), {"tokens": tokens})

    def load_tokens(self):
        return mx.load(str(self.token_path()))["tokens"]

    def save_batch(self, batch_idx: int, indices, logprobs, tail_mass) -> None:
        mx.save_safetensors(
            str(self.batch_path(batch_idx)),
            {"indices": indices, "logprobs": logprobs, "tail_mass": tail_mass},
        )

    def load_batch(self, batch_idx: int):
        return mx.load(str(self.batch_path(batch_idx)))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate KL divergence against a cached baseline"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to candidate model or Hugging Face model",
    )
    baseline_group = parser.add_mutually_exclusive_group(required=True)
    baseline_group.add_argument(
        "--baseline-model",
        type=str,
        help="Path to baseline model or Hugging Face model",
    )
    baseline_group.add_argument(
        "--baseline-cache",
        type=str,
        help="Existing baseline cache directory to reuse",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1024,
        help="Number of baseline tokens to cache per position",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="allenai/tulu-3-sft-mixture",
        help="Local dataset directory or Hugging Face dataset",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=512,
        help="Sequence length for evaluation",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=256,
        help="Number of samples to use (-1 for all available)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for cache build and evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for dataset sampling",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    return parser


def canonicalize_ref(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())
    return value


def slugify_name(name: str) -> str:
    tail = name.replace("\\", "/").rstrip("/").split("/")[-1]
    slug = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in tail.strip())
    return slug.strip("._") or "baseline"


def build_manifest(
    args,
    *,
    vocab_size: Optional[int] = None,
    vocab_hash: Optional[str] = None,
) -> Dict:
    manifest = {name: getattr(args, name) for name in CACHE_KEY_FIELDS}
    manifest["vocab_size"] = vocab_size
    manifest["vocab_hash"] = vocab_hash
    return manifest


def load_manifest(cache_dir: Path) -> Dict:
    try:
        with open(cache_dir / "manifest.json", "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise _invalid_cache(cache_dir) from exc

    if not isinstance(data, dict):
        raise _invalid_cache(cache_dir)

    try:
        manifest = {
            "baseline_model": data["baseline_model"],
            "top_k": int(data["top_k"]),
            "data_path": data["data_path"],
            "sequence_length": int(data["sequence_length"]),
            "num_samples": int(data["num_samples"]),
            "seed": int(data["seed"]),
            "batch_size": int(data["batch_size"]),
            "vocab_size": (
                None if data.get("vocab_size") is None else int(data["vocab_size"])
            ),
            "vocab_hash": data.get("vocab_hash"),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise _invalid_cache(cache_dir) from exc

    validate_top_k(manifest["top_k"], manifest["vocab_size"])
    return manifest


def write_manifest(cache_dir: Path, manifest: Dict) -> None:
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def derive_cache_dir(manifest: Dict) -> Path:
    payload = {name: manifest[name] for name in CACHE_KEY_FIELDS}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:8]
    slug = slugify_name(manifest["baseline_model"])
    return CACHE_DIR / f"{slug}-{digest}"


def load_model_or_raise(
    model_ref: str,
    role: str,
    *,
    trust_remote_code: bool,
    lazy: bool = False,
):
    _log(f"Loading {role} model from {model_ref}...")
    tokenizer_config = {"trust_remote_code": True if trust_remote_code else None}
    try:
        return load(model_ref, lazy=lazy, tokenizer_config=tokenizer_config)
    except FileNotFoundError as exc:
        if "No safetensors found in" not in str(exc):
            raise
        raise ValueError(
            f"Failed to load {role} model from {model_ref}. "
            "kld requires an MLX model directory or MLX-compatible Hugging Face repo "
            "with model*.safetensors weights."
        ) from exc


def load_candidate_model(args, manifest: Dict):
    model, tokenizer = load_model_or_raise(
        args.model,
        "candidate",
        trust_remote_code=args.trust_remote_code,
    )
    _validate_candidate_tokenizer(tokenizer, manifest)
    return model


def build_baseline_cache(args, cache_dir: Path) -> BaselineCache:
    validate_top_k(args.top_k)
    model, tokenizer = load_model_or_raise(
        args.baseline_model,
        "baseline",
        trust_remote_code=args.trust_remote_code,
        lazy=True,
    )

    vocab_size = _get_vocab_size(tokenizer)
    validate_top_k(args.top_k, vocab_size)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = BaselineCache(cache_dir.resolve(), build_manifest(args))

    _log("Loading dataset...")
    _log(f"  Sequence length: {args.sequence_length}")
    tokens = load_eval_tokens(
        tokenizer,
        args.data_path,
        args.num_samples,
        args.sequence_length,
        seed=args.seed,
    )
    mx.eval(tokens)
    cache.save_tokens(tokens)
    _log(f"  Loaded {len(tokens)} samples")

    _log(f"Building baseline cache at {cache.path}...")
    num_batches = math.ceil(len(tokens) / args.batch_size)
    for batch_idx, batch in enumerate(iter_batches(tokens, args.batch_size)):
        baseline_logprobs = nn.log_softmax(model(batch[:, :-1]).astype(mx.float32))
        mx.eval(baseline_logprobs)
        vocab_size = vocab_size or baseline_logprobs.shape[-1]
        validate_top_k(args.top_k, baseline_logprobs.shape[-1])
        indices, top_logprobs, tail_mass = cache_topk_batch(
            baseline_logprobs, args.top_k
        )
        mx.eval(indices, top_logprobs, tail_mass)
        cache.save_batch(batch_idx, indices, top_logprobs, tail_mass)
        _log_batch_progress("Cached", batch_idx + 1, num_batches)

    cache.manifest = build_manifest(
        args,
        vocab_size=vocab_size,
        vocab_hash=_get_vocab_hash(tokenizer),
    )
    write_manifest(cache.path, cache.manifest)

    del model, tokenizer, tokens
    mx.clear_cache()
    return cache


def resolve_cache(args) -> BaselineCache:
    if args.baseline_cache:
        return BaselineCache.open(Path(args.baseline_cache))

    manifest = build_manifest(args)
    cache_dir = derive_cache_dir(manifest)
    if cache_dir.exists():
        return BaselineCache.open(cache_dir)

    model_path = _download(args.baseline_model)
    weight_file = next(model_path.glob("model*.safetensors"), None)
    metadata = (
        {}
        if weight_file is None
        else safe_open(str(weight_file), framework="np").metadata() or {}
    )
    if metadata.get("format") != "mlx":
        raise ValueError(
            f"Failed to load baseline model from {args.baseline_model}. "
            "kld requires MLX-converted weights saved by mlx-lm "
            "(expected safetensors metadata format='mlx')."
        )

    return build_baseline_cache(args, cache_dir)


def evaluate_kld(args, cache: BaselineCache) -> dict:
    model = load_candidate_model(args, cache.manifest)
    tokens = cache.load_tokens()
    batch_size = cache.manifest["batch_size"]
    num_batches = math.ceil(len(tokens) / batch_size)
    all_kl = []
    _log(f"Evaluating KL with batch size {batch_size}...")

    for batch_idx, batch in enumerate(iter_batches(tokens, batch_size)):
        model_logprobs = nn.log_softmax(model(batch[:, :-1]).astype(mx.float32))
        kl = kl_from_cached_batch(model_logprobs, cache.load_batch(batch_idx)).flatten()
        mx.eval(kl)
        all_kl.append(kl)
        _log_batch_progress("Processed", batch_idx + 1, num_batches)

    stats = _summarize_kl(all_kl)
    del model, tokens
    mx.clear_cache()
    return stats


def build_summary(args, cache: BaselineCache, stats: Dict, elapsed: float) -> Dict:
    total_positions = stats["total_positions"]
    return {
        "baseline_cache": str(cache.path),
        "baseline_model": cache.manifest["baseline_model"],
        "elapsed_seconds": elapsed,
        "mean_kl_per_token": stats["mean_kl_per_token"],
        "metric": "KL(baseline || model)",
        "model": args.model,
        "num_samples": cache.manifest["num_samples"],
        "sequence_length": cache.manifest["sequence_length"],
        "stderr": stats["stderr"],
        "tokens_per_second": total_positions / elapsed if elapsed > 0 else 0.0,
        "top_k": cache.manifest["top_k"],
        "total_positions": total_positions,
    }


def write_json(data: Dict) -> None:
    json.dump(data, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)
    args.baseline_model = canonicalize_ref(args.baseline_model)
    args.data_path = canonicalize_ref(args.data_path)

    cache = resolve_cache(args)
    start = time.time()
    stats = evaluate_kld(args, cache)
    elapsed = time.time() - start
    write_json(build_summary(args, cache, stats, elapsed))


def _get_vocab_size(tokenizer):
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)
    try:
        return int(len(tokenizer))
    except TypeError:
        return None


def _get_vocab_hash(tokenizer) -> Optional[str]:
    if not hasattr(tokenizer, "get_vocab"):
        return None
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        return None
    if not isinstance(vocab, dict):
        return None

    try:
        items = sorted((str(token), int(token_id)) for token, token_id in vocab.items())
    except (TypeError, ValueError):
        return None
    payload = json.dumps(items, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_candidate_tokenizer(tokenizer, manifest: Dict) -> None:
    candidate_vocab_size = _get_vocab_size(tokenizer)
    if (
        manifest["vocab_size"] is not None
        and candidate_vocab_size is not None
        and manifest["vocab_size"] != candidate_vocab_size
    ):
        raise ValueError(
            "Candidate tokenizer is incompatible with the baseline cache: "
            f"expected vocab size {manifest['vocab_size']}, got {candidate_vocab_size}."
        )

    candidate_hash = _get_vocab_hash(tokenizer)
    if (
        manifest["vocab_hash"] is not None
        and candidate_hash is not None
        and manifest["vocab_hash"] != candidate_hash
    ):
        raise ValueError(
            "Candidate tokenizer is incompatible with the baseline cache: "
            "token IDs do not match the cached baseline."
        )


def validate_top_k(top_k: int, vocab_size: Optional[int] = None) -> None:
    if top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")
    if vocab_size is not None and top_k >= vocab_size:
        raise ValueError(
            f"--top-k must be smaller than the vocabulary size ({vocab_size})."
        )


def iter_batches(tokens, batch_size: int):
    for start in range(0, len(tokens), batch_size):
        yield tokens[start : start + batch_size]


def cache_topk_batch(logprobs, top_k: int):
    kth = logprobs.shape[-1] - top_k
    indices = mx.argpartition(logprobs, kth=kth, axis=-1)[..., -top_k:].astype(mx.int32)
    top_logprobs = mx.take_along_axis(logprobs, indices, axis=-1)
    order = mx.argsort(-top_logprobs, axis=-1)
    indices = mx.take_along_axis(indices, order, axis=-1)
    top_logprobs = mx.take_along_axis(top_logprobs, order, axis=-1)
    tail_mass = mx.clip(1.0 - mx.sum(mx.exp(top_logprobs), axis=-1), 0.0, 1.0)
    return indices, top_logprobs, tail_mass.astype(mx.float32)


def kl_from_cached_batch(model_logprobs, cached_batch):
    base_top_logprobs = cached_batch["logprobs"]
    model_top_logprobs = mx.take_along_axis(
        model_logprobs,
        cached_batch["indices"],
        axis=-1,
    )
    base_top_probs = mx.exp(base_top_logprobs)
    kl_top = mx.sum(
        base_top_probs * (base_top_logprobs - model_top_logprobs),
        axis=-1,
    )

    base_tail_mass = mx.clip(cached_batch["tail_mass"], 0.0, 1.0)
    model_top_mass = mx.sum(mx.exp(model_top_logprobs), axis=-1)
    model_tail_mass = mx.clip(1.0 - model_top_mass, 1e-30, 1.0)
    base_tail_log = mx.log(mx.clip(base_tail_mass, 1e-30, 1.0))
    model_tail_log = mx.log(model_tail_mass)
    kl_tail = mx.where(
        base_tail_mass > 0,
        base_tail_mass * (base_tail_log - model_tail_log),
        0.0,
    )
    return kl_top + kl_tail


def _summarize_kl(all_kl) -> dict:
    values = all_kl[0] if len(all_kl) == 1 else mx.concatenate(all_kl)
    mean = values.mean().item()
    count = int(values.size)
    if count > 1:
        variance = mx.var(values, ddof=1).item()
        stderr = math.sqrt(max(variance, 0.0)) / math.sqrt(count)
    else:
        stderr = 0.0
    return {
        "mean_kl_per_token": mean,
        "stderr": stderr,
        "total_positions": count,
    }


def _invalid_cache(cache_dir: Path) -> ValueError:
    return ValueError(
        f"Baseline cache at {cache_dir} is invalid. "
        "Delete it or choose a different --baseline-cache."
    )


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _log_batch_progress(prefix: str, current: int, total: int) -> None:
    if total <= 10:
        _log(f"  {prefix} {current}/{total} batches")
        return

    interval = max(1, total // 10)
    if current == 1 or current == total or current % interval == 0:
        _log(f"  {prefix} {current}/{total} batches")


if __name__ == "__main__":
    main()
