# ABOUTME: Profiles a single prefill chunk for per-layer timing without mx.compile.
# ABOUTME: Emits attention/MLP/overlay timing breakdown for ramp analysis.

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlx.core as mx

from mlx_lm import load
from mlx_lm.server_batched.graph_decode.llama_prefill import LlamaPrefillGraph


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--prefill-chunk", type=int, default=64)
    parser.add_argument("--prefill-ramp-chunk", type=int, default=64)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--summary-path", type=str, default=None)
    parser.add_argument(
        "--prefill-profile",
        dest="prefill_profile",
        action="store_true",
        help="Enable MLXLM_PREFILL_PROFILE so the graph emits per-layer timings.",
    )
    parser.add_argument(
        "--no-prefill-profile",
        dest="prefill_profile",
        action="store_false",
        help="Disable MLXLM_PREFILL_PROFILE even if the environment sets it.",
    )
    parser.set_defaults(prefill_profile=True)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    model, tokenizer = load(args.repo)
    chunk_len = max(1, int(args.prefill_chunk))
    ramp = max(1, min(chunk_len, int(args.prefill_ramp_chunk)))

    if args.prefill_profile:
        os.environ["MLXLM_PREFILL_PROFILE"] = "1"
    else:
        os.environ.pop("MLXLM_PREFILL_PROFILE", None)

    graph = LlamaPrefillGraph(model)
    fn = graph.get_compiled(
        batch_size=1,
        chunk_len=chunk_len,
        block_tables_shape=(1, 1),
        k_cache_shape=(
            len(model.layers),
            model.layers[0].self_attn.n_kv_heads,
            1,
            chunk_len,
            graph.head_dim,
        ),
        v_cache_shape=(
            len(model.layers),
            model.layers[0].self_attn.n_kv_heads,
            1,
            chunk_len,
            graph.head_dim,
        ),
        kv_map_shape=None,
        dtype=mx.float16,
        pending_flag=0,
    )

    tokens = mx.zeros((1, chunk_len), dtype=mx.int32)
    base_lens = mx.zeros((1,), dtype=mx.int32)
    block_tables = mx.zeros((1, 1), dtype=mx.int32)
    n_layers = len(model.layers)
    n_kv_heads = model.layers[0].self_attn.n_kv_heads
    head_dim = graph.head_dim
    cache_dtype = model.layers[0].self_attn.q_proj.weight.dtype
    k_cache = mx.zeros(
        (n_layers, n_kv_heads, 1, chunk_len, head_dim), dtype=cache_dtype
    )
    v_cache = mx.zeros_like(k_cache)
    overlay_shape = (n_layers, chunk_len, 1, n_kv_heads, head_dim)
    pending_k = mx.zeros(overlay_shape, dtype=cache_dtype)
    pending_v = mx.zeros_like(pending_k)

    fn(tokens, base_lens, block_tables, k_cache, v_cache, pending_k, pending_v)
    metrics = graph.consume_metrics()

    summary = {
        "chunk_len": chunk_len,
        "prefill_ramp_chunk": ramp,
        "array_prefill_attn_s": metrics.get("array_prefill_attn_s", 0.0),
        "array_prefill_mlp_s": metrics.get("array_prefill_mlp_s", 0.0),
        "array_prefill_overlay_s": metrics.get("array_prefill_overlay_s", 0.0),
        "array_prefill_qkv_s": metrics.get("array_prefill_qkv_s", 0.0),
        "array_prefill_rope_s": metrics.get("array_prefill_rope_s", 0.0),
        "array_prefill_layer_attn_s": metrics.get("array_prefill_layer_attn_s", []),
        "array_prefill_layer_mlp_s": metrics.get("array_prefill_layer_mlp_s", []),
        "array_prefill_layer_overlay_s": metrics.get(
            "array_prefill_layer_overlay_s", []
        ),
    }

    if args.summary_path:
        Path(args.summary_path).write_text(json.dumps(summary, indent=2))
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
