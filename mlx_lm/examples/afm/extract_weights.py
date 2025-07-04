import argparse
import json
import textwrap
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import LlamaTokenizerFast


def share_data(a, b):
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def get_model_config():
    return {
        "model_type": "afm7",
        "vocab_size": 153600,
        "hidden_dim": 2048,
        "num_layers": 56,
        "num_kv_reuse_layers": 21,
        "num_heads": 16,
        "num_kv_heads": 2,
        "hidden_dim_scale_factor": 3.25,
        "rope_theta": 500000.0,
    }


def get_adapter_config():
    return {
        "num_layers": 56,
        "lora_parameters": {
            "rank": 32,
            "scale": 0.5,
            "dropout": 0.0,
            "keys": [
                "mlp.gate_proj",
                "mlp.down_proj",
                "mlp.up_proj",
                "self_attn.qkv_proj",
                "self_attn.q_proj",
                "self_attn.out_proj",
            ],
        },
    }


def get_chat_template():
    return textwrap.dedent(
        """
        {%- set default_system_message = "A conversation between a user and a helpful assistant." %}
        {%- if messages[0]['role'] == 'system' %}
            {%- set system_message = messages[0]['content'] %}
            {%- set loop_messages = messages[1:] %}
        {%- else %}
            {%- set system_message = default_system_message %}
            {%- set loop_messages = messages %}
        {%- endif %}
        {{- '<turn_start> system<n>' + system_message -}}
        {% if tools %}
            {{- ('<n>system tools: ' + (tools | map('tojson') | join('<n>'))) -}}
        {% endif %}
        {{- '<turn_end>' -}}
        {% for message in loop_messages %}
            {{- '<turn_start> ' + message['role'] + '<n>' + message['content'] + '<turn_end>' -}}
        {% endfor %}
        {% if add_generation_prompt is defined and add_generation_prompt %}
            {% if messages[-1]['role'] != 'assistant' %}
                {{- '<turn_start> assistant<n>' -}}
            {% endif %}
        {% endif %}"""
    ).strip()


def map_model_keys(state):
    model_keys = {}
    for old in state:
        if "adapter" in old:
            continue
        if "kv_quantizer" in old:
            continue

        new = old
        if new.startswith("layers."):
            new = new[7:]
            new = new.replace("layer_", "")
            new = new.replace("attention.norm", "input_layernorm")
            new = new.replace(".attention.", ".self_attn.")
            new = new.replace("self_attn.output_transform", "self_attn.out_proj")
            new = new.replace("feed_forward.norm", "post_attention_layernorm")
            new = new.replace(".feed_forward.", ".mlp.")
            new = new.replace("hidden_transform.linear_0", "gate_proj")
            new = new.replace("hidden_transform.linear_1", "up_proj")
            new = new.replace("mlp.output_transform", "mlp.down_proj")
            if new.startswith("segment_0"):
                new = new.replace("segment_0", "layers")
                new = new.replace(".qkv_transform.", ".qkv_proj.")
                new = new.replace(".fused_linear.", ".")
                new = new.replace(".qk_norm.query_norm.", ".q_norm.")
                new = new.replace(".qk_norm.key_norm.", ".k_norm.")
            elif new.startswith("segment_1"):
                new = new.replace("segment_1", "kv_reuse_layers")
                new = new.replace(".q_transform.", ".q_proj.")
                new = new.replace(".q_norm.query_norm.", ".q_norm.")
        new = new.replace(".wrapped.", ".")
        new = "model." + new
        model_keys[old] = new
    return model_keys


def map_adapter_keys(state):
    adapter_keys = {}
    for old in state:
        if "adapter" not in old:
            continue

        new = old
        new = new[7:]
        new = new.replace("layer_", "")
        new = new.replace(".attention.", ".self_attn.")
        new = new.replace("self_attn.output_transform", "self_attn.out_proj")
        new = new.replace(".feed_forward.", ".mlp.")
        new = new.replace("hidden_transform.linear_0", "gate_proj")
        new = new.replace("hidden_transform.linear_1", "up_proj")
        new = new.replace("mlp.output_transform", "mlp.down_proj")
        if new.startswith("segment_0"):
            new = new.replace("segment_0", "layers")
            new = new.replace(".qkv_transform.", ".qkv_proj.")
            new = new.replace(".fused_linear.", ".")
        elif new.startswith("segment_1"):
            new = new.replace("segment_1", "kv_reuse_layers")
            new = new.replace(".q_transform.", ".q_proj.")

        new = new.replace(".lora_0.b_transpose", ".b_transpose.0")
        new = new.replace(".lora_1.b_transpose", ".b_transpose.1")
        new = new.replace(".lora_2.b_transpose", ".b_transpose.2")
        new = new.replace(".lora_0.a_transpose", ".a_transpose.0")
        new = new.replace(".lora_1.a_transpose", ".a_transpose.1")
        new = new.replace(".lora_2.a_transpose", ".a_transpose.2")
        new = new.replace("adapters.base_adapter.b_transpose", "lora_b")
        new = new.replace("adapters.base_adapter.a_transpose", "lora_a")
        new = "model." + new
        adapter_keys[old] = new
    return adapter_keys


def add_kv_quant_weights(new_state, old_state, dt):
    for k, v in old_state.items():
        if "range" not in k:
            continue

        v = v.tolist()
        weight = "quant_key_scale" if "key_quantizer" in k else "quant_value_scale"
        new_k = k[: k.find("kv_quantizer")]
        new_k = new_k.replace("segment_0.layer_", "")
        new_k = new_k.replace("attention", "self_attn")
        new_k = "model." + new_k + weight
        quant_scale = torch.tensor(max(v[0] / (-128), v[1] / 127), dtype=dt)
        new_state[new_k] = quant_scale


def cast(x, dt):
    info = torch.finfo(dt)
    a, b = info.min, info.max
    return x.clip(a, b).to(dt)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Map the PT weights to MLX-LM safetensors"
    )
    parser.add_argument("source", help="The source weights in PT format")
    parser.add_argument("tokenizer", help="The source tokenizer file")
    parser.add_argument("destination", help="The folder to write the model weights in")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="float32"
    )
    parser.add_argument(
        "--adapter-dtype", choices=["bfloat16", "float16", "float32"], default="float32"
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="If set overwrite the weight files in the destination folder",
    )
    args = parser.parse_args(argv)

    destination = Path(args.destination)
    if not destination.exists():
        destination.mkdir()
    model_file = destination / "model.safetensors"
    adapter_file = destination / "adapters.safetensors"
    if (model_file.exists() or adapter_file.exists()) and not args.force:
        print("Model files already exist. Delete them or use --force to overwrite them")
        return

    # Write the configuration files
    with (destination / "config.json").open("w") as f:
        json.dump(get_model_config(), f, indent=4)
    with (destination / "adapter_config.json").open("w") as f:
        json.dump(get_adapter_config(), f, indent=4)

    # Pop the tied output transform
    state = torch.load(args.source)
    if share_data(state["embedding.weight"], state["output_transform.weight"]):
        state.pop("output_transform.weight")

    # Map the weights
    model_keys = map_model_keys(state)
    adapter_keys = map_adapter_keys(state)

    # Make the new weight dictionaries
    dt = getattr(torch, args.dtype)
    adapter_dt = getattr(torch, args.adapter_dtype)
    adapters = {
        k_new: cast(state[k_old], adapter_dt) for k_old, k_new in adapter_keys.items()
    }
    model = {k_new: cast(state[k_old], dt) for k_old, k_new in model_keys.items()}
    add_kv_quant_weights(model, state, dt)

    # Save them to disk
    save_file(model, model_file)
    save_file(adapters, adapter_file)

    # Save the tokenizer
    tok = LlamaTokenizerFast(vocab_file=args.tokenizer)
    tok.chat_template = get_chat_template()
    tok.eos_token_ids = tok.convert_tokens_to_ids("<turn_end>")
    tok.save_pretrained(str(destination))
    with (destination / "tokenizer_config.json").open("r+") as f:
        config = json.load(f)
        config["tokenizer_class"] = "NewlineTokenizer"
        f.seek(0)
        json.dump(config, f, indent=4)
        f.truncate()
    with (destination / "tokenizer.json").open("r+") as f:
        tok = json.load(f)
        tok["decoder"]["decoders"].insert(
            1,
            {"type": "Replace", "pattern": {"String": "<n>"}, "content": "\n"},
        )
        f.seek(0)
        json.dump(tok, f, indent=4)
        f.truncate()


if __name__ == "__main__":
    main(None)
