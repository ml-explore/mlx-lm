# Fine-Tuning with Direct Preference Optimization (DPO)

You can use the `mlx-lm` package to fine-tune an LLM with Direct Preference Optimization (DPO) for human preference alignment.[^dpo] DPO allows you to train models to prefer certain responses over others without requiring a separate reward model.

## Contents

- [What is DPO](#What-is-DPO)
- [Quick Start](#Quick-Start) 
- [DPO-Specific Options](#DPO-Specific-Options)
- [Preference Data Format](#Preference-Data-Format)
- [Configuration Examples](#Configuration-Examples)
- [DPO vs RLHF](#DPO-vs-RLHF)

## What is DPO

Direct Preference Optimization (DPO) is a method for training language models to align with human preferences. Unlike traditional RLHF which requires training a separate reward model, DPO directly optimizes on preference data.

**Key benefits:**
- **Simpler**: Single-stage training process (no reward model needed)
- **Stable**: More stable than PPO-based RLHF training  
- **Effective**: Mathematically equivalent to RLHF under certain conditions
- **Memory Efficient**: Can work with LoRA/QLoRA for efficient fine-tuning

## Quick Start

Install training dependencies:
```shell
pip install "mlx-lm[train]"
```

Basic DPO fine-tuning:
```shell
mlx_lm.dpo \
    --model mlx-community/Meta-Llama-3-8B-Instruct-4bit \
    --train \
    --data /path/to/preference_data \
    --beta 0.1 \
    --iters 1000
```

For help with all options:
```shell
mlx_lm.dpo --help
```

## DPO-Specific Options

### Beta Parameter (`--beta`)
Controls the strength of the KL penalty. Higher values = stronger preference optimization.

- `--beta 0.01`: Conservative, stays close to reference model
- `--beta 0.1`: Standard (default)
- `--beta 0.5`: Aggressive preference optimization

### Reference Model (`--reference-model`)
By default, uses the initial policy model. You can specify a different one:

```shell
mlx_lm.dpo \
    --model <policy_model> \
    --reference-model <path_to_reference> \
    --train \
    --data <preference_data>
```

### Using Finetuned Adapters as Reference Model (`--reference-adapter-path`)
You can use previously finetuned LoRA adapters as the reference model for DPO training. This allows you to chain multiple fine-tuning stages:

```shell
# Use a finetuned adapter as reference model
mlx_lm.dpo \
    --model mlx-community/Meta-Llama-3-8B-Instruct-4bit \
    --reference-model mlx-community/Meta-Llama-3-8B-Instruct-4bit \
    --reference-adapter-path /path/to/previous/adapters \
    --train \
    --data <preference_data>
```

**Use cases:**
- **Multi-stage fine-tuning**: First fine-tune with LoRA, then use that as reference for DPO
- **Domain adaptation + preference alignment**: Use domain-adapted model as reference
- **Iterative improvement**: Use previous DPO results as reference for further optimization

**Requirements:**
- Adapter directory must contain `adapter_config.json` and `adapters.safetensors`
- Base model specified in `--reference-model` should match the original model used for the adapters

### Fine-Tuning Types (`--fine-tune-type`)
Choose how much of the model to update:

- `--fine-tune-type lora` (default): Low-rank adaptation - efficient, small adapter files
- `--fine-tune-type dora`: DoRA (Weight-Decomposed Low-Rank Adaptation) - better quality than LoRA
- `--fine-tune-type full`: Full parameter fine-tuning - highest quality, largest memory usage

```shell
# LoRA fine-tuning (recommended)
mlx_lm.dpo --fine-tune-type lora --train --data <data>

# Full fine-tuning for maximum quality
mlx_lm.dpo --fine-tune-type full --train --data <data>
```

## Preference Data Format

DPO requires preference data with `chosen` and `rejected` response pairs. Create `train.jsonl` and `valid.jsonl` files in your data directory.

### Simple Format
```jsonl
{"prompt": "What is the capital of France?", "chosen": "The capital of France is Paris, a beautiful city known for its art, culture, and the Eiffel Tower.", "rejected": "Paris."}
```

### Chat Format  
```jsonl
{"messages": [{"role": "user", "content": "Hello, how are you?"}], "chosen": "Hello! I'm doing well, thank you for asking. How can I help you today?", "rejected": "Hi."}
```

### Data Quality Tips
- **Clear preferences**: Chosen should be meaningfully better than rejected
- **Same context**: Both responses address the same prompt
- **Realistic alternatives**: Rejected responses should be plausible but suboptimal

### Using Your Own Data
```shell
# Point to your preference data directory
mlx_lm.dpo --data /path/to/preference_data --train
```

## Configuration Examples

### Basic YAML Config
```yaml
model: "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
train: true
data: "preference_data/"
beta: 0.1
batch_size: 4
iters: 1000
learning_rate: 1e-6
fine_tune_type: "lora"
```

### Memory-Optimized Config
```yaml
model: "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
train: true
data: "preference_data/"
beta: 0.1
batch_size: 1
iters: 1000
learning_rate: 1e-6
max_seq_length: 512
num_layers: 4
grad_checkpoint: true
```

### Multi-Stage Fine-Tuning Config
```yaml
model: "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
reference_model: "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
reference_adapter_path: "models/checkpoints/lora_adapters"
train: true
data: "preference_data/"
beta: 0.1
batch_size: 4
iters: 1000
learning_rate: 1e-6
fine_tune_type: "lora"
```

## DPO vs RLHF

**Traditional RLHF:**
- Train reward model on preferences
- Use PPO to optimize policy against reward model
- Complex multi-stage process
- Unstable training

**DPO:**
- Direct optimization on preference data
- Single training stage
- More stable training
- Mathematically equivalent results

**When to use DPO:**
- You have preference data (chosen/rejected pairs)
- Want simpler training than RLHF
- Need stable preference optimization  
- Working with limited compute resources

[^dpo]: Refer to the [arXiv paper](https://arxiv.org/abs/2305.18290) "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" for more details on DPO.