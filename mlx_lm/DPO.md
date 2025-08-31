# Fine-Tuning with Direct Preference Optimization (DPO)

You can use the `mlx-lm` package to fine-tune an LLM with Direct Preference Optimization (DPO) for human preference alignment.[^dpo] DPO training works with the following model families:

- Mistral
- Llama
- Phi2
- Mixtral
- Qwen2
- Gemma
- OLMo
- MiniCPM
- InternLM2

## Contents

- [Run](#Run)
  - [Fine-tune](#Fine-tune)
  - [Evaluate](#Evaluate)
  - [Generate](#Generate)
- [Fuse](#Fuse)
- [Data](#Data)
- [Memory Issues](#Memory-Issues)
- [Understanding DPO](#Understanding-DPO)

## Run

First, make sure you have the training dependencies installed:

```shell
pip install "mlx-lm[train]"
```

The main command is `mlx_lm.dpo`. To see a full list of command-line options run:

```shell
mlx_lm.dpo --help
```

Note, in the following the `--model` argument can be any compatible Hugging Face repo or a local path to a converted model.

You can also specify a YAML config with `-c`/`--config`. For more on the format see the [example YAML](#Configuration-Examples). For example:

```shell
mlx_lm.dpo --config /path/to/config.yaml
```

If command-line flags are also used, they will override the corresponding values in the config.

### Fine-tune

To fine-tune a model with DPO use:

```shell
mlx_lm.dpo \
    --model <path_to_model> \
    --train \
    --data <path_to_data> \
    --iters 1000 \
    --beta 0.1
```

To fine-tune the full model weights, add the `--fine-tune-type full` flag. Currently supported fine-tuning types are `lora` (default), `dora`, and `full`.

The `--data` argument must specify a path to a `train.jsonl`, `valid.jsonl` when using `--train` and a path to a `test.jsonl` when using `--test`. For more details on the data format see the section on [Data](#Data).

For example, to fine-tune a Llama 7B model with DPO you can use:

```shell
mlx_lm.dpo \
    --model mlx-community/Meta-Llama-3-8B-Instruct-4bit \
    --train \
    --data /path/to/preference_data \
    --beta 0.1 \
    --fine-tune-type lora
```

If `--model` points to a quantized model, then the training will use QLoRA, otherwise it will use regular LoRA.

By default, the adapter config and learned weights are saved in `adapters/`. You can specify the output location with `--adapter-path`.

You can resume fine-tuning with an existing adapter with `--resume-adapter-file <path_to_adapters.safetensors>`.

#### DPO-Specific Options

**Beta Parameter (`--beta`)**
The temperature parameter that controls the strength of the KL penalty in DPO. Common values:
- `0.1` (default): Standard DPO training
- `0.01`: More conservative, stays closer to reference model  
- `0.5`: More aggressive preference optimization

**Reference Model (`--reference-model`)**
By default, DPO uses the initial policy model as the reference model. You can specify a different reference model:

```shell
mlx_lm.dpo \
    --model <policy_model> \
    --reference-model <reference_model> \
    --train \
    --data <path_to_data>
```

#### Logging

You can log training metrics to Weights & Biases using `--report-to wandb`, or to SwanLab using `--report-to swanlab`. Make sure to install the required packages beforehand: `pip install wandb` or `pip install swanlab`. You can enable both tracking tools simultaneously by separating them with a comma, for example: `--report-to wandb,swanlab`.

To specify a project name for the logging tracker, use `--project-name <YOUR PROJECT NAME>`.

### Evaluate

To compute test set performance use:

```shell
mlx_lm.dpo \
    --model <path_to_model> \
    --adapter-path <path_to_adapters> \
    --data <path_to_data> \
    --test
```

### Generate

For generation use `mlx_lm.generate`:

```shell
mlx_lm.generate \
    --model <path_to_model> \
    --adapter-path <path_to_adapters> \
    --prompt "<your_model_prompt>"
```

## Fuse

You can generate a model fused with the low-rank adapters using the `mlx_lm.fuse` command. This works the same way as LoRA adapters. See the main [LORA.md](LORA.md#Fuse) documentation for details.

## Data

The DPO command expects you to provide preference data with `--data`. The data should contain pairs of preferred and dispreferred responses for the same prompt.

Datasets can be specified in `*.jsonl` files locally or loaded from Hugging Face.

### Local Datasets

For fine-tuning (`--train`), the data loader expects a `train.jsonl` and a `valid.jsonl` to be in the data directory. For evaluation (`--test`), the data loader expects a `test.jsonl` in the data directory.

Currently, `*.jsonl` files support two preference data formats:

#### Simple Prompt Format

```jsonl
{"prompt": "What is the capital of France?", "chosen": "The capital of France is Paris, a beautiful city known for its art, culture, and the Eiffel Tower.", "rejected": "France's capital is Paris."}
```

This format is ideal for simple question-answer pairs where you have a clear prompt and want to train the model to prefer more detailed, helpful responses.

#### Chat Format

```jsonl
{"messages": [{"role": "user", "content": "Hello, how are you?"}], "chosen": "Hello! I'm doing well, thank you for asking. How can I help you today?", "rejected": "Hi."}
```

This format is better for conversational models where you want to maintain chat context. The `messages` field contains the conversation history, and `chosen`/`rejected` are the assistant's alternative responses.

### Data Quality Guidelines

For effective DPO training, ensure your preference data has:

1. **Clear Preferences**: The chosen response should be meaningfully better than the rejected one
2. **Same Context**: Both responses should address the same prompt/conversation
3. **Realistic Alternatives**: Rejected responses should be plausible but suboptimal
4. **Diverse Examples**: Include various types of improvements (helpfulness, accuracy, safety, style)

### Example Preference Pairs

**Good example - Helpfulness**:
```json
{
  "prompt": "How do I make scrambled eggs?",
  "chosen": "To make scrambled eggs: 1) Crack eggs into a bowl and whisk with salt and pepper, 2) Heat butter in a non-stick pan over medium-low heat, 3) Pour in eggs and gently stir constantly until they form soft, creamy curds. Remove from heat while still slightly wet as they'll continue cooking.",
  "rejected": "Just crack some eggs in a pan and stir them around until cooked."
}
```

**Good example - Safety/Accuracy**:
```json
{
  "prompt": "What should I do if I think I have a serious medical condition?",
  "chosen": "If you think you have a serious medical condition, you should consult with a healthcare professional immediately. They can properly evaluate your symptoms and provide appropriate medical advice. Don't delay seeking professional medical care for serious concerns.",
  "rejected": "You can probably just look it up online and self-treat with home remedies."
}
```

### Hugging Face Datasets

To use Hugging Face preference datasets, you can specify them directly if they're in the supported format:

```shell
mlx_lm.dpo --data anthropic/hh-rlhf --train
```

Popular DPO datasets include:
- `anthropic/hh-rlhf`: Human preference data for helpfulness and harmlessness
- `Anthropic/hhrlhf`: Constitutional AI preference data
- `berkeley-nest/Nectar`: Multi-turn conversation preferences

For custom mappings, use a YAML config:

```yaml
hf_dataset:
  path: "your-dataset/preference-data"
  prompt_feature: "question"
  chosen_feature: "good_answer"
  rejected_feature: "bad_answer"
```

## Memory Issues

DPO training requires approximately 2x the memory of standard LoRA training since it needs to run forward passes through both the policy and reference models. Here are tips to reduce memory use:

1. **Use Quantization**: QLoRA with DPO significantly reduces memory usage:
   ```shell
   mlx_lm.dpo --model <quantized_model> --train --data <data>
   ```

2. **Reduce Batch Size**: Use `--batch-size 1` or `2` instead of the default `4`:
   ```shell
   mlx_lm.dpo --batch-size 1 --train --data <data>
   ```

3. **Limit Layers**: Reduce layers to fine-tune with `--num-layers 8` or `4`

4. **Sequence Length**: Use shorter sequences with `--max-seq-length 512` or `1024`

5. **Gradient Checkpointing**: Trade computation for memory with `--grad-checkpoint`

Example for a machine with 32 GB:

```shell
mlx_lm.dpo \
    --model mlx-community/Meta-Llama-3-8B-Instruct-4bit \
    --train \
    --batch-size 1 \
    --num-layers 4 \
    --max-seq-length 512 \
    --data preference_data/ \
    --beta 0.1
```

## Understanding DPO

Direct Preference Optimization (DPO) is a method for training language models to align with human preferences without requiring a separate reward model. Here's how it works:

### Key Concepts

1. **Preference Learning**: Instead of training on single "correct" responses, DPO learns from pairs of responses where one is preferred over the other

2. **Reference Model**: DPO uses a reference model (typically the initial policy model) to regularize training and prevent the model from deviating too far from its original behavior

3. **Beta Parameter**: Controls the tradeoff between following preferences and staying close to the reference model. Higher beta = stronger preference optimization

### DPO vs RLHF

Traditional RLHF requires:
- Training a reward model on preference data
- Using reinforcement learning (PPO) to optimize against the reward model
- Complex multi-stage training process

DPO simplifies this to:
- Direct optimization on preference data
- Single-stage training process  
- Mathematically equivalent results to RLHF

### Training Process

During DPO training, the model learns to:
1. Increase the likelihood of chosen responses
2. Decrease the likelihood of rejected responses  
3. Stay close to the reference model (controlled by beta)
4. Maintain general language modeling capabilities

### Configuration Examples

**Basic DPO Config (YAML)**:
```yaml
model: "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
train: true
fine_tune_type: "lora"
data: "preference_data/"
beta: 0.1
batch_size: 4
iters: 1000
learning_rate: 1e-6
max_seq_length: 1024
lora_parameters:
  rank: 8
  dropout: 0.0
  scale: 20.0
```

**Advanced DPO Config**:
```yaml  
model: "meta-llama/Llama-2-7b-hf"
train: true
fine_tune_type: "lora"
data: "anthropic/hh-rlhf"
beta: 0.2
batch_size: 8
iters: 2000
learning_rate: 5e-7
max_seq_length: 2048
lora_parameters:
  rank: 16
  dropout: 0.1
  scale: 10.0
reference_model: "path/to/reference/model"
grad_checkpoint: true
report_to: "wandb"
project_name: "dpo-llama-experiments"
```

## Best Practices

1. **Start Small**: Begin with a small learning rate (1e-6) and low beta (0.1)

2. **Quality over Quantity**: Better to have fewer, high-quality preference pairs than many noisy ones

3. **Monitor KL Divergence**: Watch that your model doesn't drift too far from the reference (target KL < 2.0)

4. **Validate Regularly**: Use a held-out preference validation set to monitor training

5. **Reference Model**: For best results, use a well-trained chat/instruct model as your starting point

[^dpo]: Refer to the [arXiv paper](https://arxiv.org/abs/2305.18290) "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" for more details on DPO.