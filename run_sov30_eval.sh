#!/bin/bash

# Check if model path is provided, otherwise use default
if [ -z "$1" ]; then
    MODEL_PATH="/Users/rachittibrewal/Documents/mlx/mlx-lm/sov-30b-fp8"
    echo "No model path provided. Using default: $MODEL_PATH"
else
    MODEL_PATH="$1"
fi

# Ensure mlx-lm is installed and visible
# Assuming running from the root of the repo or installed in environment
# If running source:
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "================================================="
echo "Running Evaluation for Sov30 MoE"
echo "Model Path: $MODEL_PATH"
echo "Benchmarks: GSM8K, MMLU"
echo "================================================="

# Run Evaluation using mlx_lm.evaluate
# Note: This requires 'lm_eval' to be installed (pip install lm-eval)
# 'mlx_lm.evaluate' uses the lm-evaluation-harness

# Evaluating GSM8K (Chain of Thought/generation typically, usually 5-shot or as defined by task)
# Evaluating MMLU (Multiple Choice, usually 5-shot)

# We use the python executable from the environment if determined, else default 'python'
PYTHON_EXEC="python"
if [ -x "/opt/anaconda3/envs/mlx_env/bin/python" ]; then
    PYTHON_EXEC="/opt/anaconda3/envs/mlx_env/bin/python"
fi

$PYTHON_EXEC -m mlx_lm.evaluate \
    --model "$MODEL_PATH" \
    --tasks gsm8k mmlu \
    --batch-size 4 \
    --trust-remote-code

echo "================================================="
echo "Evaluation Complete."
echo "Results saved in current directory."
