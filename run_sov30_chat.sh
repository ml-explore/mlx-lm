#!/bin/bash

# Default model path
DEFAULT_MODEL="/Users/rachittibrewal/Documents/mlx/mlx-lm/sov-30b-fp8"

# Check if model path is provided as argument
if [ -z "$1" ]; then
    MODEL_PATH="$DEFAULT_MODEL"
    echo "No model path provided. Using default: $MODEL_PATH"
else
    MODEL_PATH="$1"
fi

# Ensure mlx-lm is visible
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "================================================="
echo "Starting Chat with Sov30 MoE"
echo "Model Path: $MODEL_PATH"
echo "================================================="

# Use the correct python environment
PYTHON_EXEC="python"
if [ -x "/opt/anaconda3/envs/mlx_env/bin/python" ]; then
    PYTHON_EXEC="/opt/anaconda3/envs/mlx_env/bin/python"
fi

# Run chat
# Adjust max-tokens, temp etc as needed.
$PYTHON_EXEC -m mlx_lm.chat \
    --model "$MODEL_PATH" \
    --trust-remote-code \
    --max-tokens 512 \
    --temp 0.7 
