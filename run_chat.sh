th
DEFAULT_MODEL="/Users/ec2-user/mlx-lm/ckpt-28232"

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

# Use python3 explicitly
PYTHON_EXEC="python3"

# Run chat
# Adjust max-tokens, temp etc as needed.
$PYTHON_EXEC -m mlx_lm.chat \
    --model "$MODEL_PATH" \
    --trust-remote-code \
    --max-tokens 512 \
    --temp 0.7
