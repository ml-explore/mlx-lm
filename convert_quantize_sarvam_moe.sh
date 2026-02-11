#!/bin/bash

# Check if model path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <model_path>"
  exit 1
fi

MODEL_PATH="$1"
MODEL_NAME="sarvam_moe"
QUANTIZED_MODEL_PATH="${MODEL_NAME}-mxfp4"

echo "Converting model from: $MODEL_PATH"
echo "Output path: $QUANTIZED_MODEL_PATH"

# Convert and quantize sarvam_moe from provided path to 4 bits (mxfp4)
python3 -m mlx_lm.convert \
  --hf-path "$MODEL_PATH" \
  -q \
  --q-bits 4 \
  --q-mode mxfp4 \
  --q-group-size 32 \
  --mlx-path "$QUANTIZED_MODEL_PATH"

# Generate text using the quantized model
python3 -m mlx_lm.generate \
  --model "$QUANTIZED_MODEL_PATH" \
  --prompt "What is the capital of France?" \
  --max-tokens 50
