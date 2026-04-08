#!/bin/bash
# DFlash 27B Server Startup Script
# Usage: ./start_dflash_27b.sh

MODEL="/Users/ali/.lmstudio/models/mlx-community/Qwen3.5-27B-4bit"
DRAFT_MODEL="z-lab/Qwen3.5-27B-DFlash"
HOST="0.0.0.0"
PORT=11111
MAX_TOKENS=16384

echo "Starting DFlash 27B Server..."
echo "Target Model: $MODEL"
echo "Draft Model: $DRAFT_MODEL"
echo "Max Tokens: $MAX_TOKENS"
echo "Host: $HOST:$PORT"

python -m mlx_lm.server \
  --model "$MODEL" \
  --draft-model "$DRAFT_MODEL" \
  --quantize-draft \
  --host "$HOST" \
  --port $PORT \
  --max-tokens $MAX_TOKENS \
  --log-level INFO
