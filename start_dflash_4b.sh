#!/bin/bash
# DFlash 4B Server Startup Script
# Usage: ./start_dflash_4b.sh

MODEL="Qwen/Qwen3.5-4B"
DRAFT_MODEL="z-lab/Qwen3.5-4B-DFlash"
HOST="0.0.0.0"
PORT=11113
MAX_TOKENS=16384
BLOCK_SIZE=16

echo "Starting DFlash 4B Server..."
echo "Target Model: $MODEL"
echo "Draft Model: $DRAFT_MODEL"
echo "Block Size: $BLOCK_SIZE"
echo "Max Tokens: $MAX_TOKENS"
echo "Host: $HOST:$PORT"

python -m mlx_lm.server \
  --model "$MODEL" \
  --draft-model "$DRAFT_MODEL" \
  --block-size $BLOCK_SIZE \
  --host "$HOST" \
  --port $PORT \
  --max-tokens $MAX_TOKENS \
  --log-level INFO
