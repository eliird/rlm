#!/bin/bash
# Deploy DeepSeek-R1-Distill-Qwen-32B using vLLM OpenAI-compatible server.
# Run from repo root: bash math/benchmark/serve.sh
# Logs are written to math/benchmark/logs/vllm_serve.log

MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
CACHE_DIR="/data/cache/huggingface/hub"
PORT=8000
LOG_DIR="math/benchmark/logs"
LOG_FILE="$LOG_DIR/vllm_serve.log"

mkdir -p "$LOG_DIR"
echo "Logs -> $LOG_FILE"

.venv/bin/vllm serve "$MODEL_ID" \
    --download-dir "$CACHE_DIR" \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --port "$PORT" \
    --served-model-name deepseek-r1-32b  

