#!/bin/bash
# Deploy gpt-oss-120b using vLLM OpenAI-compatible server.
# Run from repo root: bash math/benchmark/serve.sh
# Logs are written to math/benchmark/logs/vllm_serve.log

MODEL_ID="openai/gpt-oss-120b"
CACHE_DIR="/data/cache/huggingface/hub"
PORT=8000
LOG_DIR="math/benchmark/logs"
LOG_FILE="$LOG_DIR/vllm_serve.log"

mkdir -p "$LOG_DIR"
echo "Logs -> $LOG_FILE"

.venv/bin/vllm serve "$MODEL_ID" \
    --download-dir "$CACHE_DIR" \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --port "$PORT" \
    --served-model-name gpt-oss-120b \
    2>&1 | tee "$LOG_FILE"
