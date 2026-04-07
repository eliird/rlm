#!/bin/bash
# RLM training loop: N iterations of
#   serve -> infer (1000) -> eval -> generate (10k) -> train (2 epochs) -> stop server
#
# Checkpoints : math/checkpoints/iter_{N}/
# Corrections : math/data/iter_{N}_corrections.jsonl
# Responses   : math/benchmark/results/iter_{N}_responses.jsonl
#
# Run from repo root: bash math/rlm_loop.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=".venv/bin/python"
DEEPSPEED=".venv/bin/deepspeed"

ITERATIONS=3
TRAIN_EPOCHS=2
INFER_LIMIT=1000
INFER_BATCH=512
GEN_TARGET=10000
GEN_BATCH=512
MIN_FREE_GB=300   # abort before saving checkpoint if less than this is free

SERVER_LOG="math/benchmark/logs/vllm_serve.log"
SERVER_PID_FILE="/tmp/vllm_server.pid"

# ── helpers ──────────────────────────────────────────────────────────────────

free_gb() {
    df -BG /data | awk 'NR==2 {gsub("G",""); print $4}'
}

check_space() {
    local needed_gb=$1
    local label=$2
    local avail
    avail=$(free_gb)
    if [ "$avail" -lt "$needed_gb" ]; then
        echo "ERROR: Only ${avail}GB free on /data, need ${needed_gb}GB before ${label}. Aborting."
        exit 1
    fi
    echo "Space check OK: ${avail}GB free (need ${needed_gb}GB for ${label})"
}

start_server() {
    echo "==> Starting vLLM server..."
    mkdir -p "$(dirname "$SERVER_LOG")"
    nohup bash math/benchmark/serve.sh > "$SERVER_LOG" 2>&1 &
    echo $! > "$SERVER_PID_FILE"
    echo "    Server PID: $(cat $SERVER_PID_FILE) | log: $SERVER_LOG"

    echo -n "    Waiting for server to be ready"
    for i in $(seq 1 120); do
        if $PYTHON -c "import requests; requests.get('http://localhost:8000/health', timeout=3).raise_for_status()" 2>/dev/null; then
            echo " ready!"
            return 0
        fi
        echo -n "."
        sleep 5
    done
    echo ""
    echo "ERROR: Server did not become ready after 10 minutes. Check $SERVER_LOG"
    exit 1
}

stop_server() {
    echo "==> Stopping vLLM server..."
    if [ -f "$SERVER_PID_FILE" ]; then
        local pid
        pid=$(cat "$SERVER_PID_FILE")
        kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
        rm -f "$SERVER_PID_FILE"
        sleep 10
        echo "    Server stopped."
    else
        echo "    No PID file found, skipping."
    fi
}

# ── main loop ─────────────────────────────────────────────────────────────────

echo "======================================================"
echo "  RLM Loop: ${ITERATIONS} iterations, ${TRAIN_EPOCHS} epochs/iter"
echo "  $(date)"
echo "======================================================"

for ITER in $(seq 1 $ITERATIONS); do
    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  ITERATION ${ITER} / ${ITERATIONS}  —  $(date)"
    echo "══════════════════════════════════════════════════════"

    DATA_PATH="math/data/iter_${ITER}_corrections.jsonl"
    CKPT_DIR="math/checkpoints/iter_${ITER}"
    RESPONSES_PATH="math/benchmark/results/iter_${ITER}_responses.jsonl"

    # ── 1. Start server ──────────────────────────────────────────────────────
    start_server

    # ── 2. Inference on test set ─────────────────────────────────────────────
    echo ""
    echo "--> [${ITER}] Inference (limit=${INFER_LIMIT}, batch=${INFER_BATCH})..."
    RESPONSES_PATH="$RESPONSES_PATH" \
    $PYTHON math/benchmark/infer.py \
        --limit "$INFER_LIMIT" \
        --batch-size "$INFER_BATCH"

    # ── 3. Evaluate ──────────────────────────────────────────────────────────
    echo ""
    echo "--> [${ITER}] Evaluating..."
    $PYTHON math/benchmark/evaluate.py

    # ── 4. Generate corrections ───────────────────────────────────────────────
    echo ""
    echo "--> [${ITER}] Generating corrections (target=${GEN_TARGET}, batch=${GEN_BATCH})..."
    $PYTHON math/generate_corrections.py \
        --target "$GEN_TARGET" \
        --batch-size "$GEN_BATCH"

    # Copy to per-iteration path for record keeping
    cp math/data/corrections.jsonl "$DATA_PATH"

    # ── 5. Stop server before training ───────────────────────────────────────
    stop_server

    # ── 6. Space check before writing checkpoint ──────────────────────────────
    check_space "$MIN_FREE_GB" "iter_${ITER} checkpoint (~64GB)"

    # ── 7. Train ──────────────────────────────────────────────────────────────
    echo ""
    echo "--> [${ITER}] Training (epochs=${TRAIN_EPOCHS})..."
    $DEEPSPEED --num_gpus=8 \
        math/train.py \
        --epochs "$TRAIN_EPOCHS" \
        --output-dir "$CKPT_DIR" \
        --data "$DATA_PATH"

    echo ""
    echo "==> Iteration ${ITER} complete. Checkpoint: ${CKPT_DIR}"
    echo "    $(date)"
done

echo ""
echo "======================================================"
echo "  All ${ITERATIONS} iterations complete."
echo "  Checkpoints:"
for ITER in $(seq 1 $ITERATIONS); do
    echo "    math/checkpoints/iter_${ITER}/"
done
echo "  Generated data:"
for ITER in $(seq 1 $ITERATIONS); do
    echo "    math/data/iter_${ITER}_corrections.jsonl"
done
echo "  $(date)"
echo "======================================================"
