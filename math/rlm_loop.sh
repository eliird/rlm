#!/bin/bash
# RLM training loop: N iterations of
#   serve -> infer (1000) -> eval -> generate (10k) -> train (2 epochs) -> stop server
#
# Checkpoints : math/checkpoints/iter_{N}/
# Corrections : math/data/iter_{N}_corrections.jsonl
# Responses   : math/benchmark/results/iter_{N}_responses.jsonl
#
# Resumable: re-running skips steps whose output already exists on disk.
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
    local model_path="${1:-}"
    echo "==> Starting vLLM server (model: ${model_path:-base})..."
    mkdir -p "$(dirname "$SERVER_LOG")"
    nohup bash math/benchmark/serve.sh $model_path > "$SERVER_LOG" 2>&1 &
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
    pkill -f "vllm serve" 2>/dev/null || true
    rm -f "$SERVER_PID_FILE"
    # Wait until port 8000 is free
    for i in $(seq 1 30); do
        if ! fuser 8000/tcp &>/dev/null 2>&1; then
            echo "    Server stopped."
            return 0
        fi
        sleep 2
    done
    # Force kill if still alive
    fuser -k 8000/tcp &>/dev/null 2>&1 || true
    echo "    Server stopped (force killed)."
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
    EVAL_DIR="math/benchmark/results/iter_${ITER}_eval"

    NEED_SERVER=0
    [ ! -f "$RESPONSES_PATH" ] && NEED_SERVER=1
    [ ! -f "$DATA_PATH" ]      && NEED_SERVER=1

    # ── 1. Start server if needed ─────────────────────────────────────────────
    if [ "$NEED_SERVER" -eq 1 ]; then
        PREV_CKPT=""
        if [ "$ITER" -gt 1 ]; then
            PREV_CKPT="math/checkpoints/iter_$((ITER - 1))"
        fi
        start_server "$PREV_CKPT"
    else
        echo "    [skip] Server not needed (responses + corrections already exist)"
    fi

    # ── 2. Inference on test set ──────────────────────────────────────────────
    if [ ! -f "$RESPONSES_PATH" ]; then
        echo ""
        echo "--> [${ITER}] Inference (limit=${INFER_LIMIT}, batch=${INFER_BATCH})..."
        $PYTHON math/benchmark/infer.py \
            --limit "$INFER_LIMIT" \
            --batch-size "$INFER_BATCH" \
            --output "$RESPONSES_PATH"
    else
        echo "    [skip] Responses already exist: $RESPONSES_PATH"
    fi

    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    if [ ! -f "$EVAL_DIR/eval_summary.txt" ]; then
        echo ""
        echo "--> [${ITER}] Evaluating..."
        $PYTHON math/benchmark/evaluate.py \
            --input "$RESPONSES_PATH" \
            --output-dir "$EVAL_DIR"
    else
        echo "    [skip] Eval already exists: $EVAL_DIR/eval_summary.txt"
    fi

    # ── 4. Generate corrections ───────────────────────────────────────────────
    if [ ! -f "$DATA_PATH" ]; then
        echo ""
        echo "--> [${ITER}] Generating corrections (target=${GEN_TARGET}, batch=${GEN_BATCH})..."
        $PYTHON math/generate_corrections.py \
            --target "$GEN_TARGET" \
            --batch-size "$GEN_BATCH"
        mv math/data/corrections.jsonl "$DATA_PATH"
    else
        echo "    [skip] Corrections already exist: $DATA_PATH"
    fi

    # ── 5. Stop server if it was started ─────────────────────────────────────
    if [ "$NEED_SERVER" -eq 1 ]; then
        stop_server
    fi

    # ── 6. Train ──────────────────────────────────────────────────────────────
    if [ ! -f "$CKPT_DIR/config.json" ]; then
        check_space "$MIN_FREE_GB" "iter_${ITER} checkpoint (~64GB)"
        echo ""
        echo "--> [${ITER}] Training (epochs=${TRAIN_EPOCHS})..."
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $DEEPSPEED --num_gpus=8 \
            math/train.py \
            --epochs "$TRAIN_EPOCHS" \
            --output-dir "$CKPT_DIR" \
            --data "$DATA_PATH"
    else
        echo "    [skip] Checkpoint already exists: $CKPT_DIR"
    fi

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