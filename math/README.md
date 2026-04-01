# Math Error-Correction Finetuning

## Setup

All commands run from repo root using `.venv`:
```bash
.venv/bin/python math/datasets/download.py
```

The vLLM server must be running for inference and data generation:
```bash
bash math/benchmark/serve.sh
```

---

## Dataset

### Download raw datasets
```bash
.venv/bin/python math/datasets/download.py
```

### Build combined train/test split
```bash
.venv/bin/python math/datasets/build_combined.py
```

Output: `math/datasets/combined/train.parquet` and `math/datasets/combined/test.parquet`

---

## Benchmark

### 1. Start the vLLM server
```bash
bash math/benchmark/serve.sh
```

Wait for `Application startup complete` in the logs. Runs the model across all 8 GPUs via tensor parallelism on port 8000. Logs saved to `math/benchmark/logs/vllm_serve.log`.

Test the server:
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [{"role": "user", "content": "What is 2 + 2?"}],
    "max_tokens": 256
  }' | python -m json.tool
```

### 2. Run inference (resumable)
```bash
# Quick test
python math/benchmark/infer.py --limit 10 --reasoning low

# Full test set
python math/benchmark/infer.py --reasoning high --batch-size 64
```

Responses saved incrementally to `math/benchmark/results/responses.jsonl`. Re-running resumes from where it left off.

`--batch-size` controls concurrent requests to the vLLM server. 64 is recommended given available KV cache headroom.

### 3. Evaluate
```bash
python math/benchmark/evaluate.py
```

Can be run at any point, even mid-inference.

Output:
- `math/benchmark/results/eval.parquet` — full results with predicted answers and correct flags
- `math/benchmark/results/eval_summary.txt` — accuracy breakdown by source

---

## Data Generation

### Generate error-correction training data (resumable)
Requires vLLM server running.

```bash
# Quick test
python math/generate_corrections.py --target 50 --reasoning-attempt low --reasoning-reflect medium

# Full run (default: 10K corrections)
python math/generate_corrections.py
```

Defaults: `--target 10000`, `--batch-size 128`, `--reasoning-attempt high`, `--reasoning-reflect high`

Pipeline per problem:
1. Run model on problem → get attempt + answer
2. If correct → discard
3. If wrong → send reflection prompt with wrong reasoning + reference solution
4. Save problem-centric correction as SFT training example

Problems are sampled evenly across all sources (stratified round-robin). Stops when `--target` corrections are saved.

Output: `math/data/corrections.jsonl`

---

## Iterative Training Loop

The full pipeline runs in rounds. Each round uses the current model to generate corrections, finetunes on them, then evaluates. Repeat until accuracy plateaus.

```
Round N:
  1. python math/generate_corrections.py --target 10000   # generate from current model
  2. python math/train.py                                  # finetune
  3. python math/benchmark/infer.py --reasoning high       # run inference on test set
  4. python math/benchmark/evaluate.py                     # evaluate
  5. compare accuracy to previous round → continue or stop
```

Each round's corrections target the *current* model's failures. As easy errors get fixed, later rounds produce harder, more targeted training signal.

---

## Reasoning effort

`--reasoning` controls the model's thinking budget (`low`, `medium`, `high`). Use `low` for quick checks, `high` for data generation and final evaluation.
