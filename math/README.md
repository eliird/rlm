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

Serves `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` on port 8000 with tensor-parallel-size 2 across 2x H100s (~33GB per GPU). The `<think>` block is parsed from the response content by splitting on `</think>`.

Wait for `Application startup complete` before sending requests.

Test the server:
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-32b",
    "messages": [{"role": "user", "content": "What is 2 + 2?"}],
    "max_tokens": 256
  }' | python -m json.tool
```

Or run the generation test script:
```bash
python math/benchmark/test_generation.py
```

### 2. Run inference (resumable)
```bash
# Quick test
python math/benchmark/infer.py --limit 1000 --batch-size 2

# Full test set
python math/benchmark/infer.py --batch-size 64
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
python math/generate_corrections.py --target 50

# Full run (default: 10K corrections)
python math/generate_corrections.py
```

Defaults: `--target 10000`, `--batch-size 128`

Pipeline per problem:
1. Run model on problem → get attempt + answer
2. If correct → discard
3. If wrong → send reflection prompt with wrong attempt + reference solution
4. Model produces a fresh `<think>` block (reasoning that naturally identifies the pitfall) + clean solution
5. Save `think` and `response` separately as the SFT training example

The reflection prompt gives the model the wrong attempt and reference solution as context only. The output is written as if solving fresh — no references to "the correct solution" or "the wrong approach".

Problems are sampled evenly across all sources (stratified round-robin). Stops when `--target` corrections are saved.

Output: `math/data/corrections.jsonl`

---

## Training

### Finetune on corrections
Requires `adam-mini`:
```bash
.venv/bin/pip install adam-mini
```

Run with torchrun across all 8 GPUs:
```bash
torchrun --nproc_per_node=8 math/train.py
```

Each training example is formatted as:
```
<|User|> [problem]
<|Assistant|> <think>
[reasoning that identifies the pitfall and works through the problem carefully]
</think>
[clean step-by-step solution ending in \boxed{}]
```

Loss is computed only on the assistant turn. Checkpoints saved to `math/checkpoints/epoch_N/`.

---

## Iterative Training Loop

The full pipeline runs in rounds. Each round uses the current model to generate corrections, finetunes on them, then evaluates. Repeat until accuracy plateaus.

```
Round N:
  1. python math/generate_corrections.py --target 10000   # generate from current model
  2. torchrun --nproc_per_node=8 math/train.py            # finetune
  3. python math/benchmark/infer.py                       # run inference on test set
  4. python math/benchmark/evaluate.py                    # evaluate
  5. compare accuracy to previous round → continue or stop
```

Each round's corrections target the *current* model's failures. As easy errors get fixed, later rounds produce harder, more targeted training signal.
