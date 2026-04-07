# Math Error-Correction Finetuning

## Setup
```bash
.venv/bin/python math/datasets/download.py
.venv/bin/python math/datasets/build_combined.py
```

---

## Full Loop (3 iterations)
```bash
bash math/rlm_loop.sh
```

Checkpoints → `math/checkpoints/iter_N/`  
Generated data → `math/data/iter_N_corrections.jsonl`

---

## Individual Steps

### Server
```bash
bash math/benchmark/serve.sh
```

### Inference
```bash
python math/benchmark/infer.py --limit 1000 --batch-size 512 --output math/benchmark/results/responses.jsonl
```

### Evaluate
```bash
python math/benchmark/evaluate.py --input math/benchmark/results/responses.jsonl --output-dir math/benchmark/results
```

### Generate corrections
```bash
python math/generate_corrections.py --target 10000 --batch-size 512 --output math/data/corrections.jsonl
```

### Train
```bash
deepspeed --num_gpus=8 math/train.py \
  --epochs 2 \
  --output-dir math/checkpoints/iter_1 \
  --data math/data/corrections.jsonl
```


## Comparision

```sh
# Baseline:


bash math/benchmark/serve.sh
python math/benchmark/infer.py --limit 1000 --output math/benchmark/results/baseline_responses.jsonl
# stop server (Ctrl+C)
python math/benchmark/evaluate.py --input math/benchmark/results/baseline_responses.jsonl --output-dir math/benchmark/results/baseline

# Finetuned:
bash math/benchmark/serve.sh math/checkpoints/iter_1
python math/benchmark/infer.py --limit 1000 --output math/benchmark/results/iter1_responses.jsonl
# stop server (Ctrl+C)
python math/benchmark/evaluate.py --input math/benchmark/results/iter1_responses.jsonl --output-dir math/benchmark/results/iter1
Results will be in math/benchmark/results/baseline/eval_summary.txt and math/benchmark/results/iter1/eval_summary.txt.
```