# Sentence-Level LM

## Download weights

```sh
huggingface-cli download bert-base-uncased --local-dir ./sentence-lm/bert_weights
huggingface-cli download gpt2 --local-dir ./sentence-lm/gpt2_weights
```

## Download data

```sh
python sentence-lm/download_data.py
```

## Train

Single GPU:
```sh
python sentence-lm/train.py
```

Multi-GPU (8x H200):
```sh
torchrun --nproc_per_node=8 sentence-lm/train.py
```

> Note: do not run `train.py` from an interactive Python shell — CUDA error 802 (system not yet initialized) will occur. Run as a script only.