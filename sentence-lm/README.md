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

## Analyse segment lengths

```sh
python sentence-lm/analyze_seg_lengths.py
```

## Generate

```sh
# default prompts
python sentence-lm/generate.py --checkpoint sentence-lm/checkpoints/ckpt_XXXXXX.ckpt

# custom prompt
python sentence-lm/generate.py --checkpoint sentence-lm/checkpoints/ckpt_XXXXXX.ckpt --prompt "The stock market crashed unexpectedly."

# use JEPA predictor to guide next-segment CLS
python sentence-lm/generate.py --checkpoint sentence-lm/checkpoints/ckpt_XXXXXX.ckpt --jepa
```

## Test context sensitivity

```sh
python sentence-lm/test_context_sensitivity.py --checkpoint sentence-lm/checkpoints/ckpt_XXXXXX.ckpt
```