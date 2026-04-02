"""
Finetune DeepSeek-R1-Distill-Qwen-32B on error-correction data.

Full finetune (no LoRA), FSDP across all available GPUs.
Freezes embed_tokens and lm_head — trains all transformer layers.

Run:
  accelerate launch --config_file math/fsdp_config.yaml math/train.py
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
CACHE_DIR = "/data/cache/huggingface/hub"
CORRECTIONS_PATH = Path("math/data/corrections.jsonl")
OUTPUT_DIR = Path("math/checkpoints")

MAX_LENGTH = 8192
BATCH_SIZE_PER_GPU = 1
GRAD_ACCUM_STEPS = 16
LR = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
MAX_EPOCHS = 3


def build_assistant_content(rec: dict) -> str:
    return f"<think>\n{rec['think'].strip()}\n</think>\n{rec['response'].strip()}"


class CorrectionDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if "think" in rec and "response" in rec and "\\boxed{" in rec["response"]:
                    self.examples.append(rec)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        rec = self.examples[idx]
        messages = [
            {"role": "user", "content": rec["problem"]},
            {"role": "assistant", "content": build_assistant_content(rec)},
        ]
        full = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        user_only = self.tokenizer.apply_chat_template(
            messages[:1], tokenize=True, add_generation_prompt=True
        )

        input_ids = full[: self.max_length]
        prefix_len = len(user_only)

        labels = [-100] * prefix_len + input_ids[prefix_len:]
        labels = labels[: self.max_length]

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        attention_mask = [1] * len(input_ids) + [0] * (self.max_length - len(input_ids))
        labels = labels + [-100] * (self.max_length - len(labels))
        input_ids = input_ids + [pad_id] * (self.max_length - len(input_ids))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    tokenizer.padding_side = "right"

    dataset = CorrectionDataset(CORRECTIONS_PATH, tokenizer, MAX_LENGTH)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
    )

    # Freeze embeddings and lm_head — train only transformer layers
    for name, param in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            param.requires_grad_(False)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=MAX_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE_PER_GPU,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        bf16=True,

        logging_steps=10,
        save_strategy="epoch",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        fsdp="full_shard auto_wrap",
        fsdp_config="math/fsdp_config.json",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
