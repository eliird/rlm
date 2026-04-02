"""
Finetune DeepSeek-R1-Distill-Qwen-32B on error-correction data.

Full finetune (no LoRA), FSDP across all available GPUs.
Freezes embed_tokens and lm_head — trains all transformer layers.

Requires adam-mini:
  .venv/bin/pip install adam-mini

Run from repo root with torchrun:
  torchrun --nproc_per_node=8 math/train.py

Or on fewer GPUs:
  torchrun --nproc_per_node=2 math/train.py
"""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
import functools

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
CACHE_DIR = "/data/cache/huggingface/hub"
CORRECTIONS_PATH = Path("math/data/corrections.jsonl")
OUTPUT_DIR = Path("math/checkpoints")

# Training hyperparameters
MAX_LENGTH = 4096
BATCH_SIZE_PER_GPU = 2
GRAD_ACCUM_STEPS = 16          # effective global batch = 2 * 8 * 16 = 256
LR = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
MAX_EPOCHS = 3


def build_assistant_content(rec: dict) -> str:
    """
    Format the assistant turn:
      <think> model's informed reasoning that naturally identifies pitfalls </think>
      clean step-by-step solution ending in \\boxed{}
    """
    return f"<think>\n{rec['think'].strip()}\n</think>\n{rec['response'].strip()}"


class CorrectionDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                # Filter: must have think block and boxed answer in response
                if "think" in rec and "response" in rec and "\\boxed{" in rec["response"]:
                    self.examples.append(rec)

        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"Loaded {len(self.examples)} corrections from {path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        rec = self.examples[idx]
        messages = [
            {"role": "user", "content": rec["problem"]},
            {"role": "assistant", "content": build_assistant_content(rec)},
        ]
        # Full conversation tokenized
        full = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        # User-only prefix to find where assistant response starts
        user_only = self.tokenizer.apply_chat_template(
            messages[:1], tokenize=True, add_generation_prompt=True
        )

        input_ids = full[: self.max_length]
        prefix_len = len(user_only)

        # Loss mask: -100 on prompt tokens, real token ids on assistant tokens
        labels = [-100] * prefix_len + input_ids[prefix_len:]
        labels = labels[: self.max_length]

        # Pad to max_length
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        attention_mask = [1] * len(input_ids) + [0] * (self.max_length - len(input_ids))
        labels = labels + [-100] * (self.max_length - len(labels))
        input_ids = input_ids + [pad_id] * (self.max_length - len(input_ids))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def setup_fsdp(model):
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2DecoderLayer},
    )
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,  # required for per-param learning rate / freezing
    )
    return model


def freeze_embeddings(model):
    """Freeze embed_tokens and lm_head — train only transformer layers."""
    for name, param in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            param.requires_grad_(False)


def main():
    # Init distributed
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"World size: {world_size}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    tokenizer.padding_side = "right"

    # Dataset
    dataset = CorrectionDataset(CORRECTIONS_PATH, tokenizer, MAX_LENGTH)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_PER_GPU,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )

    # Model — load in bf16 on CPU first, then FSDP will shard across GPUs
    if rank == 0:
        print(f"Loading model {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
    )
    freeze_embeddings(model)
    model = setup_fsdp(model)

    # Optimizer — Adam-Mini, falls back to AdamW if not installed
    try:
        from adam_mini import Adam_mini
        optimizer = Adam_mini(
            model,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            model_sharding=True,         # required with FSDP
            n_feature=model.config.hidden_size,
            n_head=model.config.num_attention_heads,
            n_kv_head=model.config.num_key_value_heads,
        )
        if rank == 0:
            print("Using Adam-Mini optimizer")
    except ImportError:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
        if rank == 0:
            print("adam-mini not installed, using AdamW. Install with: .venv/bin/pip install adam-mini")

    total_steps = len(loader) * MAX_EPOCHS // GRAD_ACCUM_STEPS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - WARMUP_STEPS)
    )

    # Training loop
    global_step = 0
    for epoch in range(MAX_EPOCHS):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Linear warmup
                if global_step <= WARMUP_STEPS:
                    lr_scale = global_step / WARMUP_STEPS
                    for pg in optimizer.param_groups:
                        pg["lr"] = LR * lr_scale

                if rank == 0 and global_step % 10 == 0:
                    print(f"epoch={epoch+1} step={global_step} loss={loss.item() * GRAD_ACCUM_STEPS:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

        if rank == 0:
            print(f"Epoch {epoch+1} done.")

        # Save checkpoint after each epoch
        dist.barrier()
        if rank == 0:
            ckpt_dir = OUTPUT_DIR / f"epoch_{epoch+1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            # Gather full model state on rank 0 for saving
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict = model.state_dict()
            model.config.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            torch.save(state_dict, ckpt_dir / "model.pt")
            print(f"Checkpoint saved to {ckpt_dir}")
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
