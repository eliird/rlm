import re
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, GPT2Tokenizer

from model import HierarchicalLM

# ── Config ────────────────────────────────────────────────────────────────────
BERT_DIR        = "sentence-lm/bert_weights"
GPT2_DIR        = "sentence-lm/gpt2_weights"
DATA_PATH       = "sentence-lm/data/train.parquet"
CHECKPOINT_DIR  = Path("sentence-lm/checkpoints")

BATCH_SIZE      = 8
LR              = 3e-4
MAX_ITERS       = 100_000
GRAD_CLIP       = 1.0
CHECKPOINT_EVERY = 1000
LOG_EVERY       = 10

JEPA_WEIGHT     = 1.0
RECON_WEIGHT    = 1.0

MAX_SEGMENTS    = 20
MAX_BERT_LEN    = 64
MAX_GPT_LEN     = 64
MIN_SEG_TOKENS  = 8

# ── Segmentation ──────────────────────────────────────────────────────────────
_PUNCT_BOUNDARY = re.compile(r'(?<=[.?!;:,])\s+')


def split_into_segments(text: str, tokenizer: BertTokenizer, min_tokens: int = MIN_SEG_TOKENS) -> list[str]:
    """Split text at pause boundaries; merge chunks shorter than min_tokens."""
    raw = _PUNCT_BOUNDARY.split(text.strip())
    raw = [s.strip() for s in raw if s.strip()]

    merged = []
    buffer = ""
    for seg in raw:
        candidate = (buffer + " " + seg).strip() if buffer else seg
        if len(tokenizer.tokenize(candidate)) >= min_tokens:
            merged.append(candidate)
            buffer = ""
        else:
            buffer = candidate

    if buffer:
        if merged:
            merged[-1] = merged[-1] + " " + buffer
        else:
            merged.append(buffer)

    return merged if merged else [text.strip()]


# ── Dataset ───────────────────────────────────────────────────────────────────
class SegmentDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        bert_tokenizer: BertTokenizer,
        gpt_tokenizer: GPT2Tokenizer,
        max_segments: int = MAX_SEGMENTS,
        max_bert_len: int = MAX_BERT_LEN,
        max_gpt_len: int = MAX_GPT_LEN,
    ):
        self.df = pd.read_parquet(parquet_path)
        self.bert_tok = bert_tokenizer
        self.gpt_tok = gpt_tokenizer
        self.max_segments = max_segments
        self.max_bert_len = max_bert_len
        self.max_gpt_len = max_gpt_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["text"]
        segments = split_into_segments(text, self.bert_tok)
        segments = segments[: self.max_segments]

        bert_enc = self.bert_tok(
            segments,
            max_length=self.max_bert_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        gpt_enc = self.gpt_tok(
            segments,
            max_length=self.max_gpt_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "bert_ids": bert_enc["input_ids"],          # (n_seg, L_bert)
            "bert_seg_ids": bert_enc["token_type_ids"], # (n_seg, L_bert)
            "gpt_ids": gpt_enc["input_ids"],            # (n_seg, L_gpt)
            "n_segments": len(segments),
        }


def collate_fn(batch: list[dict]) -> dict:
    max_n = max(item["n_segments"] for item in batch)
    B = len(batch)
    L_bert = batch[0]["bert_ids"].shape[-1]
    L_gpt = batch[0]["gpt_ids"].shape[-1]

    bert_ids    = torch.zeros(B, max_n, L_bert, dtype=torch.long)
    bert_seg_ids = torch.zeros(B, max_n, L_bert, dtype=torch.long)
    # GPT padding uses eos token id; targets use -1 as ignore_index
    gpt_ids     = torch.zeros(B, max_n, L_gpt, dtype=torch.long)
    n_segs      = torch.tensor([item["n_segments"] for item in batch])

    for i, item in enumerate(batch):
        n = item["n_segments"]
        bert_ids[i, :n]     = item["bert_ids"]
        bert_seg_ids[i, :n] = item["bert_seg_ids"]
        gpt_ids[i, :n]      = item["gpt_ids"]

    # Targets: shift right by 1, pad positions = -1 (ignored in CE)
    gpt_targets = torch.full((B, max_n, L_gpt), fill_value=-1, dtype=torch.long)
    gpt_targets[:, :, :-1] = gpt_ids[:, :, 1:]
    # Padded segment slots have gpt_ids = 0 (eos); their targets stay -1

    return {
        "bert_ids":     bert_ids,
        "bert_seg_ids": bert_seg_ids,
        "gpt_ids":      gpt_ids,
        "gpt_targets":  gpt_targets,
        "n_segments":   n_segs,
    }


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    bert_tokenizer = BertTokenizer.from_pretrained(BERT_DIR)
    gpt_tokenizer  = GPT2Tokenizer.from_pretrained(GPT2_DIR)
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

    print("Loading model...")
    model = HierarchicalLM.from_pretrained(BERT_DIR, GPT2_DIR).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(),              "lr": LR * 0.1},
            {"params": model.decoder.transformer.parameters(),  "lr": LR * 0.1},
            {"params": model.embed.parameters(),                "lr": LR},
            {"params": model.jepa_predictor.parameters(),       "lr": LR},
        ],
        weight_decay=0.01,
    )

    dataset = SegmentDataset(DATA_PATH, bert_tokenizer, gpt_tokenizer)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

    iter_num = 0
    running_loss = running_jepa = running_recon = 0.0

    for epoch in range(1000):
        for batch in loader:
            if iter_num >= MAX_ITERS:
                print("Training complete.")
                return

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            loss, jepa_loss, recon_loss = model(
                bert_ids=batch["bert_ids"],
                bert_seg_ids=batch["bert_seg_ids"],
                gpt_ids=batch["gpt_ids"],
                gpt_targets=batch["gpt_targets"],
                n_segments=batch["n_segments"],
                jepa_weight=JEPA_WEIGHT,
                recon_weight=RECON_WEIGHT,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            model.update_ema()
            optimizer.zero_grad(set_to_none=True)

            running_loss  += loss.item()
            running_jepa  += jepa_loss.item()
            running_recon += recon_loss.item()

            if iter_num % LOG_EVERY == 0:
                avg = lambda x: x / LOG_EVERY
                print(
                    f"iter {iter_num:6d} | "
                    f"loss {avg(running_loss):.4f} | "
                    f"jepa {avg(running_jepa):.4f} | "
                    f"recon {avg(running_recon):.4f}",
                    flush=True,
                )
                running_loss = running_jepa = running_recon = 0.0

            if iter_num > 0 and iter_num % CHECKPOINT_EVERY == 0:
                ckpt_path = CHECKPOINT_DIR / f"ckpt_{iter_num:06d}.pt"
                torch.save(
                    {
                        "iter_num": iter_num,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"Checkpoint saved: {ckpt_path}")

            iter_num += 1


if __name__ == "__main__":
    train()
