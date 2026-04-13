import re
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, GPT2Tokenizer

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from model import HierarchicalLM

# ── Config ────────────────────────────────────────────────────────────────────
BERT_DIR         = "sentence-lm/bert_weights"
GPT2_DIR         = "sentence-lm/gpt2_weights"
DATA_PATH        = "sentence-lm/data/train.parquet"
CHECKPOINT_DIR   = Path("sentence-lm/checkpoints")

BATCH_SIZE       = 8          # per GPU
LR               = 3e-4
MAX_STEPS        = 100_000
GRAD_CLIP        = 1.0
LOG_EVERY        = 10

JEPA_WEIGHT      = 1.0
RECON_WEIGHT     = 1.0

MAX_SEGMENTS     = 20
MAX_BERT_LEN     = 64
MAX_GPT_LEN      = 64
MIN_SEG_TOKENS   = 8

# ── Segmentation ──────────────────────────────────────────────────────────────
_PUNCT_BOUNDARY = re.compile(r'(?<=[.?!;:,])\s+')


def split_into_segments(text: str, tokenizer: BertTokenizer, min_tokens: int = MIN_SEG_TOKENS) -> list[str]:
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
            "bert_ids":    bert_enc["input_ids"],           # (n_seg, L_bert)
            "bert_seg_ids": bert_enc["token_type_ids"],     # (n_seg, L_bert)
            "gpt_ids":     gpt_enc["input_ids"],            # (n_seg, L_gpt)
            "n_segments":  len(segments),
        }


def collate_fn(batch: list[dict]) -> dict:
    max_n  = max(item["n_segments"] for item in batch)
    B      = len(batch)
    L_bert = batch[0]["bert_ids"].shape[-1]
    L_gpt  = batch[0]["gpt_ids"].shape[-1]

    bert_ids     = torch.zeros(B, max_n, L_bert, dtype=torch.long)
    bert_seg_ids = torch.zeros(B, max_n, L_bert, dtype=torch.long)
    gpt_ids      = torch.zeros(B, max_n, L_gpt,  dtype=torch.long)
    n_segs       = torch.tensor([item["n_segments"] for item in batch])

    for i, item in enumerate(batch):
        n = item["n_segments"]
        bert_ids[i, :n]     = item["bert_ids"]
        bert_seg_ids[i, :n] = item["bert_seg_ids"]
        gpt_ids[i, :n]      = item["gpt_ids"]

    gpt_targets = torch.full((B, max_n, L_gpt), fill_value=-1, dtype=torch.long)
    gpt_targets[:, :, :-1] = gpt_ids[:, :, 1:]

    return {
        "bert_ids":     bert_ids,
        "bert_seg_ids": bert_seg_ids,
        "gpt_ids":      gpt_ids,
        "gpt_targets":  gpt_targets,
        "n_segments":   n_segs,
    }


# ── Lightning Module ──────────────────────────────────────────────────────────
class HierarchicalLMLit(L.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = HierarchicalLM.from_pretrained(BERT_DIR, GPT2_DIR)

    def forward(self, batch):
        return self.model(
            bert_ids=batch["bert_ids"],
            bert_seg_ids=batch["bert_seg_ids"],
            gpt_ids=batch["gpt_ids"],
            gpt_targets=batch["gpt_targets"],
            n_segments=batch["n_segments"],
            jepa_weight=JEPA_WEIGHT,
            recon_weight=RECON_WEIGHT,
        )

    def training_step(self, batch, batch_idx):
        loss, jepa_loss, recon_loss = self(batch)
        self.log("train/loss",  loss,       on_step=True, on_epoch=False, prog_bar=True,  sync_dist=True)
        self.log("train/jepa",  jepa_loss,  on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log("train/recon", recon_loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        return loss

    def on_after_backward(self):
        # EMA update after gradients are computed but before optimizer step.
        # Each rank has identical encoder weights (DDP syncs grads), so each
        # rank's EMA update is identical — no cross-rank sync needed.
        self.model.update_ema()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.model.encoder.parameters(),             "lr": LR * 0.1},
                {"params": self.model.decoder.transformer.parameters(), "lr": LR * 0.1},
                {"params": self.model.embed.parameters(),               "lr": LR},
                {"params": self.model.jepa_predictor.parameters(),      "lr": LR},
            ],
            weight_decay=0.01,
        )
        return optimizer


# ── Lightning DataModule ──────────────────────────────────────────────────────
class SegmentDataModule(L.LightningDataModule):

    def __init__(self):
        super().__init__()
        self.bert_tokenizer = None
        self.gpt_tokenizer  = None

    def setup(self, stage=None):
        self.bert_tokenizer = BertTokenizer.from_pretrained(BERT_DIR)
        self.gpt_tokenizer  = GPT2Tokenizer.from_pretrained(GPT2_DIR)
        self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
        self.dataset = SegmentDataset(DATA_PATH, self.bert_tokenizer, self.gpt_tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="ckpt_{step:06d}",
        every_n_train_steps=1000,
        save_top_k=-1,       # keep all checkpoints
    )

    trainer = L.Trainer(
        max_steps=MAX_STEPS,
        gradient_clip_val=GRAD_CLIP,
        log_every_n_steps=LOG_EVERY,
        callbacks=[checkpoint_cb],
        strategy=DDPStrategy(find_unused_parameters=False),
        precision="bf16-mixed",
        default_root_dir=str(CHECKPOINT_DIR),
    )

    lit_model  = HierarchicalLMLit()
    datamodule = SegmentDataModule()

    trainer.fit(lit_model, datamodule=datamodule)
