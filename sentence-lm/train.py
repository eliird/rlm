import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, GPT2Tokenizer
from datasets import load_dataset

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.strategies import DDPStrategy

from model import HierarchicalLM

# ── Config ────────────────────────────────────────────────────────────────────
BERT_DIR         = "sentence-lm/bert_weights"
GPT2_DIR         = "sentence-lm/gpt2_weights"
HF_DATASET       = "HuggingFaceFW/fineweb-edu"
HF_SUBSET        = "sample-10BT"
CHECKPOINT_DIR   = Path("sentence-lm/checkpoints")

BATCH_SIZE       = 64       # per GPU
GLOBAL_BATCH     = 2048     # target global batch size; grad accum fills the gap
LR               = 3e-4
MAX_STEPS        = 100_000
WARMUP_FRACTION  = 0.05   # 5% of total steps
GRAD_CLIP        = 1.0
LOG_EVERY        = 10
VAL_BATCHES      = 64       # number of batches to use for validation

JEPA_WEIGHT      = 1.0
RECON_WEIGHT     = 1.0

MAX_SEGMENTS     = 64
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
        hf_dataset,
        bert_tokenizer: BertTokenizer,
        gpt_tokenizer: GPT2Tokenizer,
        max_segments: int = MAX_SEGMENTS,
        max_bert_len: int = MAX_BERT_LEN,
        max_gpt_len: int = MAX_GPT_LEN,
    ):
        self.ds = hf_dataset
        self.bert_tok = bert_tokenizer
        self.gpt_tok = gpt_tokenizer
        self.max_segments = max_segments
        self.max_bert_len = max_bert_len
        self.max_gpt_len = max_gpt_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        text = self.ds[idx]["text"]
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

    def __init__(self, warmup_steps: int = 200):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.model = torch.compile(HierarchicalLM.from_pretrained(BERT_DIR, GPT2_DIR))

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

    def validation_step(self, batch, batch_idx):
        loss, jepa_loss, recon_loss = self(batch)
        self.log("val/loss",  loss,       on_step=False, on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log("val/jepa",  jepa_loss,  on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/recon", recon_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
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
        warmup = self.warmup_steps
        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(MAX_STEPS - warmup, 1)
            return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}



# ── Generation callback ───────────────────────────────────────────────────────
GENERATION_PROMPTS = [
    "The researchers discovered a new species of deep-sea fish.",
    "Economic growth slowed in the third quarter as inflation remained high.",
    "She opened the letter and began to read.",
    "The algorithm failed to converge after one thousand iterations.",
]


class GenerationCallback(Callback):
    """Run greedy generation on a few fixed prompts at the end of each epoch."""

    def __init__(self, bert_tokenizer, gpt_tokenizer, n_segments=3, max_tokens=48, temperature=0.8, top_k=50):
        self.bert_tok    = bert_tokenizer
        self.gpt_tok     = gpt_tokenizer
        self.n_segments  = n_segments
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.top_k       = top_k

    def on_validation_epoch_end(self, trainer, pl_module):
        # only run on rank 0
        if trainer.global_rank != 0:
            return

        import re
        import torch.nn.functional as F
        from embeddings import CausalAttentionMask

        _split = re.compile(r'(?<=[.?!])\s+')
        model  = pl_module.model
        device = pl_module.device

        model.eval()
        print(f"\n{'='*70}")
        print(f"Generation samples — epoch {trainer.current_epoch}  step {trainer.global_step}")

        with torch.no_grad():
            for prompt in GENERATION_PROMPTS:
                prompt_segs = [s.strip() for s in _split.split(prompt.strip()) if s.strip()] or [prompt.strip()]

                enc = self.bert_tok(
                    prompt_segs, max_length=MAX_BERT_LEN, padding="max_length",
                    truncation=True, return_tensors="pt",
                )
                ids     = enc["input_ids"].to(device)
                seg_ids = enc["token_type_ids"].to(device)
                cls_vectors = model.encoder(ids, seg_ids)[:, 0].unsqueeze(0)  # (1, N, 768)

                print(f"\n  Prompt: {prompt}")
                for _ in range(self.n_segments):
                    k = cls_vectors.shape[1]
                    token_ids = torch.tensor([[self.gpt_tok.bos_token_id]], device=device)
                    for _ in range(self.max_tokens):
                        T = token_ids.shape[1]
                        decoder_input, _, _ = model.embed(cls_vectors, token_ids)
                        mask   = CausalAttentionMask.build(k, T, device=device)
                        logits, _ = model.decoder(inputs_embeds=decoder_input, attn_mask=mask)
                        next_logits = logits[0, -1, :] / self.temperature
                        topk_vals, _ = torch.topk(next_logits, self.top_k)
                        next_logits[next_logits < topk_vals[-1]] = float("-inf")
                        next_logits[self.gpt_tok.eos_token_id] = float("-inf")
                        probs      = F.softmax(next_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
                        token_ids  = torch.cat([token_ids, next_token], dim=1)

                    text = self.gpt_tok.decode(token_ids[0, 1:].tolist(), skip_special_tokens=True).strip()
                    print(f"    [{k+1}] {text}")

                    new_cls     = model.encoder(
                        self.bert_tok([text], max_length=MAX_BERT_LEN, padding="max_length",
                                      truncation=True, return_tensors="pt")["input_ids"].to(device),
                        torch.zeros(1, MAX_BERT_LEN, dtype=torch.long, device=device),
                    )[:, 0].unsqueeze(0)
                    cls_vectors = torch.cat([cls_vectors, new_cls], dim=1)

        print(f"{'='*70}\n")
        model.train()


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
        hf_ds    = load_dataset(HF_DATASET, name=HF_SUBSET, split="train")
        full_ds  = SegmentDataset(hf_ds, self.bert_tokenizer, self.gpt_tokenizer)
        val_size = VAL_BATCHES * BATCH_SIZE
        train_size = len(full_ds) - val_size
        self.train_dataset, self.val_dataset = random_split(
            full_ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    datamodule = SegmentDataModule()
    datamodule.setup()

    num_devices = torch.cuda.device_count() or 1
    accum = max(1, GLOBAL_BATCH // (BATCH_SIZE * num_devices))
    effective_batch = BATCH_SIZE * num_devices * accum
    print(f"devices: {num_devices}  per-GPU batch: {BATCH_SIZE}  "
          f"grad_accum: {accum}  effective global batch: {effective_batch}")

    steps_per_epoch = len(datamodule.train_dataset) // effective_batch
    warmup_steps = max(1, int(MAX_STEPS * WARMUP_FRACTION))
    warmup_steps = min(warmup_steps, steps_per_epoch)  # cap at 1 epoch
    print(f"steps_per_epoch: {steps_per_epoch}  warmup_steps: {warmup_steps}")

    checkpoint_cb = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="ckpt_{step:06d}",
        every_n_train_steps=10_000,
        save_top_k=-1,       # keep all checkpoints
    )

    gen_cb = GenerationCallback(
        bert_tokenizer=datamodule.bert_tokenizer,
        gpt_tokenizer=datamodule.gpt_tokenizer,
    )

    trainer = L.Trainer(
        max_steps=MAX_STEPS,
        gradient_clip_val=GRAD_CLIP,
        log_every_n_steps=LOG_EVERY,
        accumulate_grad_batches=accum,
        callbacks=[checkpoint_cb, gen_cb],
        strategy=DDPStrategy(find_unused_parameters=True),
        devices="auto",
        precision="bf16-mixed",
        default_root_dir=str(CHECKPOINT_DIR),
    )

    lit_model  = HierarchicalLMLit(warmup_steps=warmup_steps)

    trainer.fit(lit_model, datamodule=datamodule)
