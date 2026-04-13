"""
Tests whether the hierarchical model actually uses prior sentence context.

Three probes:
  1. Shuffled context       -- replace real CLS prefix with CLS from a different document.
                               Loss should rise if context is being used.
  2. Zero context           -- replace CLS prefix with all-zeros.
                               Stronger ablation; loss should rise even more.
  3. JEPA coherence         -- check whether predicted next-segment CLS is closer
                               to the real next segment than to a random segment.

Run from repo root:
    python sentence-lm/test_context_sensitivity.py --checkpoint sentence-lm/checkpoints/ckpt_XXXXXX.ckpt
"""

import argparse
import sys
import re

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, GPT2Tokenizer
from datasets import load_dataset

sys.path.insert(0, "sentence-lm")
from model import HierarchicalLM
from embeddings import CausalAttentionMask
from train import (
    split_into_segments,
    SegmentDataset,
    collate_fn,
    BERT_DIR,
    GPT2_DIR,
    HF_DATASET,
    HF_SUBSET,
    MAX_SEGMENTS,
    MAX_BERT_LEN,
    MAX_GPT_LEN,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_DOCS = 256   # documents to evaluate over


# ── helpers ───────────────────────────────────────────────────────────────────

def recon_loss_with_cls(model, bert_ids, bert_seg_ids, gpt_ids, gpt_targets,
                         n_segments, cls_override=None):
    """
    Compute reconstruction loss only.
    If cls_override is given (B, N, 768) it replaces the encoder output.
    """
    B, N, L_bert = bert_ids.shape
    device = bert_ids.device

    flat_ids = bert_ids.view(B * N, L_bert)
    flat_seg = bert_seg_ids.view(B * N, L_bert)

    if cls_override is not None:
        all_cls = cls_override
    else:
        all_cls = model.encoder(flat_ids, flat_seg)[:, 0].view(B, N, 768)

    T = gpt_ids.shape[2]
    recon_loss_total = all_cls.new_zeros(1).squeeze()
    recon_count = 0

    for k in range(1, N):
        cls_prefix = all_cls[:, :k, :]
        tok_ids  = gpt_ids[:, k, :]
        tok_tgts = gpt_targets[:, k, :]

        decoder_input, _, _ = model.embed(cls_prefix, tok_ids)
        mask = CausalAttentionMask.build(k, T, device=device)
        logits, _ = model.decoder(inputs_embeds=decoder_input, attn_mask=mask)
        token_logits = logits[:, k:, :]

        valid = k < n_segments
        if valid.any():
            loss_k = F.cross_entropy(
                token_logits[valid].reshape(-1, token_logits.size(-1)),
                tok_tgts[valid].reshape(-1),
                ignore_index=-1,
            )
            recon_loss_total = recon_loss_total + loss_k
            recon_count += 1

    return (recon_loss_total / max(recon_count, 1)).item()


# ── probes ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def probe_context_sensitivity(model, dataset, n_docs):
    """Probe 1 & 2: shuffled vs zero context vs real context."""
    from torch.utils.data import DataLoader, Subset
    import random

    indices = random.sample(range(len(dataset)), min(n_docs, len(dataset)))
    loader  = DataLoader(
        Subset(dataset, indices),
        batch_size=16,
        collate_fn=collate_fn,
    )

    real_losses, shuffled_losses, zero_losses = [], [], []

    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        B = batch["bert_ids"].shape[0]

        flat_ids = batch["bert_ids"].view(B * MAX_SEGMENTS, MAX_BERT_LEN)
        flat_seg = batch["bert_seg_ids"].view(B * MAX_SEGMENTS, MAX_BERT_LEN)
        real_cls = model.encoder(flat_ids, flat_seg)[:, 0].view(B, MAX_SEGMENTS, 768)

        # Probe 1: real context
        l_real = recon_loss_with_cls(
            model, batch["bert_ids"], batch["bert_seg_ids"],
            batch["gpt_ids"], batch["gpt_targets"], batch["n_segments"],
            cls_override=real_cls,
        )

        # Probe 2: shuffled context — roll batch by 1 so each doc gets another doc's CLS
        shuffled_cls = torch.roll(real_cls, shifts=1, dims=0)
        l_shuf = recon_loss_with_cls(
            model, batch["bert_ids"], batch["bert_seg_ids"],
            batch["gpt_ids"], batch["gpt_targets"], batch["n_segments"],
            cls_override=shuffled_cls,
        )

        # Probe 3: zero context
        zero_cls = torch.zeros_like(real_cls)
        l_zero = recon_loss_with_cls(
            model, batch["bert_ids"], batch["bert_seg_ids"],
            batch["gpt_ids"], batch["gpt_targets"], batch["n_segments"],
            cls_override=zero_cls,
        )

        real_losses.append(l_real)
        shuffled_losses.append(l_shuf)
        zero_losses.append(l_zero)

    avg = lambda xs: sum(xs) / len(xs)
    r = avg(real_losses)
    s = avg(shuffled_losses)
    z = avg(zero_losses)

    print("\n── Probe 1 & 2: Context sensitivity ──")
    print(f"  Real context:     {r:.4f}")
    print(f"  Shuffled context: {s:.4f}   (delta: {s - r:+.4f})")
    print(f"  Zero context:     {z:.4f}   (delta: {z - r:+.4f})")
    if s > r:
        print("  [PASS] Model uses prior context — shuffled context hurts.")
    else:
        print("  [FAIL] Shuffled context does not increase loss. Model may be ignoring CLS prefix.")


@torch.no_grad()
def probe_jepa_coherence(model, dataset, n_docs):
    """
    Probe 3: does the JEPA predictor predict the next segment better than chance?

    For each consecutive pair (seg_k, seg_{k+1}):
      - predict CLS_{k+1} from CLS_k via jepa_predictor
      - measure cosine sim to real CLS_{k+1} (positive)
      - measure cosine sim to a random segment's CLS (negative)
    """
    from torch.utils.data import DataLoader, Subset
    import random

    indices = random.sample(range(len(dataset)), min(n_docs, len(dataset)))
    loader  = DataLoader(
        Subset(dataset, indices),
        batch_size=16,
        collate_fn=collate_fn,
    )

    pos_sims, neg_sims = [], []

    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        B, N = batch["bert_ids"].shape[:2]

        flat_ids = batch["bert_ids"].view(B * N, MAX_BERT_LEN)
        flat_seg = batch["bert_seg_ids"].view(B * N, MAX_BERT_LEN)
        with torch.no_grad():
            ema_cls = model.ema_encoder(flat_ids, flat_seg)[:, 0].view(B, N, 768)
            all_cls = model.encoder(flat_ids, flat_seg)[:, 0].view(B, N, 768)

        for k in range(N - 1):
            valid = (k + 1) < batch["n_segments"]
            if not valid.any():
                continue

            pred = model.jepa_predictor(all_cls[valid, k, :])         # (V, 768)
            target = ema_cls[valid, k + 1, :]                          # (V, 768)

            # random negative: shuffle the targets within the batch
            neg_idx = torch.randperm(target.shape[0], device=DEVICE)
            negative = target[neg_idx]

            pos_sim = F.cosine_similarity(pred, target).mean().item()
            neg_sim = F.cosine_similarity(pred, negative).mean().item()
            pos_sims.append(pos_sim)
            neg_sims.append(neg_sim)

    avg = lambda xs: sum(xs) / len(xs)
    p = avg(pos_sims)
    n = avg(neg_sims)

    print("\n── Probe 3: JEPA next-segment prediction ──")
    print(f"  Cosine sim to real next segment:   {p:.4f}")
    print(f"  Cosine sim to random segment:      {n:.4f}")
    print(f"  Gap: {p - n:+.4f}")
    if p > n:
        print("  [PASS] JEPA predictor targets real next segment better than random.")
    else:
        print("  [FAIL] JEPA predictor shows no preference for real next segment.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--n_docs", type=int, default=N_DOCS)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    from train import HierarchicalLMLit
    lit = HierarchicalLMLit.load_from_checkpoint(args.checkpoint)
    model = lit.model.to(DEVICE).eval()

    bert_tok = BertTokenizer.from_pretrained(BERT_DIR)
    gpt_tok  = GPT2Tokenizer.from_pretrained(GPT2_DIR)
    gpt_tok.pad_token = gpt_tok.eos_token

    hf_ds = load_dataset(HF_DATASET, name=HF_SUBSET, split="train")
    dataset = SegmentDataset(hf_ds, bert_tok, gpt_tok)
    print(f"Dataset: {len(dataset)} documents. Evaluating on {args.n_docs}.")

    probe_context_sensitivity(model, dataset, args.n_docs)
    probe_jepa_coherence(model, dataset, args.n_docs)


if __name__ == "__main__":
    main()
