import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from bert import BERT
from gpt import GPT
from embeddings import HierarchicalEmbedding, CausalAttentionMask


class HierarchicalLM(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder: BERT
        self.ema_encoder: BERT
        self.decoder: GPT
        self.embed: HierarchicalEmbedding
        self.jepa_predictor: nn.Sequential
        self._ema_decay: float

    @classmethod
    def from_pretrained(cls, bert_dir: str, gpt2_dir: str) -> "HierarchicalLM":
        model = cls()

        model.encoder = BERT.from_pretrained(bert_dir)
        model.decoder = GPT.from_local(gpt2_dir)

        # vocab_size must match the decoder's actual wte shape (GPT-2 pads to 50304)
        actual_vocab_size = model.decoder.transformer.wte.weight.shape[0]
        model.embed = HierarchicalEmbedding(d_model=768, vocab_size=actual_vocab_size)

        # Copy decoder token embeddings into embed so both share the same starting point
        with torch.no_grad():
            model.embed.token_embedding.weight.copy_(model.decoder.transformer.wte.weight)

        # Identity init for cls_proj so CLS vectors pass through unchanged at the start
        with torch.no_grad():
            nn.init.eye_(model.embed.cls_proj.weight)
            nn.init.zeros_(model.embed.cls_proj.bias)

        model.jepa_predictor = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Linear(768, 768),
        )

        model._ema_decay = 0.999

        # EMA encoder: frozen deep copy of the online encoder
        model.ema_encoder = copy.deepcopy(model.encoder)
        for p in model.ema_encoder.parameters():
            p.requires_grad_(False)

        def _count(m):
            return sum(p.numel() for p in m.parameters())

        print(f"encoder:        {_count(model.encoder)/1e6:.2f}M")
        print(f"ema_encoder:    {_count(model.ema_encoder)/1e6:.2f}M  (frozen)")
        print(f"decoder:        {_count(model.decoder)/1e6:.2f}M")
        print(f"embed:          {_count(model.embed)/1e6:.2f}M")
        print(f"jepa_predictor: {_count(model.jepa_predictor)/1e6:.2f}M")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"total:          {total/1e6:.2f}M  ({trainable/1e6:.2f}M trainable)")

        return model

    @torch.no_grad()
    def update_ema(self):
        decay = self._ema_decay
        for ema_p, enc_p in zip(self.ema_encoder.parameters(), self.encoder.parameters()):
            ema_p.data.mul_(decay).add_(enc_p.data, alpha=1.0 - decay)

    def forward(
        self,
        bert_ids: torch.Tensor,       # (B, N, L_bert)
        bert_seg_ids: torch.Tensor,   # (B, N, L_bert)
        gpt_ids: torch.Tensor,        # (B, N, L_gpt)
        gpt_targets: torch.Tensor,    # (B, N, L_gpt)  shifted by 1, pad=-1
        n_segments: torch.Tensor,     # (B,)
        jepa_weight: float = 1.0,
        recon_weight: float = 1.0,
    ):
        B, N, L_bert = bert_ids.shape
        device = bert_ids.device

        # === Step A: Encode all segments with the online encoder ===
        flat_ids = bert_ids.view(B * N, L_bert)
        flat_seg = bert_seg_ids.view(B * N, L_bert)
        all_cls = self.encoder(flat_ids, flat_seg)[:, 0].view(B, N, 768)
        # all_cls: (B, N, 768)

        # === Step B: EMA encoder — no grad ===
        with torch.no_grad():
            ema_cls = self.ema_encoder(flat_ids, flat_seg)[:, 0].view(B, N, 768)
        # ema_cls: (B, N, 768)

        # === Step C: JEPA loss — predict next segment CLS ===
        if N > 1:
            pred_input = all_cls[:, :-1, :].reshape(B * (N - 1), 768)
            pred_output = self.jepa_predictor(pred_input)              # (B*(N-1), 768)
            jepa_targets = ema_cls[:, 1:, :].reshape(B * (N - 1), 768)

            # Mask out pairs where segment n+1 doesn't exist
            valid_pairs = (
                torch.arange(N - 1, device=device).unsqueeze(0) < (n_segments - 1).unsqueeze(1)
            )  # (B, N-1)

            mse = F.mse_loss(pred_output, jepa_targets.detach(), reduction="none")  # (B*(N-1), 768)
            mse = mse.view(B, N - 1, 768).mean(-1)  # (B, N-1)
            jepa_loss = (mse * valid_pairs).sum() / valid_pairs.sum().clamp(min=1)
        else:
            jepa_loss = all_cls.new_zeros(1).squeeze()

        # === Step D: Reconstruction loss ===
        recon_loss_total = all_cls.new_zeros(1).squeeze()
        recon_count = 0
        T = gpt_ids.shape[2]

        for k in range(1, N):
            cls_prefix = all_cls[:, :k, :]  # (B, k, 768)

            tok_ids = gpt_ids[:, k, :]       # (B, T)
            tok_tgts = gpt_targets[:, k, :]  # (B, T)

            # Build hierarchical decoder input
            decoder_input, _, _ = self.embed(cls_prefix, tok_ids)
            # decoder_input: (B, k+T, 768)

            # Hierarchical causal mask: CLS attends causally to CLS, tokens attend to all CLS + causal tokens
            mask = CausalAttentionMask.build(k, T, device=device)  # (k+T, k+T) bool

            logits, _ = self.decoder(inputs_embeds=decoder_input, attn_mask=mask)
            # logits: (B, k+T, vocab_size) — but we only want token positions

            token_logits = logits[:, k:, :]  # (B, T, vocab_size)

            # Only compute loss for documents that have a segment k
            valid = k < n_segments  # (B,) bool
            if valid.any():
                loss_k = F.cross_entropy(
                    token_logits[valid].reshape(-1, token_logits.size(-1)),
                    tok_tgts[valid].reshape(-1),
                    ignore_index=-1,
                )
                recon_loss_total = recon_loss_total + loss_k
                recon_count += 1

        recon_loss = recon_loss_total / max(recon_count, 1)

        # === Step E: Combined loss ===
        loss = jepa_weight * jepa_loss + recon_weight * recon_loss
        return loss, jepa_loss.detach(), recon_loss.detach()


if __name__ == "__main__":
    # Smoke test with random tensors (no weights needed)
    import sys

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    B, N, L_bert, L_gpt = 2, 4, 32, 32
    bert_ids = torch.randint(1, 30522, (B, N, L_bert)).to(device)
    bert_seg_ids = torch.zeros_like(bert_ids)
    gpt_ids = torch.randint(0, 50257, (B, N, L_gpt)).to(device)
    gpt_targets = torch.full_like(gpt_ids, -1)
    gpt_targets[:, :, :-1] = gpt_ids[:, :, 1:]
    n_segments = torch.tensor([N, N - 1], device=device)

    print("Loading model from pretrained weights...")
    model = HierarchicalLM.from_pretrained("sentence-lm/bert_weights", "sentence-lm/gpt2_weights")
    model = model.to(device)
    model.train()

    loss, jepa_loss, recon_loss = model(
        bert_ids=bert_ids,
        bert_seg_ids=bert_seg_ids,
        gpt_ids=gpt_ids,
        gpt_targets=gpt_targets,
        n_segments=n_segments,
    )
    print(f"loss={loss.item():.4f}  jepa={jepa_loss.item():.4f}  recon={recon_loss.item():.4f}")

    loss.backward()
    print("Backward pass OK")

    model.update_ema()
    print("EMA update OK")
    print("Smoke test passed.")
