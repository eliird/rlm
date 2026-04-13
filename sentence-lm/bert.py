import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import BertTokenizer


class BERTEmbedding(nn.Module):
    """BERT Embedding: token + learned positional + segment, followed by LayerNorm."""

    def __init__(self, vocab_size, embed_size, max_len=512, dropout=0.1):
        super().__init__()
        self.token = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position = nn.Embedding(max_len, embed_size)
        self.segment = nn.Embedding(2, embed_size)
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence, segment_label):
        pos_ids = torch.arange(sequence.size(1), device=sequence.device).unsqueeze(0)
        x = self.token(sequence) + self.position(pos_ids) + self.segment(segment_label)
        return self.dropout(self.norm(x))


class MultiHeadAttention(nn.Module):
    """Fused QKV multi-head attention with F.scaled_dot_product_attention."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.gelu(self.w1(x))))


class TransformerBlock(nn.Module):
    """Post-norm Transformer block (matches HF BERT): Attention → Residual → LN → FFN → Residual → LN"""

    def __init__(self, hidden, attn_heads, ff_hidden, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(hidden, attn_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.ff = FeedForward(hidden, ff_hidden, dropout)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.attn(x, mask=mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class BERT(nn.Module):
    """
    BERT: Bidirectional Encoder Representations from Transformers.
    Architecture matches HuggingFace bert-base-uncased for weight loading.
    Returns hidden states for all positions; use [:, 0] for CLS embedding.
    """

    def __init__(self, vocab_size=30522, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.embedding = BERTEmbedding(vocab_size, hidden, dropout=dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)]
        )

    def forward(self, x, segment_info=None):
        if segment_info is None:
            segment_info = torch.zeros_like(x)
        mask = (x > 0).unsqueeze(1).unsqueeze(2)
        x = self.embedding(x, segment_info)
        for block in self.blocks:
            x = block(x, mask)
        return x

    @classmethod
    def from_pretrained(cls, weights_dir):
        """Load weights from HuggingFace bert-base-uncased pytorch_model.bin."""
        weights_dir = Path(weights_dir)
        hf = torch.load(weights_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)

        model = cls()
        sd = model.state_dict()

        # Embeddings
        sd["embedding.token.weight"] = hf["bert.embeddings.word_embeddings.weight"]
        sd["embedding.position.weight"] = hf["bert.embeddings.position_embeddings.weight"]
        sd["embedding.segment.weight"] = hf["bert.embeddings.token_type_embeddings.weight"]
        sd["embedding.norm.weight"] = hf["bert.embeddings.LayerNorm.gamma"]
        sd["embedding.norm.bias"] = hf["bert.embeddings.LayerNorm.beta"]

        # Transformer blocks
        for i in range(model.n_layers):
            prefix = f"bert.encoder.layer.{i}"

            # Fuse Q, K, V into single QKV
            q_w = hf[f"{prefix}.attention.self.query.weight"]
            k_w = hf[f"{prefix}.attention.self.key.weight"]
            v_w = hf[f"{prefix}.attention.self.value.weight"]
            sd[f"blocks.{i}.attn.qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

            q_b = hf[f"{prefix}.attention.self.query.bias"]
            k_b = hf[f"{prefix}.attention.self.key.bias"]
            v_b = hf[f"{prefix}.attention.self.value.bias"]
            sd[f"blocks.{i}.attn.qkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

            # Attention output
            sd[f"blocks.{i}.attn.out.weight"] = hf[f"{prefix}.attention.output.dense.weight"]
            sd[f"blocks.{i}.attn.out.bias"] = hf[f"{prefix}.attention.output.dense.bias"]

            # Post-attention LayerNorm
            sd[f"blocks.{i}.norm1.weight"] = hf[f"{prefix}.attention.output.LayerNorm.gamma"]
            sd[f"blocks.{i}.norm1.bias"] = hf[f"{prefix}.attention.output.LayerNorm.beta"]

            # FFN
            sd[f"blocks.{i}.ff.w1.weight"] = hf[f"{prefix}.intermediate.dense.weight"]
            sd[f"blocks.{i}.ff.w1.bias"] = hf[f"{prefix}.intermediate.dense.bias"]
            sd[f"blocks.{i}.ff.w2.weight"] = hf[f"{prefix}.output.dense.weight"]
            sd[f"blocks.{i}.ff.w2.bias"] = hf[f"{prefix}.output.dense.bias"]

            # Post-FFN LayerNorm
            sd[f"blocks.{i}.norm2.weight"] = hf[f"{prefix}.output.LayerNorm.gamma"]
            sd[f"blocks.{i}.norm2.bias"] = hf[f"{prefix}.output.LayerNorm.beta"]

        model.load_state_dict(sd)
        return model


if __name__ == "__main__":

    tokenizer = BertTokenizer.from_pretrained("sentence-lm/bert_weights")
    model = BERT.from_pretrained("sentence-lm/bert_weights")
    model.eval()

    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can learn representations of language.",
        "asdf qwerty zxcv gibberish nonsense bloop.",
    ]

    for sent in sentences:
        tokens = tokenizer(sent, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(tokens["input_ids"], tokens["token_type_ids"])
        cls = out[:, 0]
        print(f"Text:     {sent}")
        print(f"Tokens:   {tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])}")
        print(f"CLS norm: {cls.norm().item():.4f}")
        print(f"CLS[:8]:  {cls[0, :8].tolist()}")
        print()